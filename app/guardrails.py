"""序贯护栏：对二项失败率做 Wald 序贯概率比检验（SPRT）。

每个指标声明基线失败率 p0 = baseline_rate 与最大可接受劣化
p1 = p0 + max_increase，逐批累计对数似然比（LLR）：

    LLR += failures * ln(p1/p0) + (total - failures) * ln((1-p1)/(1-p0))

- LLR >= ln((1-beta)/alpha)  → rollback（劣化证据充分）；
- LLR <= ln(beta/(1-alpha))  → pass（无劣化证据充分）；
- 其余                        → hold（继续观察）。

任一指标首次越界后即冻结：不再消费后续批次，判定与计数保持不变。
总决策：任一指标 rollback → rollback（停止批次取各回滚指标中最早者）；
全部指标 pass → promote（停止批次取最后放行者）；其余 → hold。

铺开执行审计（app.rollout_audit）逐事件接收批次，复用本模块的阈值公式
与冻结语义：MetricAccumulator 为单个指标的流式累计器，
overall_decision 把各累计器的当前判定汇总为发布级总决策。
"""

from __future__ import annotations

import math

from app.models import (
    AuditGuardrailSpec,
    GuardrailDecision,
    GuardrailEvaluateRequest,
    GuardrailEvaluateResponse,
    GuardrailMetric,
    GuardrailMetricResult,
    GuardrailPathEntry,
    OverallDecision,
)


def evaluate_metric(metric: GuardrailMetric) -> GuardrailMetricResult:
    """对单个指标逐批累计 LLR，首次越界后冻结。"""
    p0 = metric.baseline_rate
    p1 = p0 + metric.max_increase
    lower = math.log(metric.beta / (1 - metric.alpha))
    upper = math.log((1 - metric.beta) / metric.alpha)
    # 单观测的 LLR 增量：失败计 ln(p1/p0)，成功计 ln((1-p1)/(1-p0))。
    log_fail = math.log(p1 / p0)
    log_ok = math.log((1 - p1) / (1 - p0))

    llr = 0.0
    failures = 0
    total = 0
    path: list[GuardrailPathEntry] = []
    decision = GuardrailDecision.HOLD
    stop_batch: int | None = None

    for index, batch in enumerate(metric.batches, start=1):
        llr += batch.failures * log_fail + (batch.total - batch.failures) * log_ok
        failures += batch.failures
        total += batch.total
        path.append(
            GuardrailPathEntry(
                batch=index,
                failures=batch.failures,
                total=batch.total,
                llr=llr,
            )
        )
        if llr >= upper:
            decision = GuardrailDecision.ROLLBACK
            stop_batch = index
            break
        if llr <= lower:
            decision = GuardrailDecision.PASS
            stop_batch = index
            break

    return GuardrailMetricResult(
        id=metric.id,
        decision=decision,
        stop_batch=stop_batch,
        failures=failures,
        total=total,
        p0=p0,
        p1=p1,
        lower_threshold=lower,
        upper_threshold=upper,
        llr=llr,
        path=path,
    )


def evaluate_guardrails(
    request: GuardrailEvaluateRequest,
) -> GuardrailEvaluateResponse:
    """汇总各指标判定为发布级总决策。"""
    results = [evaluate_metric(metric) for metric in request.metrics]

    rollback_batches = [
        r.stop_batch
        for r in results
        if r.decision == GuardrailDecision.ROLLBACK and r.stop_batch is not None
    ]
    if rollback_batches:
        decision = OverallDecision.ROLLBACK
        stop_batch: int | None = min(rollback_batches)
    elif all(r.decision == GuardrailDecision.PASS for r in results):
        decision = OverallDecision.PROMOTE
        # 全部 pass 时各指标均有停止批次，总停止批次取最后放行者。
        stop_batch = max(
            r.stop_batch for r in results if r.stop_batch is not None
        )
    else:
        decision = OverallDecision.HOLD
        stop_batch = None

    return GuardrailEvaluateResponse(
        decision=decision,
        stop_batch=stop_batch,
        metric_count=len(results),
        batch_count=len(request.metrics[0].batches),
        metrics=results,
    )


class MetricAccumulator:
    """单个指标的流式 Wald SPRT 累计器：一批一喂，首次越界后冻结。

    与 evaluate_metric 共用同一阈值/增量公式：frozen 为 True 后
    add_batch 不再改变任何状态（调用方据此判定 metric_frozen 违规）。
    """

    def __init__(self, spec: AuditGuardrailSpec) -> None:
        self.id = spec.id
        p0 = spec.baseline_rate
        p1 = p0 + spec.max_increase
        self._p0 = p0
        self._p1 = p1
        self.lower_threshold = math.log(spec.beta / (1 - spec.alpha))
        self.upper_threshold = math.log((1 - spec.beta) / spec.alpha)
        self._log_fail = math.log(p1 / p0)
        self._log_ok = math.log((1 - p1) / (1 - p0))

        self.llr = 0.0
        self.failures = 0
        self.total = 0
        self.path: list[GuardrailPathEntry] = []
        self.decision = GuardrailDecision.HOLD
        self.stop_batch: int | None = None

    @property
    def frozen(self) -> bool:
        """是否已在某一批首次越界（pass/rollback 均冻结）。"""
        return self.decision is not GuardrailDecision.HOLD

    def add_batch(self, failures: int, total: int) -> None:
        """消费一批增量并更新判定；冻结后的调用为空操作。"""
        if self.frozen:
            return
        self.llr += failures * self._log_fail + (total - failures) * self._log_ok
        self.failures += failures
        self.total += total
        batch_index = len(self.path) + 1
        self.path.append(
            GuardrailPathEntry(
                batch=batch_index,
                failures=failures,
                total=total,
                llr=self.llr,
            )
        )
        # 与离线评估一致：同批同时触及两侧时，rollback 优先（先检上阈值）。
        if self.llr >= self.upper_threshold:
            self.decision = GuardrailDecision.ROLLBACK
            self.stop_batch = batch_index
        elif self.llr <= self.lower_threshold:
            self.decision = GuardrailDecision.PASS
            self.stop_batch = batch_index

    def result(self) -> GuardrailMetricResult:
        """导出为与 /guardrails/evaluate 同构的指标结果（含似然路径）。"""
        return GuardrailMetricResult(
            id=self.id,
            decision=self.decision,
            stop_batch=self.stop_batch,
            failures=self.failures,
            total=self.total,
            p0=self._p0,
            p1=self._p1,
            lower_threshold=self.lower_threshold,
            upper_threshold=self.upper_threshold,
            llr=self.llr,
            path=list(self.path),
        )


def overall_decision(
    accumulators: list[MetricAccumulator],
) -> OverallDecision:
    """把各指标流式累计器的当前判定汇总为发布级总决策。

    任一指标 rollback → rollback；全部指标 pass → promote；其余 → hold。
    与离线评估不同：批次逐批到达时各指标批次数可能暂不相同，停止批次的
    汇总交给调用方按需计算，这里只给总判定。
    """
    if any(acc.decision is GuardrailDecision.ROLLBACK for acc in accumulators):
        return OverallDecision.ROLLBACK
    if all(acc.decision is GuardrailDecision.PASS for acc in accumulators):
        return OverallDecision.PROMOTE
    return OverallDecision.HOLD
