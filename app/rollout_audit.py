"""SDK 节点铺开执行审计：折叠事件流并按分波计划与序贯护栏重放。

事件可乱序提交，处理前先做两步折叠：

- 按 ``sequence`` 升序排列（同 sequence 与 event_id 内容冲突已在请求模型
  层拒绝，422）；
- 同一 event_id 只保留一条（内容完全一致的重复上报视为幂等重投）。

随后逐条重放 ``wave_started / node_completed / metric_batch``：

- 节点只能在**所属波启动后**完成；
- 每波**全部节点完成后**才可接收该波的指标批次；
- 各波**独立**复用序贯护栏（Wald SPRT）逐批累计 LLR，指标首次越界即
  冻结，冻结后再喂批次为 ``metric_frozen`` 违规；
- 护栏 ``promote``（全部指标 pass）后方可启动下一波；仍为 ``hold`` 时
  越波（启动/完成/上报后续波）违规；
- 任一指标判 ``rollback`` 后该波进入 rollback_required，此后任何执行
  事件（含 hold 下不可能出现、但重投的批次）都算
  ``execution_after_rollback``。

遇到首个违规即停止：响应给出违规事件与处理该事件**之前**的状态快照；
日志全部合法时给出各波完成节点、护栏累计路径与 running/completed/
rollback_required 总状态。
"""

from __future__ import annotations

from app.guardrails import MetricAccumulator, overall_decision
from app.models import (
    AuditGuardrailSpec,
    GuardrailMetricResult,
    OverallDecision,
    RolloutAuditRequest,
    RolloutAuditResponse,
    RolloutAuditStateSnapshot,
    RolloutAuditStatus,
    RolloutEvent,
    RolloutEventType,
    RolloutPlanResponse,
    RolloutStatus,
    RolloutViolation,
    RolloutViolationCode,
    RolloutWaveAuditState,
)


class RolloutAuditError(ValueError):
    """审计请求与原计划不一致（计划不可执行或被结构性篡改），对应 422。"""


class _WaveRuntime:
    """单个波次的可变运行态：启动标记、完成节点与各指标流式累计器。"""

    def __init__(self, spec_by_id: dict[str, AuditGuardrailSpec]) -> None:
        # 波启动时即为每个声明指标建一个累计器：promote 要求“全部”指标
        # pass，尚未收到任何批次的指标自然仍是 hold。
        self.started = False
        self.completed: set[str] = set()
        self.accumulators: list[MetricAccumulator] = [
            MetricAccumulator(spec) for spec in spec_by_id.values()
        ]

    def accumulator(self, metric_id: str) -> MetricAccumulator | None:
        for accumulator in self.accumulators:
            if accumulator.id == metric_id:
                return accumulator
        return None

    def decision(self) -> OverallDecision | None:
        if not self.started:
            return None
        return overall_decision(self.accumulators)


def audit_rollout(request: RolloutAuditRequest) -> RolloutAuditResponse:
    """按 sequence 折叠事件流并重放；首个违规即停止，返回事件与事前状态。"""
    plan = request.plan
    _validate_plan(plan)

    # 护栏声明按 ID 排序：各波累计器与响应 metrics 均以该顺序输出。
    specs = sorted(request.guardrails, key=lambda spec: spec.id)
    spec_by_id = {spec.id: spec for spec in specs}

    wave_count = len(plan.waves)
    wave_nodes = [set(wave.node_ids) for wave in plan.waves]
    # 计划内任一节点（含待更新与已达目标）的归属波次；已达目标节点无波次。
    node_wave: dict[str, int] = {}
    all_planned_nodes: set[str] = set(plan.up_to_date_nodes)
    for index, wave in enumerate(plan.waves):
        for node_id in wave.node_ids:
            node_wave[node_id] = index
            all_planned_nodes.add(node_id)

    runtimes = [_WaveRuntime(spec_by_id) for _ in plan.waves]
    # 已达目标节点无需更新，审计开始即计入累计完成节点。
    completed_nodes: set[str] = set(plan.up_to_date_nodes)
    current_wave: int | None = None  # 已启动且尚未 promote/rollback 的波（0 起始）
    promoted_waves: list[int] = []
    rolled_back = False

    # 折叠：按 sequence 升序；同 event_id 仅首条生效（内容一致性已在模型层校验）。
    unique_events: dict[str, RolloutEvent] = {}
    for event in request.events:
        unique_events.setdefault(event.event_id, event)
    ordered = sorted(unique_events.values(), key=lambda event: event.sequence)

    def snapshot_status() -> RolloutAuditStatus:
        if rolled_back:
            return RolloutAuditStatus.ROLLBACK_REQUIRED
        if len(promoted_waves) == wave_count:
            return RolloutAuditStatus.COMPLETED
        return RolloutAuditStatus.RUNNING

    def build_snapshot() -> RolloutAuditStateSnapshot:
        wave_states: list[RolloutWaveAuditState] = []
        for index, runtime in enumerate(runtimes):
            metrics: list[GuardrailMetricResult] = []
            if runtime.started:
                metrics = [acc.result() for acc in runtime.accumulators]
            wave_states.append(
                RolloutWaveAuditState(
                    wave=index + 1,
                    started=runtime.started,
                    completed_node_ids=sorted(runtime.completed),
                    decision=runtime.decision(),
                    metrics=metrics,
                )
            )
        return RolloutAuditStateSnapshot(
            status=snapshot_status(),
            current_wave=None if current_wave is None else current_wave + 1,
            promoted_waves=list(promoted_waves),
            completed_node_ids=sorted(completed_nodes),
            waves=wave_states,
        )

    for event in ordered:
        before = build_snapshot()

        if event.type is RolloutEventType.WAVE_STARTED:
            violation = _replay_wave_started(
                wave=event.wave - 1,
                wave_count=wave_count,
                started_count=sum(1 for runtime in runtimes if runtime.started),
                runtimes=runtimes,
                current_wave=current_wave,
                rolled_back=rolled_back,
            )
            if violation is not None:
                return _violation_response(plan, violation, event, before)
            target = event.wave - 1
            runtimes[target].started = True
            current_wave = target

        elif event.type is RolloutEventType.NODE_COMPLETED:
            violation = _replay_node_completed(
                node_id=event.node_id,
                node_wave=node_wave,
                all_planned_nodes=all_planned_nodes,
                runtimes=runtimes,
                current_wave=current_wave,
                completed_nodes=completed_nodes,
                rolled_back=rolled_back,
            )
            if violation is not None:
                return _violation_response(plan, violation, event, before)
            owner = node_wave[event.node_id]
            # 只记录到波内完成集合；全局累计完成在该波 promote 时才统一并入，
            # 与规划响应“逐波累计完成”的语义保持一致。
            runtimes[owner].completed.add(event.node_id)

        else:  # metric_batch
            violation = _replay_metric_batch(
                wave=event.wave - 1,
                metric_id=event.metric_id,
                wave_count=wave_count,
                wave_nodes=wave_nodes,
                runtimes=runtimes,
                current_wave=current_wave,
                spec_by_id=spec_by_id,
                rolled_back=rolled_back,
            )
            if violation is not None:
                return _violation_response(plan, violation, event, before)
            runtime = runtimes[event.wave - 1]
            runtime.accumulator(event.metric_id).add_batch(
                event.failures, event.total
            )

        # 事件重放后推进波次门控：rollback 锁死；全部指标 pass 即 promote，
        # 活动波清空，其节点全部计入累计完成（promote 隐含整波落地）。
        if current_wave is not None:
            decision = runtimes[current_wave].decision()
            if decision is OverallDecision.ROLLBACK:
                rolled_back = True
            elif decision is OverallDecision.PROMOTE:
                promoted_waves.append(current_wave + 1)
                completed_nodes.update(wave_nodes[current_wave])
                current_wave = None

    final_snapshot = build_snapshot()
    return RolloutAuditResponse(
        target_digest=plan.target_digest,
        status=final_snapshot.status,
        current_wave=final_snapshot.current_wave,
        promoted_waves=final_snapshot.promoted_waves,
        completed_node_ids=final_snapshot.completed_node_ids,
        waves=final_snapshot.waves,
        violation=None,
    )


def _replay_wave_started(
    *,
    wave: int,
    wave_count: int,
    started_count: int,
    runtimes: list[_WaveRuntime],
    current_wave: int | None,
    rolled_back: bool,
) -> RolloutViolationCode | None:
    """校验波启动事件；返回 None 表示合法（由调用方应用状态变更）。"""
    if rolled_back:
        return RolloutViolationCode.EXECUTION_AFTER_ROLLBACK
    if not 0 <= wave < wave_count:
        return RolloutViolationCode.UNKNOWN_WAVE
    if runtimes[wave].started:
        # 重放当前活动波、或更早（已 promote）的波都属于重复启动。
        return RolloutViolationCode.WAVE_ALREADY_STARTED
    if current_wave is not None:
        # 有未判定的活动波：启动任何尚未启动的后续波都是 hold 越波。
        return RolloutViolationCode.PREVIOUS_WAVE_NOT_PROMOTED
    # 无活动波：已启动的波必然都已 promote，下一波必须严格按顺序启动。
    if wave != started_count:
        return RolloutViolationCode.WAVE_SKIPPED
    return None


def _replay_node_completed(
    *,
    node_id: str,
    node_wave: dict[str, int],
    all_planned_nodes: set[str],
    runtimes: list[_WaveRuntime],
    current_wave: int | None,
    completed_nodes: set[str],
    rolled_back: bool,
) -> RolloutViolationCode | None:
    """校验节点完成事件；返回 None 表示合法。"""
    if rolled_back:
        return RolloutViolationCode.EXECUTION_AFTER_ROLLBACK
    if node_id in completed_nodes:
        # 已 promote 波的节点（含审计开始即已达目标的节点）：完成事件不可重放。
        return RolloutViolationCode.NODE_ALREADY_COMPLETED
    if node_id not in all_planned_nodes:
        return RolloutViolationCode.UNKNOWN_NODE
    owner = node_wave.get(node_id)
    if owner is None:
        # 计划内但无所属波（已达目标节点）：上面已按重复完成拦截，兜底拒绝。
        return RolloutViolationCode.NODE_NOT_IN_WAVE
    if current_wave is None or owner != current_wave:
        # 所属波尚未启动（活动波为空或另一波，后者即 hold 期间抢跑后续波）。
        return RolloutViolationCode.WAVE_NOT_STARTED
    if node_id in runtimes[owner].completed:
        # 同一波内重复回报完成。
        return RolloutViolationCode.NODE_ALREADY_COMPLETED
    return None


def _replay_metric_batch(
    *,
    wave: int,
    metric_id: str,
    wave_count: int,
    wave_nodes: list[set[str]],
    runtimes: list[_WaveRuntime],
    current_wave: int | None,
    spec_by_id: dict[str, AuditGuardrailSpec],
    rolled_back: bool,
) -> RolloutViolationCode | None:
    """校验指标批次事件；返回 None 表示合法。"""
    if rolled_back:
        return RolloutViolationCode.EXECUTION_AFTER_ROLLBACK
    if not 0 <= wave < wave_count:
        return RolloutViolationCode.UNKNOWN_WAVE
    if metric_id not in spec_by_id:
        return RolloutViolationCode.UNKNOWN_METRIC
    if not runtimes[wave].started:
        # 波尚未启动（抢跑，或上一波仍 hold 时上报后续波）。
        return RolloutViolationCode.WAVE_NOT_STARTED
    runtime = runtimes[wave]
    if not wave_nodes[wave] <= runtime.completed:
        # 波已启动但尚有节点未完成：指标只可在全节点完成后接收。
        return RolloutViolationCode.METRICS_BEFORE_WAVE_COMPLETE
    if current_wave != wave:
        # 波已 promote（活动波已清空或前移）：护栏门已关闭，
        # 对已放行波再喂批次按 wave_already_promoted 拒绝。
        return RolloutViolationCode.WAVE_ALREADY_PROMOTED
    accumulator = runtime.accumulator(metric_id)
    if accumulator.frozen:
        return RolloutViolationCode.METRIC_FROZEN
    return None


def _validate_plan(plan: RolloutPlanResponse) -> None:
    """校验审计所依据的计划：必须是 status=planned 且结构自洽。"""
    if plan.status is not RolloutStatus.PLANNED:
        raise RolloutAuditError("只可审计 status=planned 的铺开计划")
    if plan.blockers:
        raise RolloutAuditError("status=planned 的计划不应包含 blockers")

    all_wave_nodes: set[str] = set()
    up_to_date = set(plan.up_to_date_nodes)
    cumulative: set[str] = set(up_to_date)
    for index, wave in enumerate(plan.waves, start=1):
        if wave.wave != index:
            raise RolloutAuditError(f"波次编号必须从 1 连续递增：第 {index} 波标记为 {wave.wave}")
        ids = set(wave.node_ids)
        if len(ids) != len(wave.node_ids):
            raise RolloutAuditError(f"第 {index} 波的 node_ids 存在重复")
        overlap = ids & all_wave_nodes
        if overlap:
            raise RolloutAuditError(
                f"节点在多个波次中重复出现: {sorted(overlap)}"
            )
        overlap_up_to_date = ids & up_to_date
        if overlap_up_to_date:
            raise RolloutAuditError(
                f"已达目标节点不应再出现在波次中: {sorted(overlap_up_to_date)}"
            )
        all_wave_nodes.update(ids)
        cumulative.update(ids)
        if set(wave.completed_node_ids) != cumulative:
            raise RolloutAuditError(
                f"第 {index} 波的 completed_node_ids 与计划累计完成节点不一致"
            )
        if wave.completed_node_count != len(cumulative):
            raise RolloutAuditError(
                f"第 {index} 波的 completed_node_count 与 completed_node_ids 不一致"
            )

    if plan.planned_node_count != len(all_wave_nodes):
        raise RolloutAuditError("planned_node_count 与各波节点总数不一致")


def _violation_response(
    plan: RolloutPlanResponse,
    code: RolloutViolationCode,
    event: RolloutEvent,
    before: RolloutAuditStateSnapshot,
) -> RolloutAuditResponse:
    """构造首个违规响应：顶层状态取事前快照，并附违规事件与事前状态。"""
    return RolloutAuditResponse(
        target_digest=plan.target_digest,
        status=before.status,
        current_wave=before.current_wave,
        promoted_waves=before.promoted_waves,
        completed_node_ids=before.completed_node_ids,
        waves=before.waves,
        violation=RolloutViolation(
            code=code,
            message=_VIOLATION_MESSAGES[code],
            event=event.model_dump(mode="json", exclude_none=True),
            before_state=before,
        ),
    )


_VIOLATION_MESSAGES = {
    RolloutViolationCode.UNKNOWN_WAVE: "事件引用了计划中不存在的波次",
    RolloutViolationCode.WAVE_ALREADY_STARTED: "该波已启动，wave_started 不可重复",
    RolloutViolationCode.WAVE_SKIPPED: "存在波次被跳过：必须按波次顺序逐波启动",
    RolloutViolationCode.PREVIOUS_WAVE_NOT_PROMOTED: (
        "上一波护栏尚未 promote（仍为 hold），不得越波启动后续波"
    ),
    RolloutViolationCode.WAVE_NOT_STARTED: "所属波尚未启动（或仍有更早的波处于 hold）",
    RolloutViolationCode.WAVE_ALREADY_PROMOTED: "该波护栏已 promote，门控关闭后不得再喂批次",
    RolloutViolationCode.UNKNOWN_NODE: "事件引用了计划中不存在的节点",
    RolloutViolationCode.NODE_NOT_IN_WAVE: "节点不属于当前活动波（不得在他波执行期间完成）",
    RolloutViolationCode.NODE_ALREADY_COMPLETED: "该节点已完成，完成事件不可重复",
    RolloutViolationCode.METRICS_BEFORE_WAVE_COMPLETE: (
        "每波全部节点完成之后才可接收该波指标批次"
    ),
    RolloutViolationCode.UNKNOWN_METRIC: "指标批次引用了未声明的护栏指标",
    RolloutViolationCode.METRIC_FROZEN: "指标已在首次越界时冻结，不再消费后续批次",
    RolloutViolationCode.EXECUTION_AFTER_ROLLBACK: "护栏判定 rollback 后不得再执行任何事件",
}
