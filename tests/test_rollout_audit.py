"""POST /sdk/rollout-audit 测试：事件折叠、波次门控、逐波护栏与违规即停。"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from app.guardrails import MetricAccumulator
from app.main import app
from app.models import (
    AuditGuardrailSpec,
    RolloutEvent,
    RolloutEventType,
    RolloutPlanRequest,
)
from app.rollout import build_rollout_plan

URL = "/sdk/rollout-audit"
client = TestClient(app)

WS = RolloutEventType.WAVE_STARTED
NC = RolloutEventType.NODE_COMPLETED
MB = RolloutEventType.METRIC_BATCH

# 标准指标：p0=0.01, p1=0.02, alpha=0.05, beta=0.1。
P0, MI, ALPHA, BETA = 0.01, 0.01, 0.05, 0.1
UPPER = math.log((1 - BETA) / ALPHA)
LOWER = math.log(BETA / (1 - ALPHA))


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------


def make_plan(nodes: list[str] | None = None, *, parallel: int = 1, target: str = "new"):
    """默认两节点两波（parallel=1）的 planned 计划。"""
    nodes = nodes or ["a", "b"]
    request = RolloutPlanRequest(
        target_digest=target,
        nodes=[
            {
                "id": node_id,
                "current_digest": "old",
                "region": "r1",
                "failure_domain": f"fd-{node_id}",
                "traffic_weight": 1,
            }
            for node_id in nodes
        ],
        max_parallel=parallel,
        max_unavailable_weight=1_000_000,
        region_max_unavailable_weight={"r1": 1_000_000},
    )
    return build_rollout_plan(request)


def spec(id: str = "m", **overrides) -> AuditGuardrailSpec:
    params = dict(
        id=id,
        baseline_rate=P0,
        max_increase=MI,
        alpha=ALPHA,
        beta=BETA,
    )
    params.update(overrides)
    return AuditGuardrailSpec(**params)


def ev(eid: str, seq: int, type: RolloutEventType, **extra) -> RolloutEvent:
    return RolloutEvent(event_id=eid, sequence=seq, type=type, **extra)


def started(eid: str, seq: int, wave: int) -> RolloutEvent:
    return ev(eid, seq, WS, wave=wave)


def completed(eid: str, seq: int, node_id: str) -> RolloutEvent:
    return ev(eid, seq, NC, node_id=node_id)


def batch(
    eid: str,
    seq: int,
    wave: int,
    metric_id: str = "m",
    *,
    failures: int = 0,
    total: int = 100,
) -> RolloutEvent:
    return ev(eid, seq, MB, wave=wave, metric_id=metric_id, failures=failures, total=total)


def audit(events, *, plan=None, guardrails=None, status_code: int = 200) -> dict:
    plan = plan or make_plan()
    guardrails = guardrails if guardrails is not None else [spec()]
    payload = {
        "plan": plan.model_dump(mode="json"),
        "guardrails": [g.model_dump(mode="json") for g in guardrails],
        "events": [e.model_dump(mode="json", exclude_none=True) for e in events],
    }
    response = client.post(URL, json=payload)
    assert response.status_code == status_code, response.text
    return response.json()


def promote_batches(prefix: str, seq_start: int, wave: int, metric: str = "m"):
    """3 批全成功（0/100）即跌破下阈值 → 指标 pass；返回 (事件列表, 下一可用 seq)。"""
    events = [
        batch(f"{prefix}-{i}", seq_start + i, wave, metric, failures=0, total=100)
        for i in range(3)
    ]
    return events, seq_start + 3


def rollback_batch(eid: str, seq: int, wave: int, metric: str = "m"):
    # 20/100 单批 LLR ≈ 4.65 > 上阈值 2.89。
    return batch(eid, seq, wave, metric, failures=20, total=100)


def full_legal_stream(prefix: str = "e", *, waves: int = 2):
    """两波（每波单节点）完整合法流：启动→完成→3 批放行，逐波推进。"""
    events: list[RolloutEvent] = []
    seq = 0
    for wave in range(1, waves + 1):
        node = "ab"[wave - 1]
        seq += 1
        events.append(started(f"{prefix}-s{wave}", seq, wave))
        seq += 1
        events.append(completed(f"{prefix}-n{wave}", seq, node))
        new_batches, seq = promote_batches(f"{prefix}-w{wave}", seq + 1, wave)
        events.extend(new_batches)
    return events


# ---------------------------------------------------------------------------
# 合法日志：running / completed / 护栏累计
# ---------------------------------------------------------------------------


def test_empty_event_log_is_running():
    data = audit([])
    assert data["status"] == "running"
    assert data["violation"] is None
    assert data["current_wave"] is None
    assert data["promoted_waves"] == []
    assert data["completed_node_ids"] == []
    assert [w["wave"] for w in data["waves"]] == [1, 2]
    assert all(w["started"] is False for w in data["waves"])
    assert all(w["decision"] is None for w in data["waves"])
    assert all(w["metrics"] == [] for w in data["waves"])


def test_wave_started_state():
    data = audit([started("s1", 1, 1)])
    assert data["status"] == "running"
    assert data["current_wave"] == 1
    wave = data["waves"][0]
    assert wave["started"] is True
    assert wave["completed_node_ids"] == []
    # 启动即按声明建立全部指标累计器，但尚无路径。
    assert wave["decision"] == "hold"
    assert wave["metrics"][0]["id"] == "m"
    assert wave["metrics"][0]["decision"] == "hold"
    assert wave["metrics"][0]["path"] == []
    assert wave["metrics"][0]["failures"] == 0
    assert wave["metrics"][0]["total"] == 0
    # 第二波未启动。
    assert data["waves"][1]["started"] is False
    assert data["waves"][1]["metrics"] == []


def test_node_completed_after_wave_start():
    data = audit([started("s1", 1, 1), completed("n1", 2, "a")])
    assert data["status"] == "running"
    assert data["current_wave"] == 1
    assert data["waves"][0]["completed_node_ids"] == ["a"]


def test_hold_after_inconclusive_batch():
    events = [
        started("s1", 1, 1),
        completed("n1", 2, "a"),
        batch("b1", 3, 1, failures=1, total=100),
    ]
    data = audit(events)
    assert data["status"] == "running"
    metric = data["waves"][0]["metrics"][0]
    assert metric["decision"] == "hold"
    assert metric["stop_batch"] is None
    assert len(metric["path"]) == 1
    assert metric["path"][0]["batch"] == 1
    assert metric["failures"] == 1
    assert metric["total"] == 100


def test_completed_nodes_do_not_include_unpromoted_wave():
    """第二波节点在其 promote 前不计入全局 completed_node_ids。"""
    events = [
        started("s1", 1, 1),
        completed("n1", 2, "a"),
        *promote_batches("w1", 3, 1)[0],
        started("s2", 7, 2),
        completed("n2", 8, "b"),  # b 已完成，但第 2 波尚未 promote
        batch("hold", 9, 2, failures=1, total=100),
    ]
    data = audit(events)
    assert data["status"] == "running"
    assert data["current_wave"] == 2
    assert data["promoted_waves"] == [1]
    # 波内快照含 b，但全局累计完成在 promote 前不含 b。
    assert data["waves"][1]["completed_node_ids"] == ["b"]
    assert data["completed_node_ids"] == ["a"]


def test_promote_allows_next_wave_and_completes_status():
    data = audit(full_legal_stream())
    assert data["status"] == "completed"
    assert data["violation"] is None
    assert data["current_wave"] is None
    assert data["promoted_waves"] == [1, 2]
    assert data["completed_node_ids"] == ["a", "b"]
    for wave_state in data["waves"]:
        assert wave_state["started"] is True
        assert wave_state["decision"] == "promote"
        metric = wave_state["metrics"][0]
        assert metric["decision"] == "pass"
        assert metric["stop_batch"] == 3
        assert [entry["batch"] for entry in metric["path"]] == [1, 2, 3]


def test_each_wave_accumulates_guardrails_independently():
    """每波独立累计：第二波 LLR 从 0 起算，路径不含第一波批次。"""
    events = full_legal_stream()
    # 第一波第 1 批携带 1 个失败（不越界），第二波必须看不到该增量。
    events[2] = batch("e-w1-0", 3, 1, failures=1, total=100)
    data = audit(events)
    assert data["status"] == "completed"
    first = data["waves"][0]["metrics"][0]
    second = data["waves"][1]["metrics"][0]
    assert first["failures"] == 1 and first["total"] == 300
    assert second["failures"] == 0 and second["total"] == 300
    # 两波各 3 条路径，首条 LLR 不同（一波含失败、一波全成功）。
    assert len(second["path"]) == 3
    assert first["path"][0]["llr"] != second["path"][0]["llr"]
    assert second["path"][0]["llr"] == pytest.approx(100 * math.log(0.98 / 0.99))


def test_rollback_decision_freeze_and_status():
    events = [
        started("s1", 1, 1),
        completed("n1", 2, "a"),
        rollback_batch("bad", 3, 1),
    ]
    data = audit(events)
    assert data["status"] == "rollback_required"
    assert data["current_wave"] == 1
    assert data["promoted_waves"] == []
    wave_state = data["waves"][0]
    assert wave_state["decision"] == "rollback"
    metric = wave_state["metrics"][0]
    assert metric["decision"] == "rollback"
    assert metric["stop_batch"] == 1
    assert metric["llr"] >= UPPER
    assert len(metric["path"]) == 1


def test_multi_metric_all_pass_required_for_promote():
    plan = make_plan(["a"], parallel=1)  # 单波计划：全部指标 pass 即 completed
    guardrails = [spec("m1"), spec("m2")]
    g1, _ = promote_batches("g1", 3, 1, "m1")
    g2, _ = promote_batches("g2", 10, 1, "m2")
    events = [started("s1", 1, 1), completed("n1", 2, "a")] + g1 + g2
    data = audit(events, plan=plan, guardrails=guardrails)
    assert data["status"] == "completed"


def test_metrics_output_sorted_by_id():
    guardrails = [spec("zeta"), spec("alpha")]
    events = [started("s1", 1, 1), completed("n1", 2, "a")]
    data = audit(events, guardrails=guardrails)
    ids = [m["id"] for m in data["waves"][0]["metrics"]]
    assert ids == ["alpha", "zeta"]


def test_up_to_date_nodes_preseed_completed_and_need_no_events():
    plan = make_plan()  # 不用；自定义含已达目标节点的计划
    request = RolloutPlanRequest(
        target_digest="new",
        nodes=[
            {"id": "a", "current_digest": "old", "region": "r1",
             "failure_domain": "fd-a", "traffic_weight": 1},
            {"id": "b", "current_digest": "new", "region": "r1",
             "failure_domain": "fd-b", "traffic_weight": 1},
        ],
        max_parallel=1,
        max_unavailable_weight=10,
        region_max_unavailable_weight={"r1": 10},
    )
    plan = build_rollout_plan(request)
    assert plan.up_to_date_nodes == ["b"]
    w1_batches, _ = promote_batches("w1", 3, 1)
    events = [
        started("s1", 1, 1),
        completed("n1", 2, "a"),
        *w1_batches,
    ]
    data = audit(events, plan=plan)
    assert data["status"] == "completed"
    assert data["completed_node_ids"] == ["a", "b"]
    assert data["waves"][0]["completed_node_ids"] == ["a"]


def test_zero_wave_plan_completes_without_events():
    """全部节点已达目标（空计划）：无事件可执行，直接 completed。"""
    request = RolloutPlanRequest(
        target_digest="new",
        nodes=[
            {"id": "a", "current_digest": "new", "region": "r1",
             "failure_domain": "fd-a", "traffic_weight": 1},
        ],
        max_parallel=1,
        max_unavailable_weight=10,
        region_max_unavailable_weight={"r1": 10},
    )
    plan = build_rollout_plan(request)
    assert plan.waves == []
    data = audit([], plan=plan)
    assert data["status"] == "completed"
    assert data["completed_node_ids"] == ["a"]
    assert data["waves"] == []


# ---------------------------------------------------------------------------
# 折叠：按 sequence 排序、event_id 去重
# ---------------------------------------------------------------------------


def test_out_of_order_events_replay_by_sequence():
    """乱序提交等价于按 sequence 升序提交。"""
    ordered = [
        started("s1", 1, 1),
        completed("n1", 2, "a"),
        batch("b1", 3, 1, failures=1, total=100),
    ]
    shuffled = [ordered[2], ordered[0], ordered[1]]
    data_ordered = audit(ordered)
    data_shuffled = audit(shuffled)
    assert data_shuffled == data_ordered


def test_duplicate_event_id_same_content_is_idempotent():
    """同一 event_id 同内容重复上报：折叠为一条，合法流仍 completed。"""
    events = full_legal_stream()
    # 在不同位置插入与 seq=1 完全同内容的重复事件（新 sequence）。
    events.append(started("e-s1", 999, 1))
    data = audit(events)
    assert data["status"] == "completed"
    assert data["violation"] is None


def test_duplicate_event_id_different_content_is_422():
    events = [started("x", 1, 1), started("x", 2, 2)]
    audit(events, status_code=422)


def test_duplicate_sequence_is_422():
    events = [started("x", 1, 1), completed("y", 1, "a")]
    audit(events, status_code=422)


def test_sequence_need_not_be_contiguous():
    events = [
        started("s1", 10, 1),
        completed("n1", 20, "a"),
        batch("b1", 30, 1, failures=1, total=100),
    ]
    data = audit(events)
    assert data["status"] == "running"


# ---------------------------------------------------------------------------
# 违规：节点时序
# ---------------------------------------------------------------------------


def assert_violation(data, code, *, event_id: str):
    violation = data["violation"]
    assert violation is not None
    assert violation["code"] == code
    assert violation["event"]["event_id"] == event_id
    return violation


def test_node_completed_before_any_wave_started():
    data = audit([completed("n1", 1, "a")])
    assert_violation(data, "wave_not_started", event_id="n1")
    # 事前状态：无活动波、无完成节点。
    assert data["violation"]["before_state"]["current_wave"] is None
    assert data["violation"]["before_state"]["completed_node_ids"] == []


def test_node_completed_in_own_wave_requires_wave_started():
    # 第二波节点在第一波启动后完成：所属波未启动。
    events = [started("s1", 1, 1), completed("nb", 2, "b")]
    data = audit(events)
    assert_violation(data, "wave_not_started", event_id="nb")


def test_unknown_node_completed():
    events = [started("s1", 1, 1), completed("nz", 2, "zzz")]
    data = audit(events)
    assert_violation(data, "unknown_node", event_id="nz")


def test_duplicate_node_completed():
    events = [started("s1", 1, 1), completed("na", 2, "a"), completed("na2", 3, "a")]
    data = audit(events)
    assert_violation(data, "node_already_completed", event_id="na2")
    before = data["violation"]["before_state"]
    # 波尚未 promote：全局累计完成仍为空，完成记录在波内快照中。
    assert before["completed_node_ids"] == []
    assert before["waves"][0]["completed_node_ids"] == ["a"]


def test_up_to_date_node_completion_is_duplicate():
    request = RolloutPlanRequest(
        target_digest="new",
        nodes=[
            {"id": "a", "current_digest": "old", "region": "r1",
             "failure_domain": "fd-a", "traffic_weight": 1},
            {"id": "b", "current_digest": "new", "region": "r1",
             "failure_domain": "fd-b", "traffic_weight": 1},
        ],
        max_parallel=1,
        max_unavailable_weight=10,
        region_max_unavailable_weight={"r1": 10},
    )
    plan = build_rollout_plan(request)
    events = [completed("nb", 1, "b")]
    data = audit(events, plan=plan)
    assert_violation(data, "node_already_completed", event_id="nb")
    # 事前快照里已达目标节点一开始就计入完成。
    assert data["violation"]["before_state"]["completed_node_ids"] == ["b"]


# ---------------------------------------------------------------------------
# 违规：波次门控
# ---------------------------------------------------------------------------


def test_wave_started_for_unknown_wave():
    data = audit([started("s9", 1, 9)])
    assert_violation(data, "unknown_wave", event_id="s9")


def test_wave_started_twice():
    data = audit([started("s1", 1, 1), started("s1b", 2, 1)])
    assert_violation(data, "wave_already_started", event_id="s1b")


def test_wave_skipped():
    data = audit([started("s2", 1, 2)])
    assert_violation(data, "wave_skipped", event_id="s2")


def test_start_next_wave_while_hold_is_violation():
    events = [
        started("s1", 1, 1),
        completed("n1", 2, "a"),
        batch("b1", 3, 1, failures=1, total=100),  # hold
        started("s2", 4, 2),
    ]
    data = audit(events)
    assert_violation(data, "previous_wave_not_promoted", event_id="s2")
    before = data["violation"]["before_state"]
    assert before["current_wave"] == 1
    assert before["promoted_waves"] == []
    # 顶层状态与 waves 与事前快照一致。
    assert data["current_wave"] == 1
    assert data["waves"][0]["decision"] == "hold"


def test_next_wave_node_completed_while_hold_is_violation():
    events = [
        started("s1", 1, 1),
        completed("n1", 2, "a"),
        batch("b1", 3, 1, failures=1, total=100),  # hold
        completed("nb", 4, "b"),
    ]
    data = audit(events)
    assert_violation(data, "wave_not_started", event_id="nb")


def test_next_wave_metric_while_hold_is_violation():
    events = [
        started("s1", 1, 1),
        completed("n1", 2, "a"),
        batch("b1", 3, 1, failures=1, total=100),  # hold
        batch("b2w2", 4, 2),
    ]
    data = audit(events)
    assert_violation(data, "wave_not_started", event_id="b2w2")


def test_metric_batch_for_not_started_wave():
    # 第一波 promote 后第二波尚未启动即收指标：波未启动。
    w1_batches, next_seq = promote_batches("w1", 3, 1)
    events = [
        started("s1", 1, 1),
        completed("n1", 2, "a"),
        *w1_batches,
        batch("w2early", next_seq + 1, 2),
    ]
    data = audit(events)
    assert_violation(data, "wave_not_started", event_id="w2early")
    before = data["violation"]["before_state"]
    assert before["promoted_waves"] == [1]
    assert before["current_wave"] is None


# ---------------------------------------------------------------------------
# 违规：指标接收条件
# ---------------------------------------------------------------------------


def test_metric_before_wave_nodes_completed():
    events = [started("s1", 1, 1), batch("early", 2, 1)]
    data = audit(events)
    assert_violation(data, "metrics_before_wave_complete", event_id="early")


def test_unknown_metric_id():
    events = [
        started("s1", 1, 1),
        completed("n1", 2, "a"),
        batch("x", 3, 1, metric_id="nope"),
    ]
    data = audit(events)
    assert_violation(data, "unknown_metric", event_id="x")


def test_metric_batch_for_unknown_wave():
    events = [
        started("s1", 1, 1),
        completed("n1", 2, "a"),
        batch("x", 3, 7),
    ]
    data = audit(events)
    assert_violation(data, "unknown_wave", event_id="x")


def test_metric_frozen_after_pass_while_wave_holds():
    """某指标 pass 冻结但另一指标仍 hold：对已冻结指标再喂批次违规。"""
    guardrails = [spec("m1"), spec("m2")]
    g1, next_seq = promote_batches("g1", 3, 1, "m1")  # m1 第 3 批 pass
    events = (
        [started("s1", 1, 1), completed("n1", 2, "a")]
        + g1
        + [batch("g2-hold", next_seq + 1, 1, "m2", failures=1, total=100)]
        + [batch("g1-extra", next_seq + 2, 1, "m1")]
    )
    data = audit(events, guardrails=guardrails)
    assert_violation(data, "metric_frozen", event_id="g1-extra")
    before = data["violation"]["before_state"]
    assert before["status"] == "running"
    metrics = {m["id"]: m for m in before["waves"][0]["metrics"]}
    assert metrics["m1"]["decision"] == "pass"
    assert metrics["m2"]["decision"] == "hold"


def test_batch_after_wave_promoted_is_violation():
    """波 promote 后门控关闭：对该波再喂批次违规。"""
    w1_batches, next_seq = promote_batches("w1", 3, 1)
    events = [
        started("s1", 1, 1),
        completed("n1", 2, "a"),
        *w1_batches,
        batch("late", next_seq + 1, 1),
    ]
    data = audit(events)
    assert_violation(data, "wave_already_promoted", event_id="late")
    before = data["violation"]["before_state"]
    assert before["promoted_waves"] == [1]


def test_batch_after_rollback_is_execution_after_rollback():
    events = [
        started("s1", 1, 1),
        completed("n1", 2, "a"),
        rollback_batch("bad", 3, 1),
        batch("worse", 4, 1),
    ]
    data = audit(events)
    assert_violation(data, "execution_after_rollback", event_id="worse")
    before = data["violation"]["before_state"]
    # 事前状态：rollback 已发生。
    assert before["status"] == "rollback_required"
    assert before["waves"][0]["decision"] == "rollback"


# ---------------------------------------------------------------------------
# 违规：rollback 后任何执行
# ---------------------------------------------------------------------------


def test_node_completed_after_rollback():
    # 默认两波计划：第一波（仅 a）全部完成后护栏 rollback，
    # 再完成第二波节点 b 即“rollback 后执行”违规。
    events = [
        started("s1", 1, 1),
        completed("na", 2, "a"),
        rollback_batch("bad", 3, 1),
        completed("nb", 4, "b"),
    ]
    data = audit(events)
    assert_violation(data, "execution_after_rollback", event_id="nb")
    before = data["violation"]["before_state"]
    assert before["status"] == "rollback_required"
    # 波 1 未 promote：a 只在波内完成集合，全局累计完成仍为空。
    assert before["completed_node_ids"] == []
    assert before["waves"][0]["completed_node_ids"] == ["a"]


def test_wave_started_after_rollback():
    events = [
        started("s1", 1, 1),
        completed("n1", 2, "a"),
        rollback_batch("bad", 3, 1),
        started("s2", 4, 2),
    ]
    data = audit(events)
    assert_violation(data, "execution_after_rollback", event_id="s2")


def test_metric_for_later_wave_after_rollback():
    events = [
        started("s1", 1, 1),
        completed("n1", 2, "a"),
        rollback_batch("bad", 3, 1),
        batch("w2", 4, 2),
    ]
    data = audit(events)
    assert_violation(data, "execution_after_rollback", event_id="w2")


def test_first_violation_stops_replay():
    """首个违规即停止：其后的事件不产生任何状态影响。"""
    events = [
        completed("bad", 1, "a"),  # 违规：波未启动
        started("s1", 2, 1),       # 不再处理
        completed("n1", 3, "a"),   # 不再处理
    ]
    data = audit(events)
    assert_violation(data, "wave_not_started", event_id="bad")
    assert data["current_wave"] is None
    assert data["waves"][0]["started"] is False
    assert data["completed_node_ids"] == []


def test_violation_event_echoes_full_payload():
    events = [completed("bad", 7, "a")]
    data = audit(events)
    echo = data["violation"]["event"]
    assert echo == {
        "event_id": "bad",
        "sequence": 7,
        "type": "node_completed",
        "node_id": "a",
    }


# ---------------------------------------------------------------------------
# 计划本身的校验
# ---------------------------------------------------------------------------


def test_blocked_plan_cannot_be_audited():
    request = RolloutPlanRequest(
        target_digest="new",
        nodes=[
            {"id": "a", "current_digest": "old", "region": "r1",
             "failure_domain": "fd-a", "traffic_weight": 100},
        ],
        max_parallel=1,
        max_unavailable_weight=10,
        region_max_unavailable_weight={"r1": 10},
    )
    blocked = build_rollout_plan(request)
    assert blocked.status.value == "blocked"
    payload = {
        "plan": blocked.model_dump(mode="json"),
        "guardrails": [spec().model_dump()],
        "events": [],
    }
    response = client.post(URL, json=payload)
    assert response.status_code == 422


def test_tampered_wave_numbering_is_422():
    plan = make_plan()
    dumped = plan.model_dump(mode="json")
    dumped["waves"][1]["wave"] = 3  # 编号不再连续
    payload = {
        "plan": dumped,
        "guardrails": [spec().model_dump()],
        "events": [],
    }
    response = client.post(URL, json=payload)
    assert response.status_code == 422


def test_tampered_completed_list_is_422():
    plan = make_plan()
    dumped = plan.model_dump(mode="json")
    dumped["waves"][0]["completed_node_ids"] = ["a", "b", "ghost"]
    payload = {
        "plan": dumped,
        "guardrails": [spec().model_dump()],
        "events": [],
    }
    response = client.post(URL, json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 请求模型校验（422）
# ---------------------------------------------------------------------------


def test_event_shape_wave_started_rejects_node_id():
    with pytest.raises(ValueError):
        RolloutEvent(event_id="x", sequence=1, type=WS, wave=1, node_id="a")


def test_event_shape_node_completed_requires_node_id():
    with pytest.raises(ValueError):
        RolloutEvent(event_id="x", sequence=1, type=NC)


def test_event_shape_metric_batch_requires_fields():
    with pytest.raises(ValueError):
        RolloutEvent(event_id="x", sequence=1, type=MB, wave=1, metric_id="m")
    with pytest.raises(ValueError):
        RolloutEvent(event_id="x", sequence=1, type=MB, wave=1, metric_id="m",
                     failures=5, total=4)  # failures > total


def test_duplicate_guardrail_id_is_422():
    plan = make_plan()
    payload = {
        "plan": plan.model_dump(mode="json"),
        "guardrails": [spec("m").model_dump(), spec("m").model_dump()],
        "events": [],
    }
    response = client.post(URL, json=payload)
    assert response.status_code == 422


def test_extra_fields_rejected():
    plan = make_plan()
    payload = {
        "plan": plan.model_dump(mode="json"),
        "guardrails": [spec().model_dump()],
        "events": [],
        "unexpected": 1,
    }
    response = client.post(URL, json=payload)
    assert response.status_code == 422


def test_no_guardrails_is_422():
    plan = make_plan()
    payload = {"plan": plan.model_dump(mode="json"), "guardrails": [], "events": []}
    response = client.post(URL, json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 流式累计器与离线护栏一致性
# ---------------------------------------------------------------------------


def test_streaming_accumulator_matches_offline_guardrail():
    """逐批喂入的累计结果与 /guardrails/evaluate 离线计算完全一致。"""
    from app.guardrails import MetricAccumulator, evaluate_metric
    from app.models import GuardrailBatch, GuardrailMetric

    batches = [(1, 100), (0, 100), (3, 200), (0, 50)]
    metric_def = GuardrailMetric(
        id="m",
        baseline_rate=P0,
        max_increase=MI,
        alpha=ALPHA,
        beta=BETA,
        batches=[GuardrailBatch(failures=f, total=t) for f, t in batches],
    )
    offline = evaluate_metric(metric_def)

    accumulator = MetricAccumulator(spec())
    for failures, total in batches:
        accumulator.add_batch(failures, total)
    streamed = accumulator.result()

    assert streamed.decision == offline.decision
    assert streamed.stop_batch == offline.stop_batch
    assert streamed.failures == offline.failures
    assert streamed.total == offline.total
    assert streamed.llr == pytest.approx(offline.llr)
    assert [p.llr for p in streamed.path] == pytest.approx(
        [p.llr for p in offline.path]
    )


def test_streaming_accumulator_freezes_on_first_crossing():
    accumulator = MetricAccumulator(spec())
    accumulator.add_batch(20, 100)  # 越过上阈值
    assert accumulator.frozen
    llr_after_rollback = accumulator.llr
    path_len = len(accumulator.path)
    accumulator.add_batch(0, 100)  # 被冻结：空操作
    assert accumulator.llr == llr_after_rollback
    assert len(accumulator.path) == path_len
    assert accumulator.decision.value == "rollback"
