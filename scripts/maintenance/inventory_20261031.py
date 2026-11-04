"""Build the 2026-10-31 collection catalog from canonical recordings and reports.

Run with .venv-mi/Scripts/python.exe scripts/maintenance/inventory_20261031.py.
This one-off archive tool writes only data/recordings/_catalog/20261031/.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import csv
import json
import os
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DAY = "20261031"
TZ = timezone(timedelta(hours=8))
DESTINATION = ROOT / "data/recordings/_catalog/20261031"
TASK_NAMES = {"motor_execution": "实际握拳", "motor_imagery": "运动想象", "unspecified": "未记录任务类型"}
STOP_NAMES = {"target_met": "达到目标", "user_interrupted": "操作者停止",
              "stable_wait_timeout": "等待有效信号超时", "session_duration_limit": "达到时长上限",
              "attempt_limits_reached": "达到尝试次数上限", "capture_error": "采集错误"}
CENTRAL_CHANNELS = ("C3", "Cz", "C4")


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def stamp(value):
    return datetime.fromtimestamp(value, TZ).isoformat(timespec="milliseconds")


def filename_start(path):
    found = re.findall(r"(20261031)-(\d{6})", str(path))
    return datetime.strptime("".join(found[-1]), "%Y%m%d%H%M%S").replace(tzinfo=TZ).isoformat() if found else None


def recover_timing(events, elapsed):
    intervals = []
    opened = None
    for event in events:
        time = event["time_monotonic"]
        if event["type"] in {"disconnected", "forced_reconnect"} and opened is None:
            opened = time
        if opened is not None and event["type"] in {"recovered", "stopped", "failed"}:
            intervals.append(dict(start_monotonic=opened, end_monotonic=time,
                                  seconds=time-opened, end_event=event["type"], complete=True))
            opened = None
    lower = sum(row["seconds"] for row in intervals)
    upper = lower
    if opened is not None:
        observed = events[-1]["time_monotonic"]
        upper_end = events[0]["time_monotonic"] + elapsed if elapsed is not None else None
        prefix = max(0, observed-opened)
        intervals.append(dict(start_monotonic=opened, end_monotonic=None,
                              observed_through_monotonic=observed, seconds_lower=prefix,
                              seconds_upper=max(prefix, upper_end-opened) if upper_end is not None else None,
                              complete=False))
        lower += prefix
        upper = upper + max(prefix, upper_end-opened) if upper_end is not None else None
    connecting = next((e for e in events if e["type"] == "connecting"), None)
    connected = next((e for e in events if e["type"] == "connected"), None)
    return dict(post_failure_recovering_seconds_lower=lower,
                post_failure_recovering_seconds_upper=upper,
                exact=opened is None, intervals=intervals,
                initial_control_ready_seconds=(connected["time_monotonic"]-connecting["time_monotonic"]
                                               if connecting and connected else None),
                observed_event_span_seconds=(events[-1]["time_monotonic"]-events[0]["time_monotonic"] if events else None))


def constant_channel_check(epochs, baseline, channels, trials):
    """Count exact zero peak-to-peak windows without changing trial membership."""
    n = len(epochs)
    phases, masks = {}, {}
    for phase, array in (("baseline", baseline), ("task", epochs)):
        if array is None:
            phases[phase] = dict(available=False, constant_counts={name: None for name in channels},
                                nonfinite_counts={name: None for name in channels}, constant_values_uv={})
            masks[phase] = None
            continue
        with np.errstate(invalid="ignore"):
            constant = np.ptp(array, axis=-1) == 0
        finite = np.isfinite(array).all(axis=-1)
        masks[phase] = (constant, finite)
        phases[phase] = dict(available=True,
            constant_counts={name: int(constant[:, i].sum()) for i, name in enumerate(channels)},
            nonfinite_counts={name: int((~finite[:, i]).sum()) for i, name in enumerate(channels)},
            constant_values_uv={name: np.unique(array[constant[:, i], i, 0]).astype(float).tolist()
                                for i, name in enumerate(channels) if constant[:, i].any()})
    missing = [name for name in CENTRAL_CHANNELS if name not in channels]
    paired_available = baseline is not None and not missing
    evaluable = np.zeros(n, dtype=bool)
    nonconstant = np.zeros(n, dtype=bool)
    if paired_available:
        indices = [channels.index(name) for name in CENTRAL_CHANNELS]
        evaluable = masks["baseline"][1][:, indices].all(axis=1) & masks["task"][1][:, indices].all(axis=1)
        nonconstant = evaluable & ~masks["baseline"][0][:, indices].any(axis=1) & ~masks["task"][0][:, indices].any(axis=1)
    trial_checks = []
    for i in range(n):
        row = dict(stored_trial_index=i,
                   original_trial_index=trials[i].get("trial_index") if i < len(trials) else None,
                   paired_central_all_nonconstant=bool(nonconstant[i]) if evaluable[i] else None)
        for phase in ("baseline", "task"):
            row[f"{phase}_constant_channels"] = ([name for j, name in enumerate(channels) if masks[phase][0][i, j]]
                                                  if masks[phase] is not None else None)
            row[f"{phase}_nonfinite_channels"] = ([name for j, name in enumerate(channels) if not masks[phase][1][i, j]]
                                                   if masks[phase] is not None else None)
        trial_checks.append(row)
    return dict(method="np.ptp(array, axis=-1) == 0", scope="all_completed_epochs", n_trials=n,
                central_channels=list(CENTRAL_CHANNELS), missing_central_channels=missing,
                baseline=phases["baseline"], task=phases["task"],
                paired_central_all_nonconstant_trials=int(nonconstant.sum()) if paired_available else None,
                paired_central_evaluable_trials=int(evaluable.sum()),
                paired_central_with_constant_trials=int(np.count_nonzero(evaluable & ~nonconstant)),
                paired_central_unassessed_trials=int((~evaluable).sum()),
                missing_baseline_trials=n if baseline is None else 0,
                nonfinite_central_trials=int((~evaluable).sum()) if paired_available else None,
                trials=trial_checks)


reports = []
report_by_input = defaultdict(list)
for path in sorted((ROOT / "outputs/experiments").glob("**/report.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    if DAY not in str(path) and datetime.fromtimestamp(path.stat().st_mtime, TZ).strftime("%Y%m%d") != DAY:
        continue
    inputs = [str(Path(item["file"]).resolve()) for item in report.get("inputs", []) if item.get("file")]
    item = dict(path=relative(path), subject=report.get("subject"), task=report.get("task"),
                source_kind=report.get("source_kind"), n_trials=report.get("n_trials"),
                status=report.get("status", "analysis_available"), class_counts=report.get("class_counts"),
                input_files=inputs, filename_job_start_local=filename_start(path),
                benchmark_status=(report.get("benchmark") or {}).get("status"))
    reports.append(item)
    for name in inputs:
        report_by_input[name].append(relative(path))

records = []
skipped = []
for path in sorted((ROOT / "data/recordings").glob("**/*.npz")):
    in_day = DAY in path.name or datetime.fromtimestamp(path.stat().st_mtime, TZ).strftime("%Y%m%d") == DAY
    if not in_day:
        continue
    with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data:
            skipped.append(dict(path=relative(path), reason="metadata_json absent"))
            continue
        meta = json.loads(str(data["metadata_json"].item()))
        config = meta.get("configuration", {})
        summary = config.get("collection_summary", {})
        trial_summary = meta.get("trial_summary", {})
        events = meta.get("recovery_events", [])
        attempts = config.get("attempt_history", [])
        waits = config.get("stable_wait_history", [])
        rate = int(data["sample_rate"])
        raw = data["continuous"] if "continuous" in data else data["data"]
        epochs = data["epochs"] if "epochs" in data else np.empty((0, 0, 0))
        baseline = data["baseline_epochs"] if "baseline_epochs" in data else None
        channels = data["channel_names"].astype(str).tolist()
        labels = data["labels"].astype(str) if "labels" in data else np.empty(0, dtype=str)
        blocks = data["blocks"] if "blocks" in data else np.empty(0, dtype=int)
        artifact = data["artifact"].astype(bool) if "artifact" in data else np.zeros(len(epochs), dtype=bool)
        finite = np.isfinite(epochs).all(axis=(1, 2))
        if baseline is not None:
            finite &= np.isfinite(baseline).all(axis=(1, 2))
        usable = ~artifact & finite
        subject = meta["subject_id"]
        source = config.get("backend", meta.get("provenance", {}).get("backend"))
        elapsed = summary.get("elapsed_seconds")
        anchors = []
        for rows in (attempts, waits):
            for row in rows:
                if "started_at" in row and "started_monotonic" in row:
                    anchors.append(row["started_at"] - row["started_monotonic"])
        offset = float(np.median(anchors)) if anchors else None
        event_start = events[0]["time_monotonic"] if events else None
        last_stopped = next((e["time_monotonic"] for e in reversed(events) if e["type"] == "stopped"), None)
        if elapsed is None and event_start is not None and last_stopped is not None:
            elapsed = last_stopped-event_start
            elapsed_basis = "first_connecting_to_stopped_event"
        else:
            elapsed_basis = "collection_summary.elapsed_seconds" if elapsed is not None else None
        timing = recover_timing(events, summary.get("elapsed_seconds"))
        timing.update(stable_wait_seconds=sum(row.get("ended_monotonic", row["started_monotonic"])-row["started_monotonic"] for row in waits),
                      stable_wait_history_available="stable_wait_history" in config,
                      elapsed_seconds=elapsed, elapsed_basis=elapsed_basis,
                      elapsed_minus_raw_nominal_seconds=elapsed-raw.shape[1]/rate if elapsed is not None else None)
        task = config.get("task", "unspecified")
        task_hint = "motor_imagery" if task == "unspecified" and "imagery_seconds" in config else None
        stop = summary.get("stop_reason")
        stop_basis = "collection_summary.stop_reason" if stop else None
        if not stop and config.get("capture_interrupted"):
            stop, stop_basis = "user_interrupted", "configuration.capture_interrupted"
        if not stop and config.get("capture_error"):
            stop, stop_basis = "capture_error", "configuration.capture_error"
        report_paths = report_by_input.get(str(path.resolve()), [])
        if not report_paths and path.name.startswith("experiment-"):
            matches = [r["path"] for r in reports if Path(r["path"]).parent.name == path.stem[len("experiment-"):]]
            report_paths = matches
        record = dict(path=relative(path), filename_job_start_local=filename_start(path),
            source_started_local=stamp(offset+event_start) if offset is not None and event_start is not None else None,
            source_stopped_local=stamp(offset+last_stopped) if offset is not None and last_stopped is not None else None,
            time_anchor_count=len(anchors), time_anchor_spread_seconds=max(anchors)-min(anchors) if anchors else None,
            subject=subject, corrections=meta.get("corrections", []),
            session=meta.get("session_id"), task=task, task_hint_from_imagery_seconds=task_hint,
            backend=source, provenance=meta.get("provenance", {}), wear_state=config.get("wear_state"),
            primary_subject_scope=subject in {"S01", "S02", "S03", "S04"},
            sample_rate=rate, channels=channels,
            raw_samples=int(raw.shape[1]), raw_seconds=raw.shape[1]/rate,
            target_trials=summary.get("target_trials", trial_summary.get("planned", config.get("planned_trials"))),
            attempted_trials=summary.get("total_attempts", len(epochs)+len(meta.get("discarded_trials", []))),
            attempted_trials_basis="collection_summary.total_attempts" if summary else "completed_plus_discarded",
            complete_trials=len(epochs), usable_trials=int(usable.sum()),
            constant_channel_check=constant_channel_check(epochs, baseline, channels, meta.get("trials", [])),
            complete_class_counts=dict(Counter(labels.tolist())), usable_class_counts=dict(Counter(labels[usable].tolist())),
            complete_block_counts=dict(Counter(map(int, blocks.tolist()))), usable_block_counts=dict(Counter(map(int, blocks[usable].tolist()))),
            target_blocks=config.get("blocks"), artifact_trials=int(artifact.sum()),
            nonfinite_trials=int((~finite).sum()),
            discarded_trials=len(meta.get("discarded_trials", [])),
            discard_reason_counts=dict(Counter(d.get("reason", "unspecified") for d in meta.get("discarded_trials", []))),
            attempt_status_counts=dict(Counter(a.get("status", "unspecified") for a in attempts)),
            retry_count=summary.get("retry_count"), unattempted_trials=summary.get("unattempted_trials", trial_summary.get("unattempted")),
            stop_reason=stop, stop_reason_basis=stop_basis, capture_error=config.get("capture_error"),
            trial_summary_original=trial_summary, collection_summary_original=summary,
            recovery_stats=meta.get("recovery_stats", {}),
            recovery_event_type_counts=dict(Counter(e["type"] for e in events)),
            recovery_failure_reason_counts=dict(Counter(e.get("reason", "") for e in events if e["type"] in {"disconnected", "failed", "forced_reconnect"})),
            recovery_events=events, timing=timing, reports=report_paths)
        records.append(record)

records.sort(key=lambda r: r["filename_job_start_local"] or r["source_started_local"] or r["path"])
hardware = [r for r in records if r["primary_subject_scope"] and r["backend"] == "brainflow"]
other = [r for r in records if r not in hardware]
explicit_tasks = [r for r in hardware if r["task"] in {"motor_execution", "motor_imagery"}]
legacy = [r for r in hardware if r["task"] not in {"motor_execution", "motor_imagery"}]
matched_reports = {p for r in records for p in r["reports"]}
report_only = [r for r in reports if r["path"] not in matched_reports]


def aggregate(rows):
    reasons = Counter()
    stats = Counter()
    for row in rows:
        reasons.update(row["recovery_failure_reason_counts"])
        stats.update(row["recovery_stats"])
    return dict(n_recordings=len(rows), raw_samples=sum(r["raw_samples"] for r in rows),
        raw_seconds=sum(r["raw_seconds"] for r in rows), complete_trials=sum(r["complete_trials"] for r in rows),
        usable_trials=sum(r["usable_trials"] for r in rows), attempted_trials=sum(r["attempted_trials"] for r in rows),
        target_trials=sum(r["target_trials"] or 0 for r in rows),
        constant_channel_check=dict(
            paired_central_all_nonconstant_trials=(sum(r["constant_channel_check"]["paired_central_all_nonconstant_trials"] or 0 for r in rows)
                if any(r["constant_channel_check"]["paired_central_all_nonconstant_trials"] is not None for r in rows) else None),
            paired_central_evaluable_trials=sum(r["constant_channel_check"]["paired_central_evaluable_trials"] for r in rows),
            paired_central_with_constant_trials=sum(r["constant_channel_check"]["paired_central_with_constant_trials"] for r in rows),
            paired_central_unassessed_trials=sum(r["constant_channel_check"]["paired_central_unassessed_trials"] for r in rows),
            missing_baseline_trials=sum(r["constant_channel_check"]["missing_baseline_trials"] for r in rows),
            nonfinite_central_trials=sum(r["constant_channel_check"]["nonfinite_central_trials"] or 0 for r in rows),
            central_constant_counts={phase: {name: (sum(r["constant_channel_check"][phase]["constant_counts"].get(name) or 0 for r in rows)
                if any(r["constant_channel_check"][phase]["constant_counts"].get(name) is not None for r in rows) else None)
                                             for name in CENTRAL_CHANNELS} for phase in ("baseline", "task")}),
        recovery_stats=dict(stats), recovery_failure_reason_counts=dict(reasons),
        recovering_seconds_lower=sum(r["timing"]["post_failure_recovering_seconds_lower"] for r in rows),
        recovering_seconds_upper=(sum(r["timing"]["post_failure_recovering_seconds_upper"] for r in rows)
                                  if all(r["timing"]["post_failure_recovering_seconds_upper"] is not None for r in rows) else None),
        known_elapsed_seconds=sum(r["timing"]["elapsed_seconds"] or 0 for r in rows),
        recordings_without_elapsed=[r["path"] for r in rows if r["timing"]["elapsed_seconds"] is None])


def category(row):
    if row["backend"] != "brainflow":
        return "合成数据" if row["backend"] in {"synthetic", "synthetic_mi"} else row["backend"]
    return "真机明确任务" if row["task"] in {"motor_execution", "motor_imagery"} else "旧校准：任务类型未记录"


def task_name(row):
    return TASK_NAMES.get(row["task"], row["task"])


def stop_name(row):
    reason = row["stop_reason"]
    return STOP_NAMES.get(reason, reason) if reason else "未记录"


def count_text(value):
    return "未评估" if value is None else str(value)


def write_csv(rows):
    headers = ["类别", "任务开始时间", "数据源启动时间", "数据源停止时间", "受试者", "会话", "任务", "采集后端",
               "采样率Hz", "原始样本数", "原始秒数", "目标试次", "尝试次数", "完整试次", "有效试次", "有效左手", "有效右手",
               "计划实验组数", "有效试次按组分布", "停止原因", "数据中断次数", "恢复成功次数", "连接失败次数",
               "恢复耗时下界秒", "恢复耗时上界秒", "C3基线恒定试次", "C3任务恒定试次", "Cz基线恒定试次", "Cz任务恒定试次",
               "C4基线恒定试次", "C4任务恒定试次", "中央三通道配对非恒定试次", "中央三通道未评估试次", "缺基线试次", "原始文件", "报告文件"]
    with (DESTINATION / "recordings.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for row in rows:
            timing = row["timing"]
            constant = row["constant_channel_check"]
            writer.writerow([category(row), row["filename_job_start_local"], row["source_started_local"], row["source_stopped_local"],
                row["subject"], row["session"], task_name(row), row["backend"], row["sample_rate"], row["raw_samples"],
                round(row["raw_seconds"], 3), row["target_trials"], row["attempted_trials"], row["complete_trials"], row["usable_trials"],
                row["usable_class_counts"].get("left_hand", 0), row["usable_class_counts"].get("right_hand", 0),
                row["target_blocks"], json.dumps(row["usable_block_counts"], ensure_ascii=False), stop_name(row),
                row["recovery_stats"].get("gaps"), row["recovery_stats"].get("reconnects"), row["recovery_stats"].get("failures"),
                round(timing["post_failure_recovering_seconds_lower"], 3),
                round(timing["post_failure_recovering_seconds_upper"], 3) if timing["post_failure_recovering_seconds_upper"] is not None else None,
                *[constant[phase]["constant_counts"].get(name) for name in CENTRAL_CHANNELS for phase in ("baseline", "task")],
                constant["paired_central_all_nonconstant_trials"], constant["paired_central_unassessed_trials"], constant["missing_baseline_trials"],
                row["path"], " | ".join(row["reports"])])


def md_link(path, label=None):
    target = Path(os.path.relpath(ROOT / path, DESTINATION)).as_posix()
    return f"[{label or path}](<{target}>)"


def write_readme(inventory):
    formal = inventory["summary"]["explicit_tasks"]
    lines = ["# 2026-10-31 采集文件清单", "",
        f"明确记录任务类型的真机采集共 {formal['n_recordings']} 份 NPZ，保留 {formal['usable_trials']} 个有效试次："
        f"实际握拳 {inventory['summary']['motor_execution']['usable_trials']} 个，运动想象 {inventory['summary']['motor_imagery']['usable_trials']} 个。"
        f"另有 {len(legacy)} 份旧校准文件、{sum(r['usable_trials'] for r in legacy)} 个试次，任务类型未写入元数据。", "",
        "受试者编号直接读取当前 NPZ 的 subject_id。两份更正记录的原编号、确认依据和原件位置保存在 metadata.corrections，并收录于 index.json。", "",
        "所有时间均为香港时间（UTC+8）。表中的开始时间来自任务文件名；精确到毫秒的数据源启动与停止时间收录于 CSV 和 JSON。"
        "每行列出目标、尝试、完整、有效次数及有效左右手数量。有效试次通过伪迹标记与有限值检查。", "",
        "- [机器可读清单](index.json)：逐次统计、全部恢复事件、恢复耗时区间与更正历史。",
        "- [Excel 清单](recordings.csv)：UTF-8 BOM 编码，包含受试者、会话、实验组、路径和恢复耗时。", ""]

    def table(title, rows):
        lines.extend([f"## {title}", "", "| 开始时间 | 受试者 | 任务 | 原始秒数 | 目标/尝试/完整/有效 | 有效左/右 | 停止原因 | 文件 |",
                      "|---|---|---|---:|---|---|---|---|"])
        for row in rows:
            local = row["filename_job_start_local"] or row["source_started_local"]
            time = local[11:19] if local else "未记录"
            counts = "/".join(str(row[key]) if row[key] is not None else "未记录" for key in ("target_trials", "attempted_trials", "complete_trials", "usable_trials"))
            classes = row["usable_class_counts"]
            files = md_link(row["path"], Path(row["path"]).name)
            files += "".join("<br>" + md_link(report, "报告：" + Path(report).parent.name) for report in row["reports"])
            lines.append(f"| {time} | {row['subject']} | {task_name(row)} | {row['raw_seconds']:.3f} | {counts} | "
                         f"{classes.get('left_hand', 0)}/{classes.get('right_hand', 0)} | {stop_name(row)} | {files} |")
        lines.append("")

    table("真机：明确任务", explicit_tasks)
    table("旧校准：任务类型未记录", legacy)
    if legacy:
        lines.extend(["旧校准文件包含 imagery_seconds 配置，可在 JSON 中查看这一线索；本清单保留原始 task 缺失状态。", ""])
    table("合成及其他来源", other)
    lines.extend(["## 恒定通道检查", "",
        "逐试次检查每个通道的峰峰值是否精确等于 0。下表分别列出 C3、Cz、C4 基线与任务段恒定的试次数。"
        "“配对非恒定”统计三通道均有基线、全部数值有限且基线与任务段峰峰值均大于 0 的试次；缺基线或含非有限值的试次单列为未评估。"
        "恒定检查独立记录，各表原有完整与有效计数沿用采集规则。", "",
        "| 开始时间 | 受试者 | 任务/来源 | C3恒定：基线/任务 | Cz恒定：基线/任务 | C4恒定：基线/任务 | 配对非恒定 | 未评估 |",
        "|---|---|---|---|---|---|---:|---:|"])
    for row in records:
        check = row["constant_channel_check"]
        local = row["filename_job_start_local"] or row["source_started_local"]
        counts = ["/".join(count_text(check[phase]["constant_counts"].get(name)) for phase in ("baseline", "task")) for name in CENTRAL_CHANNELS]
        lines.append(f"| {local[11:19] if local else '未记录'} | {row['subject']} | {task_name(row)}/{row['backend']} | "
                     + " | ".join(counts) + f" | {count_text(check['paired_central_all_nonconstant_trials'])} | {check['paired_central_unassessed_trials']} |")
    lines.extend(["", "### 按受试者与任务汇总", "",
                  "| 受试者/任务 | 完整试次 | C3恒定：基线/任务 | Cz恒定：基线/任务 | C4恒定：基线/任务 | 配对非恒定 | 未评估 |",
                  "|---|---:|---|---|---|---:|---:|"])
    for key, summary in inventory["summary"]["by_subject_task"].items():
        check = summary["constant_channel_check"]
        counts = [f"{count_text(check['central_constant_counts']['baseline'][name])}/{count_text(check['central_constant_counts']['task'][name])}" for name in CENTRAL_CHANNELS]
        lines.append(f"| {key} | {summary['complete_trials']} | " + " | ".join(counts)
                     + f" | {count_text(check['paired_central_all_nonconstant_trials'])} | {check['paired_central_unassessed_trials']} |")
    lines.append("")
    lines.extend(["## 仅报告文件", "", "| 任务开始时间 | 受试者 | 来源 | 任务 | 有效试次 | 状态 | 报告路径 |", "|---|---|---|---|---:|---|---|"])
    for row in report_only:
        start = row["filename_job_start_local"]
        lines.append(f"| {start[11:19] if start else '未记录'} | {row['subject']} | {row['source_kind']} | "
                     f"{TASK_NAMES.get(row['task'], row['task'])} | {row['n_trials']} | {row['status']} | {md_link(row['path'])} |")
    lines.extend(["", "## 恢复时间的计算", "",
        "恢复区间从首次 disconnected 或 forced_reconnect 起，到 recovered、stopped 或 failed 止。"
        "同一区间内的重复连接和失败只计一次。缺少结束事件的区间保留已观察下界；有采集总时长的记录同时给出上界。", "",
        f"25 份真机记录的恢复状态耗时合计至少 {inventory['summary']['hardware']['recovering_seconds_lower']:.3f} 秒。"
        "S01 的 15:07:02 记录缺少停止事件和总时长，因此总上界留空。首次初始化与稳定等待另列；稳定等待可与恢复区间重叠。", "",
        "原始秒数由样本数除以采样率计算。任务总时长与原始秒数的差额包含初始化、无新样本区间、恢复和清理；"
        "Wi-Fi 链路状态时长未写入这些文件。", "",
        "## 重新生成", "", "```powershell",
        r".\.venv-mi\Scripts\python.exe -X utf8 scripts/maintenance/inventory_20261031.py", "```", ""])
    (DESTINATION / "README.md").write_text("\n".join(lines), encoding="utf-8")


inventory = dict(inventory_version=1, day="2026-10-31", timezone="Asia/Hong_Kong",
    generated_at=datetime.now(TZ).isoformat(timespec="seconds"),
    subject_corrections=[dict(path=row["path"], history=row["corrections"]) for row in records if row["corrections"]],
    timing_method={
        "start": "filename_job_start_local 来自任务生成的文件名；source_started_local 使用同一文件 started_at-started_monotonic 锚点中位数换算首次 connecting。",
        "recovery": "首次 disconnected/forced_reconnect 到 recovered/stopped/failed；中间多次 connecting/connected/disconnected 合并为一段。",
        "open_interval": "缺少结束事件时，最后事件给下界；若有 collection_summary.elapsed_seconds，则 first_event_monotonic+elapsed 给结束上界。采集计时开始于 source.start 之前。",
        "startup": "initial_control_ready_seconds 单列首次 connecting 到 connected；connected 是初始化完成，首包收取时间在该事件之后。",
        "wait": "stable_wait_seconds 来自采集等待历史，包含积累2秒信号和等待近期样本，与恢复耗时可能重叠。",
        "raw": "raw_seconds=连续样本数/采样率，表示保存的名义采样时长。elapsed_minus_raw_nominal_seconds 同时包含初始化、无新样本区间、恢复、清理和时钟差。",
        "network": "这些字段描述程序收数和恢复状态；文件未保存Wi-Fi链路断开时长。"},
    summary=dict(hardware=aggregate(hardware), explicit_tasks=aggregate(explicit_tasks), legacy_calibration=aggregate(legacy),
        motor_execution=aggregate([r for r in explicit_tasks if r["task"] == "motor_execution"]),
        motor_imagery=aggregate([r for r in explicit_tasks if r["task"] == "motor_imagery"]), other_sources=aggregate(other),
        by_subject={subject:aggregate([r for r in hardware if r["subject"]==subject]) for subject in sorted({r["subject"] for r in hardware})},
        by_subject_task={f"{subject}/{task}":aggregate([r for r in hardware if r["subject"]==subject and r["task"]==task])
                                for subject,task in sorted({(r["subject"],r["task"]) for r in hardware})},
        report_source_counts=dict(Counter(r["source_kind"] for r in reports))),
    recordings=records, reports=reports, report_only=report_only, skipped=skipped)
DESTINATION.mkdir(parents=True, exist_ok=True)
(DESTINATION / "index.json").write_text(json.dumps(inventory,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
write_csv(records)
write_readme(inventory)
print(json.dumps({"output": str(DESTINATION), "hardware_recordings": len(hardware),
                  "explicit_task_trials": inventory["summary"]["explicit_tasks"]["usable_trials"],
                  "motor_trials": inventory["summary"]["motor_execution"]["usable_trials"],
                  "imagery_trials": inventory["summary"]["motor_imagery"]["usable_trials"],
                  "legacy_calibration_trials": inventory["summary"]["legacy_calibration"]["usable_trials"],
                  "other_source_recordings": len(other), "report_only": len(report_only)}, ensure_ascii=False, indent=2))
