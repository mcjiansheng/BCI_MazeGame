"""Apply the operator's S02 -> S03 correction for two 2026-10-31 recordings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RUNS = ("motor-20261031-160005-8fe9d428", "imagery-20261031-160618-48921893")
ARCHIVE = ROOT / "data/archive/20261031_subject_correction"
REASON = "操作者于 2026-11-04 确认：16:02 所在的真实握拳记录及紧接的短运动想象记录均属于 S03。"


def inside(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(ROOT)
    return resolved


def read_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def verify_arrays(original: dict, corrected: dict) -> None:
    if original.keys() != corrected.keys():
        raise ValueError("更正前后的数组字段不同")
    for key, value in original.items():
        if key == "metadata_json":
            continue
        if value.dtype != corrected[key].dtype or not np.array_equal(
            value, corrected[key], equal_nan=value.dtype.kind in "fc"
        ):
            raise ValueError(f"更正修改了信号或标签数组：{key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    audit_path = ARCHIVE / "corrections.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        for item in audit["corrections"]:
            old = inside(ROOT / item["archived_recording"])
            new = inside(ROOT / item["recording"])
            verify_arrays(read_arrays(old), read_arrays(new))
            metadata = json.loads(str(read_arrays(new)["metadata_json"].item()))
            if metadata["subject_id"] != "S03":
                raise ValueError(f"已整理记录的受试者发生变化：{new}")
        print("两份更正已存在；原始数组复核通过。")
        return
    plans = []
    for run in RUNS:
        old = inside(ROOT / "data/recordings/S02" / f"experiment-{run}.npz")
        new = inside(ROOT / "data/recordings/S03" / old.name)
        report = inside(ROOT / "outputs/experiments/S02" / run)
        new_report = inside(ROOT / "outputs/experiments/S03" / run)
        archived = inside(ARCHIVE / "recordings/S02" / old.name)
        archived_report = inside(ARCHIVE / "reports/S02" / run)
        if not old.is_file() or not (report / "report.json").is_file():
            raise FileNotFoundError(f"缺少待整理记录或报告：{run}")
        for path in (new, new_report, archived, archived_report):
            if path.exists():
                raise FileExistsError(f"目标已有内容：{path}")
        arrays = read_arrays(old)
        metadata = json.loads(str(arrays["metadata_json"].item()))
        if metadata["subject_id"] != "S02" or metadata["configuration"]["backend"] != "brainflow":
            raise ValueError(f"待整理记录与操作者确认不符：{old}")
        plans.append((run, old, new, report, new_report, archived, archived_report, arrays, metadata))
    if not args.apply:
        for run, old, new, *_ in plans:
            print(f"{old.relative_to(ROOT)} -> {new.relative_to(ROOT)}")
        return

    from bci_maze.experiments.analysis import analyze_recordings

    (ROOT / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="curate-20261031-", dir=ROOT / "tmp") as temporary:
        staged = []
        for run, old, new, report, new_report, archived, archived_report, arrays, metadata in plans:
            stage = inside(Path(temporary) / run)
            stage.mkdir()
            staged_recording = stage / old.name
            correction = dict(field="subject_id", previous="S02", corrected="S03",
                              confirmed_on="2026-11-04", reason=REASON,
                              original_path=old.relative_to(ROOT).as_posix(),
                              archived_path=archived.relative_to(ROOT).as_posix())
            metadata["subject_id"] = "S03"
            metadata.setdefault("corrections", []).append(correction)
            corrected = dict(arrays, metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
            np.savez_compressed(staged_recording, **corrected)
            verify_arrays(arrays, read_arrays(staged_recording))
            old_report = json.loads((report / "report.json").read_text(encoding="utf-8"))
            staged_report = stage / "report"
            result = analyze_recordings([staged_recording], staged_report)
            for item in result["inputs"] + result["trial_origins"]:
                item["file"] = str(new)
            for field in ("collection_summary", "wear_state", "capture_interrupted", "capture_error"):
                if field in old_report:
                    result[field] = old_report[field]
            result["recording_path"] = new.relative_to(ROOT).as_posix()
            result["recording_paths"] = [result["recording_path"]]
            result["corrections"] = [correction]
            result["notes"] = [note.replace(old.relative_to(ROOT).as_posix(), new.relative_to(ROOT).as_posix())
                               for note in old_report.get("notes", [])]
            if result["subject"] != "S03" or result["n_trials"] != old_report["n_trials"] or result["analysis"] != old_report["analysis"]:
                raise ValueError(f"重新生成报告后试次或分析数值发生变化：{run}")
            (staged_report / "report.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            staged.append((staged_recording, staged_report, old, new, report, new_report, archived, archived_report))

        audit = dict(confirmed_on="2026-11-04", reason=REASON, corrections=[])
        for staged_recording, staged_report, old, new, report, new_report, archived, archived_report in staged:
            # Every move target was resolved inside the workspace before staging.
            for path in (new, new_report, archived, archived_report):
                path.parent.mkdir(parents=True, exist_ok=True)
            old.replace(archived)
            report.replace(archived_report)
            staged_recording.replace(new)
            staged_report.replace(new_report)
            verify_arrays(read_arrays(archived), read_arrays(new))
            audit["corrections"].append(dict(
                recording=new.relative_to(ROOT).as_posix(), report=(new_report / "report.json").relative_to(ROOT).as_posix(),
                archived_recording=archived.relative_to(ROOT).as_posix(),
                archived_report=(archived_report / "report.json").relative_to(ROOT).as_posix(),
                previous_subject="S02", subject="S03", signal_arrays_unchanged=True))
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
