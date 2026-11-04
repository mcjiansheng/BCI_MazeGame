"""Model-independent EEG spectra, ERD/ERSP and signed r-squared analyses."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import spectrogram, welch

from .benchmark import CLASSES, grouped_benchmark


BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}
EPSILON = 1e-12
METHOD_SOURCES = {
    "erd": "https://pubmed.ncbi.nlm.nih.gov/10576479/",
    "welch": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html",
    "spectrogram": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html",
    "baseline_mean": "https://mne.tools/1.8/generated/mne.baseline.rescale.html",
    "signed_r2": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pointbiserialr.html",
    "groups": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html",
    "permutation": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html",
    "montage": "https://raw.githubusercontent.com/mne-tools/mne-python/v1.8.0/mne/channels/data/montages/standard_1020.elc",
}


@dataclass
class Recordings:
    epochs: np.ndarray
    baseline_epochs: np.ndarray | None
    labels: np.ndarray
    sessions: np.ndarray
    blocks: np.ndarray
    sample_rate: int
    channels: tuple[str, ...]
    subject: str
    task: str
    source_kind: str
    excluded_counts: dict
    inputs: list[dict]
    trial_origins: list[dict]


def _source_kind(metadata: dict) -> str:
    provenance = metadata.get("provenance", {})
    configuration = metadata.get("configuration", {})
    values = [str(value) for value in (provenance.get("source_kind"), provenance.get("kind"),
                                      metadata.get("source_kind")) if value]
    backend = configuration.get("backend", provenance.get("backend"))
    inferred = {"brainflow": "hardware", "lsl": "lsl", "replay": "replay",
                "synthetic": "synthetic", "synthetic_mi": "synthetic"}.get(backend)
    if inferred:
        values.append(inferred)
    if len(set(values)) > 1:
        raise ValueError("记录中的数据来源字段相互冲突。")
    return values[0] if values else "unspecified"


def load_recordings(paths: list[Path]) -> Recordings:
    """Validate identity and EEG arrays while preserving original group membership."""
    if not paths:
        raise ValueError("请选择至少一个采集记录。")
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("同一采集记录被重复选择。")
    arrays, baselines, labels_all, sessions_all, blocks_all = [], [], [], [], []
    inputs, origins = [], []
    expected = None
    exclusions = dict(total_input_trials=0, artifact_marked=0, nonfinite_after_artifact=0,
                      total_excluded=0, missing_baseline_trials=0)
    for path in resolved:
        with np.load(path, allow_pickle=False) as saved:
            required = {"epochs", "labels", "blocks", "artifact", "channel_names", "sample_rate", "metadata_json"}
            if not required.issubset(saved.files):
                raise ValueError(f"{path.name} 缺少采集字段：{', '.join(sorted(required - set(saved.files)))}")
            metadata = json.loads(str(saved["metadata_json"].item()))
            if not isinstance(metadata, dict) or not isinstance(metadata.get("configuration", {}), dict) or not isinstance(metadata.get("provenance", {}), dict):
                raise ValueError(f"{path.name} 元数据格式错误。")
            subject, session = str(metadata.get("subject_id", "")).strip(), str(metadata.get("session_id", "")).strip()
            if not subject or not session:
                raise ValueError(f"{path.name} 需要明确的受试者编号和会话编号。")
            task = str(metadata.get("configuration", {}).get("task", metadata.get("task", "unspecified")))
            if task not in {"motor_execution", "motor_imagery", "unspecified"}:
                raise ValueError(f"{path.name} 任务类型须为 motor_execution 或 motor_imagery。")
            source_kind = _source_kind(metadata)
            rate_value = saved["sample_rate"].item()
            if not np.isfinite(rate_value) or float(rate_value) != int(rate_value) or rate_value <= 80:
                raise ValueError("分析 4–40 Hz 频谱需要高于 80 Hz 的整数采样率。")
            rate = int(rate_value)
            channel_array = np.asarray(saved["channel_names"])
            if channel_array.ndim != 1:
                raise ValueError("channel_names 必须是一维数组。")
            channels = tuple(str(value) for value in channel_array.tolist())
            if not channels or len(set(channels)) != len(channels):
                raise ValueError("通道名称必须非空且互不重复。")
            if metadata.get("units", "uV") != "uV":
                raise ValueError("输入 EEG 单位须为 uV。")
            epochs = np.asarray(saved["epochs"], dtype=np.float64)
            labels = np.asarray(saved["labels"]).astype(str)
            blocks_raw = np.asarray(saved["blocks"])
            artifact_raw = np.asarray(saved["artifact"])
            if epochs.ndim != 3 or epochs.shape[1] != len(channels) or epochs.shape[2] < rate:
                raise ValueError(f"{path.name} EEG 形状应为试次 × 通道 × 采样点，每试次至少 1 秒。")
            n = len(epochs)
            if any(values.shape != (n,) for values in (labels, blocks_raw, artifact_raw)):
                raise ValueError(f"{path.name} 标签、分组、伪迹标记与试次数量不一致。")
            if not set(labels.tolist()).issubset(CLASSES):
                raise ValueError(f"{path.name} 分析标签须为 left_hand、right_hand。")
            if not np.isfinite(blocks_raw).all() or np.any(blocks_raw != np.floor(blocks_raw)) or np.any(blocks_raw < 0):
                raise ValueError("实验组编号须为非负整数。")
            if not np.isin(artifact_raw, [False, True]).all():
                raise ValueError("伪迹标记须为布尔值或 0/1。")
            artifact = artifact_raw.astype(bool)
            blocks = blocks_raw.astype(np.int64)
            baseline = np.asarray(saved["baseline_epochs"], dtype=np.float64) if "baseline_epochs" in saved else None
            if baseline is not None and (baseline.ndim != 3 or baseline.shape[:2] != epochs.shape[:2] or baseline.shape[-1] < rate):
                raise ValueError(f"{path.name} 基线须与活动试次逐一对应，且每段至少 1 秒。")
            identity = (subject, channels, rate, task, source_kind, epochs.shape[-1])
            if expected is None:
                expected = identity
            elif identity != expected:
                raise ValueError("合并记录的受试者、通道顺序、采样率、任务、来源和活动时长须一致。")
            finite = np.isfinite(epochs).all(axis=(1, 2))
            if baseline is not None:
                finite &= np.isfinite(baseline).all(axis=(1, 2))
                if baselines and baselines[0] is not None and baseline.shape[-1] != baselines[0].shape[-1]:
                    raise ValueError("合并记录的基线时长须一致。")
            keep = ~artifact & finite
            exclusions["total_input_trials"] += n
            exclusions["artifact_marked"] += int(artifact.sum())
            exclusions["nonfinite_after_artifact"] += int(np.count_nonzero(~artifact & ~finite))
            exclusions["total_excluded"] += int(np.count_nonzero(~keep))
            if baseline is None:
                exclusions["missing_baseline_trials"] += int(keep.sum())
            arrays.append(epochs[keep])
            baselines.append(baseline[keep] if baseline is not None else None)
            labels_all.append(labels[keep])
            sessions_all.append(np.repeat(session, int(keep.sum())))
            blocks_all.append(blocks[keep])
            inputs.append(dict(file=str(path), session=session, n_trials=n, retained_trials=int(keep.sum()),
                               artifact_marked=int(artifact.sum()),
                               nonfinite_after_artifact=int(np.count_nonzero(~artifact & ~finite)),
                               total_excluded=int(np.count_nonzero(~keep)),
                               baseline_available=baseline is not None,
                               baseline_seconds=baseline.shape[-1] / rate if baseline is not None else None))
            origins.extend(dict(file=str(path), trial_index=int(index), session=session, block=int(blocks[index]))
                           for index in np.flatnonzero(keep))
    epochs = np.concatenate(arrays)
    retained_baselines = [baseline for baseline, epochs_part in zip(baselines, arrays) if len(epochs_part)]
    if not retained_baselines:
        # Preserve the input baseline shape and availability for an empty report.
        retained_baselines = baselines
    baseline = np.concatenate(retained_baselines) if all(value is not None for value in retained_baselines) else None
    return Recordings(epochs, baseline, np.concatenate(labels_all), np.concatenate(sessions_all),
                      np.concatenate(blocks_all), expected[2], expected[1], expected[0], expected[3],
                      expected[4], exclusions, inputs, origins)


def power_spectra(epochs: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(int(sample_rate), epochs.shape[-1])
    return welch(epochs, fs=sample_rate, window="hann", nperseg=nperseg, noverlap=nperseg // 2,
                 detrend="constant", scaling="density", axis=-1)


def signed_r_squared(log_power: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Signed squared point-biserial correlation; positive means right > left."""
    values, labels = np.asarray(log_power, dtype=float), np.asarray(labels)
    if values.shape[0] != len(labels) or set(labels.tolist()) != set(CLASSES):
        raise ValueError("signed r² 需要两类试次，并与功率数组一一对应。")
    if not np.isfinite(values).all():
        raise ValueError("signed r² 输入必须为有限值。")
    target = (labels == "right_hand").astype(float)
    centered = values - values.mean(axis=0, keepdims=True)
    target -= target.mean()
    numerator = np.tensordot(target, centered, axes=(0, 0))
    denominator = np.sqrt(np.sum(target**2) * np.sum(centered**2, axis=0))
    correlation = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > EPSILON)
    correlation = np.clip(correlation, -1.0, 1.0)
    return np.sign(correlation) * correlation**2


def _band_power(frequencies: np.ndarray, power: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    mask = (frequencies >= band[0]) & (frequencies <= band[1])
    return trapezoid(power[..., mask], frequencies[mask], axis=-1)


def compute_analysis(recording: Recordings) -> dict:
    rate, labels, channels = recording.sample_rate, recording.labels, recording.channels
    frequencies, activity = power_spectra(recording.epochs, rate)
    mask = (frequencies >= 4) & (frequencies <= 40)
    result = {"frequencies": frequencies[mask], "psd": [], "erd_summary": [], "topomaps": [],
              "ersp": {"available": False, "reason": "缺少逐试次静息基线，待采集含 baseline_epochs 的记录。"}}
    log_activity = 10 * np.log10(np.maximum(activity, EPSILON))
    baseline_power = power_spectra(recording.baseline_epochs, rate)[1] if recording.baseline_epochs is not None else None
    for label in CLASSES:
        select = labels == label
        if not select.any():
            continue
        for channel_index, name in enumerate(channels):
            spectra = log_activity[select, channel_index][:, mask]
            item = dict(**{"class": label}, channel=name, mean_db=spectra.mean(axis=0),
                                      sem_db=spectra.std(axis=0, ddof=1) / np.sqrt(len(spectra)) if len(spectra) > 1 else np.zeros(spectra.shape[1]),
                                       n_trials=int(select.sum()))
            if baseline_power is not None:
                baseline_db = 10 * np.log10(np.maximum(baseline_power[select, channel_index][:, mask], EPSILON))
                item["baseline_mean_db"] = baseline_db.mean(axis=0)
            result["psd"].append(item)
        for band_name, band in BANDS.items():
            active_band = _band_power(frequencies, activity[select], band)
            result["topomaps"].append(dict(**{"class": label}, band=band_name, kind="active_log_power",
                                          values=np.mean(10 * np.log10(np.maximum(active_band, EPSILON)), axis=0)))
            if baseline_power is None:
                continue
            base_band = _band_power(frequencies, baseline_power[select], band)
            ratios = np.divide(100 * (active_band - base_band), base_band,
                               out=np.full_like(base_band, np.nan), where=base_band > EPSILON)
            erd_values = []
            for channel_index, name in enumerate(channels):
                valid = np.isfinite(ratios[:, channel_index])
                change = float(ratios[valid, channel_index].mean()) if valid.any() else np.nan
                erd_values.append(change)
                result["erd_summary"].append(dict(**{"class": label}, channel=name, band=band_name,
                    baseline_power=float(base_band[:, channel_index].mean()), active_power=float(active_band[:, channel_index].mean()),
                    erd_percent=change, n_trials=int(valid.sum())))
            result["topomaps"].append(dict(**{"class": label}, band=band_name, kind="erd_percent", values=np.asarray(erd_values)))
    if set(labels.tolist()) == set(CLASSES):
        result["signed_r2"] = dict(available=True, values=signed_r_squared(log_activity[..., mask], labels),
                                  positive_class="right_hand", negative_class="left_hand", range=[-1, 1])
    else:
        result["signed_r2"] = dict(available=False, reason="signed r² 需要左右手两类有效试次。")
    if recording.baseline_epochs is not None:
        kwargs = dict(fs=rate, window="hann", nperseg=rate, noverlap=rate - max(1, rate // 4),
                      detrend="constant", scaling="density", mode="psd", axis=-1)
        frequencies_tf, times, task_tf = spectrogram(recording.epochs, **kwargs)
        _, _, baseline_tf = spectrogram(recording.baseline_epochs, **kwargs)
        # Center log power in log space so identically distributed rest and task
        # spectra have equal expected values, including stationary noise bins.
        task_db = 10 * np.log10(np.maximum(task_tf, EPSILON))
        baseline_db = 10 * np.log10(np.maximum(baseline_tf, EPSILON))
        changes = task_db - baseline_db.mean(axis=-1, keepdims=True)
        changes = np.where(baseline_tf.mean(axis=-1, keepdims=True) > EPSILON, changes, np.nan)
        frequency_mask = (frequencies_tf >= 4) & (frequencies_tf <= 40)
        panels = []
        for label in CLASSES:
            select = labels == label
            if not select.any():
                continue
            for name in ("C3", "Cz", "C4"):
                if name in channels:
                    index = channels.index(name)
                    values = changes[select, index][:, frequency_mask]
                    count = np.isfinite(values).sum(axis=0)
                    mean = np.divide(np.nansum(values, axis=0), count, out=np.full(values.shape[1:], np.nan), where=count > 0)
                    panels.append(dict(**{"class": label}, channel=name, db=mean, n_trials=int(select.sum())))
        result["ersp"] = dict(available=True, frequencies=frequencies_tf[frequency_mask], times=times,
                              panels=panels, normalization="mean_of_single_trial_log_power_minus_mean_baseline_log_power", units="dB")
    return result


def _json_values(value):
    if isinstance(value, np.ndarray):
        return _json_values(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def analyze_recordings(paths: list[Path], output_dir: Path, *, run_benchmark: bool = False,
                       permutations: int = 30, seed: int = 42, split: str = "auto") -> dict:
    """Write a reproducible JSON report and independently readable figure files."""
    from .plotting import plot_report

    recording = load_recordings(paths)
    no_usable_trials = not len(recording.epochs)
    empty_reason = (f"输入 {recording.excluded_counts['total_input_trials']} 个试次，"
                    f"剔除标记伪迹 {recording.excluded_counts['artifact_marked']} 个、"
                    f"其余含非有限值 {recording.excluded_counts['nonfinite_after_artifact']} 个；"
                    "保留的有效试次数为 0，需补采有效试次后完成分析。")
    analysis = (dict(frequencies=[], psd=[], erd_summary=[], topomaps=[],
                     ersp=dict(available=False, reason=empty_reason),
                     signed_r2=dict(available=False, reason=empty_reason))
                if no_usable_trials else compute_analysis(recording))
    notes = []
    if recording.task == "unspecified":
        notes.append("原始记录缺少 task 字段；任务类型保留为 unspecified。")
    if recording.baseline_epochs is None and not no_usable_trials:
        notes.append(f"所选有效试次中有 {recording.excluded_counts['missing_baseline_trials']} 个缺少明确的静息片段；已完成活动功率分析，ERSP 与 ERD 等待含配对基线的记录。")
    result = dict(report_version=1, subject=recording.subject, task=recording.task,
                  source_kind=recording.source_kind, sample_rate=recording.sample_rate,
                  channels=list(recording.channels), classes=list(CLASSES), n_trials=len(recording.epochs),
                  class_counts={label: int(np.count_nonzero(recording.labels == label)) for label in CLASSES},
                  baseline_available=recording.baseline_epochs is not None,
                  baseline_seconds=recording.baseline_epochs.shape[-1] / recording.sample_rate if recording.baseline_epochs is not None else None,
                  active_seconds=recording.epochs.shape[-1] / recording.sample_rate,
                  excluded_counts=recording.excluded_counts, inputs=recording.inputs,
                  trial_origins=recording.trial_origins, analysis=analysis, benchmark=None, figures=[], notes=notes,
                  methods={
                      "spectra": dict(description="逐试次 Welch 功率谱密度；Hann 窗 1 秒、重叠 50%、去常量趋势。输入单位 µV，频谱展示 4–40 Hz。",
                                      psd_units="uV^2/Hz", log_floor=EPSILON, source_urls=[METHOD_SOURCES["welch"]]),
                      "erd": dict(description="先逐试次计算 100 × (活动频带功率 − 配对静息功率) / 配对静息功率，再对同类试次取平均。负值为 ERD，正值为 ERS。",
                                  bands_hz=BANDS, power_units="uV^2", source_urls=[METHOD_SOURCES["erd"]]),
                      "ersp": dict(description="活动起点为 0 秒。活动与明确保存的静息片段分别计算 1 秒 Hann 时频窗，步长约 0.25 秒。每个时频点先取 10log10 功率；每试次减去自身同频率静息 dB 的时间均值，再进行类别平均。基线参考为几何平均功率，同分布平稳信号的期望变化为 0 dB。",
                                   formula="mean_trials(10*log10(P_task(f,t)) - mean_baseline_time(10*log10(P_baseline(f,tau))))",
                                   baseline_statistic="mean_log_power_equivalent_to_geometric_mean_power",
                                   units="dB", baseline_source="baseline_epochs", log_floor=EPSILON,
                                   source_urls=[METHOD_SOURCES["spectrogram"], METHOD_SOURCES["baseline_mean"], METHOD_SOURCES["erd"]]),
                      "signed_r2": dict(description="以逐试次 Welch 对数功率和二元类别编码计算点二列相关；输出 sign(右手均值 − 左手均值) × r²，范围 −1 至 1。色彩表示效应方向与大小，频率和通道比较采用描述性统计。",
                                        source_urls=[METHOD_SOURCES["signed_r2"]]),
                      "topomap": dict(description="采用 standard_1020 模板中本项目电极的二维俯视投影；线性插值限定在已测电极凸包内，点标出电极位置。活动图为逐试次频带对数功率均值。",
                                      source_urls=[METHOD_SOURCES["montage"]]),
                      "benchmark": dict(description="CSP + 收缩 LDA 使用 7–30 Hz EEG。按完整会话或实验组留一组评估，每折重新拟合空间滤波与分类器。置乱标签限定在组内，每次按同一分组重新训练全部折；统计量为汇总测试预测的平衡准确率，二分类基准为 50%。",
                                        source_urls=[METHOD_SOURCES["groups"], METHOD_SOURCES["permutation"]]),
                  })
    if no_usable_trials:
        result.update(status="no_usable_trials", message=empty_reason,
                      benchmark=dict(status="unavailable", reason=empty_reason, requested_split=split,
                                     n_sessions=0, n_session_blocks=0))
        notes.append(empty_reason)
    elif run_benchmark:
        try:
            result["benchmark"] = grouped_benchmark(recording.epochs, recording.labels, recording.sessions,
                recording.blocks, recording.sample_rate, permutations=permutations, seed=seed, split=split)
        except ValueError as error:
            result["benchmark"] = dict(status="unavailable", reason=str(error), requested_split=split,
                n_sessions=int(np.unique(recording.sessions).size),
                n_session_blocks=len(set(zip(recording.sessions.tolist(), recording.blocks.tolist()))))
            notes.append("分组基准待完成：" + str(error))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if not no_usable_trials:
        result["figures"] = plot_report(result, directory)
    result = _json_values(result)
    temporary = directory / "report.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(directory / "report.json")
    return result
