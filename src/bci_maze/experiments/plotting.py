"""Standalone scientific PNGs with explicit electrode coverage and shared scales."""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import numpy as np
from scipy.interpolate import griddata


# Numerical sensor coordinates (mm), MNE 1.8 standard_1020 template. Projection:
# orthographic x/y view at a fixed 100 mm head radius; nose is towards +y.
# Source: https://raw.githubusercontent.com/mne-tools/mne-python/v1.8.0/mne/channels/data/montages/standard_1020.elc
MONTAGE_MM = {
    "FC3": (-60.1819, 22.7162, 55.544), "FC4": (62.2931, 23.7228, 55.63),
    "C3": (-65.3581, -11.6317, 64.358), "Cz": (0.4009, -9.167, 100.244),
    "C4": (67.1179, -10.9003, 63.58), "CP3": (-63.5562, -47.0088, 65.624),
    "CPz": (0.3858, -47.318, 99.432), "CP4": (66.6118, -46.6372, 65.58),
    "Fz": (0.3122, 58.512, 66.462), "FC1": (-34.0619, 26.0111, 79.987),
    "FCz": (0.3761, 27.39, 88.668), "FC2": (34.7841, 26.4379, 78.808),
    "C5": (-80.2801, -13.7597, 29.16), "C6": (83.4559, -12.7763, 29.208),
    "CP1": (-35.5131, -47.2919, 91.315), "CP2": (38.3838, -47.0731, 90.695),
}
CLASS_NAMES = {"left_hand": "左手", "right_hand": "右手"}
COLORS = {"left_hand": "#167d95", "right_hand": "#bd5a32"}
BAND_NAMES = {"mu": "μ · 8–13 Hz", "beta": "β · 13–30 Hz"}


def _figure(report: dict, title: str, size=(12, 7)) -> Figure:
    figure = Figure(figsize=size, layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    source = {"synthetic": "合成数据", "hardware": "设备采集", "lsl": "LSL 数据", "replay": "回放数据"}.get(report["source_kind"], "来源待确认")
    task = {"motor_execution": "实际运动", "motor_imagery": "运动想象"}.get(report["task"], "任务待确认")
    figure.suptitle(f"{title}\n{report['subject']} · {task} · {source} · {report['n_trials']} 个有效试次", fontsize=14)
    return figure


def _save(figure: Figure, output: Path, file: str) -> None:
    # FigureCanvasAgg avoids changing the backend of any existing operator window.
    temporary = output / (file + ".tmp")
    figure.savefig(temporary, format="png", dpi=160, facecolor="white")
    temporary.replace(output / file)
    figure.clear()


def _empty(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color="#52616b", wrap=True)
    ax.set_axis_off()


def _draw_ersp(report: dict, output: Path) -> dict | None:
    data = report["analysis"]["ersp"]
    if not data["available"]:
        return None
    figure = _figure(report, "事件相关频谱变化（ERSP）", (13, 7))
    axes = figure.subplots(2, 3, squeeze=False)
    panels = {(item["class"], item["channel"]): item for item in data["panels"]}
    finite = [np.abs(np.asarray(item["db"])[np.isfinite(item["db"])]) for item in data["panels"]]
    values = np.concatenate(finite) if finite else np.empty(0)
    limit = max(3.0, float(np.percentile(values, 98))) if values.size else 3.0
    image = None
    for row, label in enumerate(CLASS_NAMES):
        for column, name in enumerate(("C3", "Cz", "C4")):
            ax = axes[row, column]
            item = panels.get((label, name))
            if item is None:
                _empty(ax, f"{CLASS_NAMES[label]} · {name}\n缺少该通道或类别")
                continue
            image = ax.pcolormesh(data["times"], data["frequencies"], np.ma.masked_invalid(item["db"]),
                                  cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
            ax.set(title=f"{CLASS_NAMES[label]} · {name}", xlabel="活动起点后时间 / s", ylabel="频率 / Hz")
            ax.set_xlim(0, report["active_seconds"])
            ax.axhline(8, color="#555555", linestyle=":", linewidth=0.7)
            ax.axhline(13, color="#555555", linestyle=":", linewidth=0.7)
            ax.axhline(30, color="#555555", linestyle=":", linewidth=0.7)
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), label="相对配对静息基线 / dB", shrink=0.82, extend="both")
    _save(figure, output, "ersp.png")
    return dict(id="ersp", title="C3 / Cz / C4 事件相关频谱变化", file="ersp.png",
                caption="每试次先计算时频功率的 10log10 值，再减去自身同频率静息 dB 的时间均值，最后按类别平均；基线对应几何平均功率。蓝色表示功率下降，红色表示功率上升；六面板共享色标。1 秒窗中心显示于活动段内，起点为 0 秒。")


def _draw_erd(report: dict, output: Path) -> dict | None:
    rows = report["analysis"]["erd_summary"]
    if not rows:
        return None
    figure = _figure(report, "运动区 μ / β 频带与静息基线对比", (11, 5))
    axes = figure.subplots(1, 2)
    names = [name for name in ("C3", "Cz", "C4") if name in report["channels"]]
    for ax, band in zip(axes, BAND_NAMES):
        if not names:
            _empty(ax, "记录未包含 C3、Cz 或 C4")
            continue
        positions = np.arange(len(names))
        for offset, label in zip((-0.19, 0.19), CLASS_NAMES):
            by_name = {item["channel"]: item for item in rows if item["class"] == label and item["band"] == band}
            changes = [by_name.get(name, {}).get("erd_percent", np.nan) for name in names]
            ax.bar(positions + offset, changes, width=0.36, color=COLORS[label], label=CLASS_NAMES[label])
        ax.axhline(0, color="#37474f", linewidth=1)
        ax.set_xticks(positions, names)
        ax.set(title=BAND_NAMES[band], ylabel="相对静息功率变化 / %")
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False)
    _save(figure, output, "erd_comparison.png")
    return dict(id="erd_comparison", title="μ / β 活动与静息对比", file="erd_comparison.png",
                caption="先按配对试次计算 100×(活动功率−静息功率)/静息功率，再求类别均值。负值为事件相关去同步化（ERD），正值为事件相关同步化（ERS）。")


def _draw_psd(report: dict, output: Path) -> dict:
    figure = _figure(report, "活动频谱与配对静息频谱" if report["baseline_available"] else "活动功率谱", (13, 4.7))
    axes = figure.subplots(1, 3)
    names = [name for name in ("C3", "Cz", "C4") if name in report["channels"]]
    if not names:
        names = report["channels"][:3]
    for index, ax in enumerate(axes):
        if index >= len(names):
            _empty(ax, "该位置无采集通道")
            continue
        name = names[index]
        for item in report["analysis"]["psd"]:
            if item["channel"] != name:
                continue
            label = item["class"]
            frequencies = report["analysis"]["frequencies"]
            mean, sem = np.asarray(item["mean_db"]), np.asarray(item["sem_db"])
            ax.plot(frequencies, mean, color=COLORS[label], label=f"{CLASS_NAMES[label]}活动")
            ax.fill_between(frequencies, mean - sem, mean + sem, color=COLORS[label], alpha=0.12)
            if "baseline_mean_db" in item:
                ax.plot(frequencies, item["baseline_mean_db"], color=COLORS[label], linestyle="--", linewidth=1.1,
                        label=f"{CLASS_NAMES[label]}配对静息")
        ax.set(title=name, xlabel="频率 / Hz", ylabel="PSD / dB (µV²/Hz)", xlim=(4, 40))
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=8)
    _save(figure, output, "power_spectra.png")
    return dict(id="power_spectra", title="活动功率谱与静息对比" if report["baseline_available"] else "活动功率谱", file="power_spectra.png",
                caption="实线为逐试次 Welch 对数功率谱均值，阴影为均值标准误；记录含配对静息片段时以同色虚线展示静息谱。")


def _draw_topomaps(report: dict, output: Path, kind: str) -> dict | None:
    panels = [item for item in report["analysis"]["topomaps"] if item["kind"] == kind]
    known = [(index, name) for index, name in enumerate(report["channels"]) if name in MONTAGE_MM]
    if not panels or not known:
        return None
    points = np.asarray([MONTAGE_MM[name][:2] for _, name in known]) / 100.0
    finite = np.concatenate([np.asarray(item["values"])[np.isfinite(item["values"])] for item in panels])
    if not finite.size:
        return None
    erd = kind == "erd_percent"
    low, high = float(finite.min()), float(finite.max())
    if erd:
        high = max(5.0, abs(low), abs(high))
        low = -high
    elif high - low < 1:
        low, high = low - 0.5, high + 0.5
    figure = _figure(report, "频带 ERD / ERS 的电极分布" if erd else "活动频带功率的电极分布", (10, 8))
    axes = figure.subplots(2, 2)
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 160), np.linspace(-1, 1, 160))
    image = None
    for row, label in enumerate(CLASS_NAMES):
        for column, band in enumerate(BAND_NAMES):
            ax = axes[row, column]
            item = next((value for value in panels if value["class"] == label and value["band"] == band), None)
            if item is None:
                _empty(ax, "该类别缺少有效试次")
                continue
            values = np.asarray([item["values"][index] for index, _ in known])
            valid = np.isfinite(values)
            if valid.sum() >= 3 and np.linalg.matrix_rank(points[valid] - points[valid].mean(axis=0)) == 2:
                field = griddata(points[valid], values[valid], (grid_x, grid_y), method="linear", fill_value=np.nan)
                field[grid_x**2 + grid_y**2 > 1] = np.nan
                ax.pcolormesh(grid_x, grid_y, np.ma.masked_invalid(field), cmap="RdBu_r" if erd else "viridis",
                              shading="auto", vmin=low, vmax=high, rasterized=True)
            image = ax.scatter(points[valid, 0], points[valid, 1], c=values[valid], cmap="RdBu_r" if erd else "viridis",
                               vmin=low, vmax=high, s=42, edgecolors="#18252d", linewidths=0.6, zorder=3)
            for (x, y), (_, name) in zip(points, known):
                ax.annotate(name, (x, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5, zorder=4)
            ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor="#687b85", linewidth=1))
            ax.plot([-0.12, 0, 0.12], [0.99, 1.12, 0.99], color="#687b85", linewidth=1)
            ax.text(-1.08, 0, "左", ha="right", va="center", fontsize=9)
            ax.text(1.08, 0, "右", ha="left", va="center", fontsize=9)
            ax.set(title=f"{CLASS_NAMES[label]} · {BAND_NAMES[band]}", xlim=(-1.2, 1.2), ylim=(-1.07, 1.18), aspect="equal")
            ax.set_axis_off()
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), label="相对静息变化 / %" if erd else "频带功率 / dB (µV²)", shrink=0.78)
    file = "topomap_erd.png" if erd else "topomap_power.png"
    _save(figure, output, file)
    return dict(id="topomap_erd" if erd else "topomap_power", title="ERD 电极地形分布" if erd else "活动功率电极地形分布",
                file=file, caption="鼻尖朝上、受试者左侧在图左。点为采集电极的标准模板位置；着色插值覆盖测量点凸包，头部其余区域留空。四面板共享色标。")


def _draw_r2(report: dict, output: Path) -> dict | None:
    data = report["analysis"]["signed_r2"]
    if not data["available"]:
        return None
    figure = _figure(report, "左右手频谱差异：signed r²", (11, max(5, len(report["channels"]) * 0.3)))
    ax = figure.subplots()
    image = ax.pcolormesh(report["analysis"]["frequencies"], np.arange(len(report["channels"])), data["values"],
                         cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
    ax.set_yticks(np.arange(len(report["channels"])), report["channels"])
    ax.invert_yaxis()
    ax.set(xlabel="频率 / Hz", ylabel="EEG 通道", xlim=(4, 40))
    figure.colorbar(image, ax=ax, label="signed r²：正值右手功率较高，负值左手较高")
    _save(figure, output, "signed_r2.png")
    return dict(id="signed_r2", title="频率 × 通道的左右手效应量", file="signed_r2.png",
                caption="逐试次对数功率与二元标签的点二列相关平方，符号取右手功率均值减左手功率均值。统一色标为 −1 至 1，结果用于描述效应方向与大小。")


def _draw_benchmark(report: dict, output: Path) -> dict | None:
    benchmark = report.get("benchmark")
    if not benchmark or benchmark.get("status") != "completed":
        return None
    figure = _figure(report, "CSP + LDA 分组评估与标签置乱", (11, 4.8))
    axes = figure.subplots(1, 2)
    matrix = np.asarray(benchmark["confusion_matrix"])
    axes[0].imshow(matrix, cmap="Blues", vmin=0)
    for (row, column), value in np.ndenumerate(matrix):
        axes[0].text(column, row, str(value), ha="center", va="center", color="white" if value > matrix.max() * 0.55 else "#203341", fontsize=16)
    names = [CLASS_NAMES[name] for name in benchmark["class_order"]]
    axes[0].set(xticks=[0, 1], xticklabels=names, yticks=[0, 1], yticklabels=names,
                xlabel="预测类别", ylabel="真实类别", title=f"分组测试汇总 · 平衡准确率 {benchmark['balanced_accuracy']:.1%}")
    permutation = benchmark["permutation"]
    scores = permutation["null_scores"]
    if scores:
        axes[1].hist(np.asarray(scores) * 100, bins=np.linspace(0, 100, 21), color="#a4b9c4", edgecolor="white")
        axes[1].set_title(f"组内标签置乱 {len(scores)} 次 · p = {permutation['p_value']:.4f}")
    else:
        axes[1].text(0.5, 0.7, "置乱次数为 0，置乱检验待运行", transform=axes[1].transAxes, ha="center")
    axes[1].axvline(50, color="#697c86", linestyle="--", label="二分类基准 50%")
    axes[1].axvline(benchmark["balanced_accuracy"] * 100, color="#b44e31", linewidth=2, label="原始标签结果")
    axes[1].set(xlabel="分组测试平衡准确率 / %", ylabel="置乱次数", xlim=(-2, 102))
    axes[1].legend(frameon=False, fontsize=9)
    _save(figure, output, "benchmark.png")
    return dict(id="benchmark", title="分组 CSP + LDA 与置乱对照", file="benchmark.png",
                caption="混淆矩阵汇总各留出组的预测。每次在组内置乱标签并重新拟合所有折；p=(1+置乱分数不低于原始分数的次数)/(N+1)。")


def plot_report(report: dict, output: Path) -> list[dict]:
    with matplotlib.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                                "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        candidates = [_draw_ersp(report, output), _draw_erd(report, output), _draw_psd(report, output),
                      _draw_topomaps(report, output, "active_log_power"), _draw_topomaps(report, output, "erd_percent"),
                      _draw_r2(report, output), _draw_benchmark(report, output)]
    return [item for item in candidates if item is not None]
