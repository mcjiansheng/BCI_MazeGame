"""Shared Matplotlib setup and readable multi-channel EEG inspection."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .quality import ChannelQuality


def configure_plot_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


class ChannelDisplay:
    """Two banks of up to eight traces, with explicit per-electrode status."""

    def __init__(self, labels: tuple[str, ...]):
        configure_plot_fonts()
        columns = 2 if len(labels) > 8 else 1
        self.figure, axes = plt.subplots(1, columns, figsize=(13, 8), squeeze=False)
        self.axes = axes[0]
        self.labels = labels
        self.lines = []
        self.figure.suptitle(f"LK-Mini-EEG16 | {len(labels)} 通道实时检查")
        self.figure.subplots_adjust(left=0.15, right=0.98, bottom=0.1, top=0.90, wspace=0.45)
        for column, ax in enumerate(self.axes):
            count = min(8, len(labels) - column * 8)
            ax.set_yticks(np.arange(count) * 100.0, labels[column * 8:column * 8 + count])
            ax.set_ylim(-100.0, count * 100.0)
            ax.invert_yaxis()
            ax.set_xlabel("最近时间 / s")
            ax.set_title("去中位数波形 / µV（逐路偏移）", fontsize=11)
            ax.grid(axis="x", alpha=0.2)
            self.lines.extend(ax.plot([], [], linewidth=0.7)[0] for _ in range(count))

    def update(self, data: np.ndarray, sample_rate: int, quality: list[ChannelQuality] | None) -> None:
        x = (np.arange(data.shape[1]) - data.shape[1]) / sample_rate
        # Center only the visual traces. Raw saved data and quality remain untouched.
        finite_data = np.where(np.isfinite(data), data, 0.0)
        centered = finite_data - np.median(finite_data, axis=1, keepdims=True)
        for column, ax in enumerate(self.axes):
            start = column * 8
            stop = min(start + 8, len(self.labels))
            span = max(50.0, float(np.percentile(np.abs(centered[start:stop]), 95)) * 3)
            labels = []
            for index in range(start, stop):
                status = quality[index].status if quality else "waiting"
                self.lines[index].set_data(x, centered[index] + (index - start) * span)
                self.lines[index].set_color({
                    "good": "#167a60", "warning": "#ac6b08", "bad": "#c13745",
                }.get(status, "#777777"))
                labels.append(f"{self.labels[index]} [{status}]")
            ax.set_yticks(np.arange(stop - start) * span, labels)
            ax.set_ylim(span * (stop - start), -span)
            ax.set_xlim(x[0], 0.0)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
