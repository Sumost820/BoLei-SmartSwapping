import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path

# ==================== 1. 数据 ====================
capacity = np.array([70, 80, 90, 100])
total_tasks = np.array([140, 156, 160, 164])
total_swaps = np.array([26, 25, 22, 20])

# ==================== 2. 字体设置 ====================
# Windows：优先使用宋体
font_candidates = [
    Path(r"C:\Windows\Fonts\simsun.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\msyh.ttc"),
]

font_path = next((p for p in font_candidates if p.is_file()), None)
if font_path is None:
    raise FileNotFoundError("未找到中文字体，请指定本机中文字体文件路径。")

label_font = FontProperties(fname=str(font_path), size=14)
legend_font = FontProperties(fname=str(font_path), size=12)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.unicode_minus": False,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# ==================== 3. 创建双纵轴 ====================
fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
ax2 = ax1.twinx()

# 左轴：搬运任务总数
line1, = ax1.plot(
    capacity,
    total_tasks,
    color="black",
    linestyle="-",
    linewidth=1.8,
    marker="o",
    markersize=6,
    markerfacecolor="white",
    markeredgewidth=1.2,
    label="搬运任务总数"
)

# 右轴：换电总次数
line2, = ax2.plot(
    capacity,
    total_swaps,
    color="black",
    linestyle=(0, (2, 1)),
    linewidth=1.8,
    marker="s",
    markersize=6,
    markerfacecolor="white",
    markeredgewidth=1.2,
    label="换电总次数"
)

# ==================== 4. 坐标轴设置 ====================
ax1.set_xlabel("电池容量", fontproperties=label_font, labelpad=10)
ax1.set_ylabel(
    "搬运任务总数（次）",
    fontproperties=label_font,
    labelpad=10
)
ax2.set_ylabel(
    "换电总次数（次）",
    fontproperties=label_font,
    labelpad=10
)

ax1.set_xlim(67, 103)
ax1.set_xticks(capacity)

ax1.set_ylim(130, 175)
ax1.set_yticks(np.arange(130, 176, 10))

ax2.set_ylim(15, 35)
ax2.set_yticks(np.arange(15, 36, 5))

ax1.tick_params(
    axis="both", direction="in", labelsize=12,
    length=4, width=1, top=False, right=False
)
ax2.tick_params(
    axis="y", direction="in", labelsize=12,
    length=4, width=1
)

# 避免两个坐标轴的边框重复叠加
for spine in ax1.spines.values():
    spine.set_color("0.35")

for name in ["left", "bottom", "top"]:
    ax2.spines[name].set_visible(False)

ax1.spines["right"].set_visible(False)
ax2.spines["right"].set_color("0.35")

ax1.grid(False)
ax2.grid(False)

# ==================== 5. 数据标注 ====================
for x, y in zip(capacity, total_tasks):
    ax1.annotate(
        str(y),
        xy=(x, y),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11
    )

for x, y in zip(capacity, total_swaps):
    ax2.annotate(
        str(y),
        xy=(x, y),
        xytext=(0, -10),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=11
    )

# ==================== 6. 合并图例 ====================
ax1.legend(
    handles=[line1, line2],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=False,
    prop=legend_font,
    handlelength=2.8,
    columnspacing=2.0
)

# ==================== 7. 保存图片 ====================
fig.tight_layout()

output_path = Path("电池容量敏感性分析.png").resolve()
fig.savefig(
    output_path,
    format="png",
    backend="agg",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    transparent=False
)

print(f"图片已保存至：{output_path}")
plt.show()