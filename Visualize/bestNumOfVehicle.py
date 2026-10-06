import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ==================== 字体设置 ====================
available_fonts = {
    font.name for font in font_manager.fontManager.ttflist
}

chinese_font = next(
    (name for name in [
        "SimSun", "Songti SC", "Noto Serif CJK SC",
        "Source Han Serif SC", "Microsoft YaHei", "SimHei"
    ] if name in available_fonts),
    None
)

if chinese_font is None:
    raise RuntimeError("请先安装宋体或 Noto Serif CJK SC 等中文字体。")

english_font = next(
    (name for name in [
        "Times New Roman", "Liberation Serif", "DejaVu Serif"
    ] if name in available_fonts),
    "DejaVu Serif"
)

plt.rcParams.update({
    "font.family": [english_font, chinese_font],
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 11,
    "axes.unicode_minus": False,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ==================== 数据 ====================
truck_count = np.arange(1, 31)

total_tasks = np.array([
    9, 18, 27, 36, 44, 52, 60, 68, 76, 84,
    92, 100, 108, 116, 124, 132, 140, 148, 156, 162,
    168, 173, 178, 183, 188, 193, 198, 203, 208, 213
])

average_tasks = total_tasks / truck_count
marginal_tasks = np.diff(total_tasks, prepend=0)

# ==================== 绘图 ====================
cm = 1 / 2.54
fig, ax = plt.subplots(figsize=(15 * cm, 10 * cm))

# 黑色实线：车均搬运次数
ax.plot(
    truck_count,
    average_tasks,
    color="black",
    linewidth=1.5,
    linestyle="-",
    label="车均搬运次数",
    zorder=2
)

# 黑色虚线：边际搬运次数
# 前4个点与实线重合，白色描边用于保持虚线可辨识
import matplotlib.patheffects as pe

ax.plot(
    truck_count,
    marginal_tasks,
    color="black",
    linewidth=1.5,
    linestyle=(0, (5, 4)),
    label="边际搬运次数",
    path_effects=[
        pe.Stroke(linewidth=3.0, foreground="white"),
        pe.Normal()
    ],
    zorder=3
)

# 坐标轴
ax.set_xlabel("出勤矿卡数量（辆）", labelpad=5, fontsize=12)
ax.set_ylabel("搬运次数（次）", labelpad=5, fontsize=12)

ax.set_xlim(0, 31)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30])

# 聚焦数据变化范围，并明确显示刻度
ax.set_ylim(2.5, 11)
ax.set_yticks(np.arange(3, 11, 1))

# 完整边框，无网格
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.9)
    spine.set_color("0.35")

ax.grid(False)

ax.tick_params(
    axis="both",
    direction="in",
    length=3,
    width=0.8,
    top=False,
    right=False,
    pad=6
)

# 顶部横排图例
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 0.99),
    ncol=2,
    frameon=False,
    handlelength=2.8,
    handletextpad=0.5,
    columnspacing=2.0,
    fontsize=12
)

fig.tight_layout(pad=1.0)

# 保存
fig.savefig("车挖配比.png", dpi=600, bbox_inches="tight")

plt.show()