import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pathlib import Path

# ==================== 1. 数据 ====================
swap_time = np.array([8, 9, 10, 11, 12])

# 按两段线性拟合残差平方和最小准则计算
optimal_trucks = np.array([23, 21, 19, 16, 15])

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

# ==================== 3. 绘制曲线 ====================
fig, ax = plt.subplots(figsize=(7.2, 4.8))

ax.plot(
    swap_time,
    optimal_trucks,
    color="black",
    linestyle="-",
    linewidth=1.8,
    marker="o",
    markersize=6,
    markerfacecolor="white",
    markeredgewidth=1.2,
    label="最优出勤矿卡数量"
)

# ==================== 4. 坐标轴设置 ====================
ax.set_xlabel(
    "换电时间（分钟）",
    fontproperties=label_font,
    labelpad=10
)

ax.set_ylabel(
    "最优出勤矿卡数量（辆）",
    fontproperties=label_font,
    labelpad=10
)

ax.set_xlim(7.7, 12.3)
ax.set_xticks(swap_time)

ax.set_ylim(13, 25)
ax.set_yticks(np.arange(13, 26, 2))

ax.tick_params(
    axis="both",
    direction="in",
    labelsize=12,
    length=4,
    width=1,
    top=False,
    right=False
)

for spine in ax.spines.values():
    spine.set_color("0.35")

ax.grid(False)

# ==================== 5. 数据标注 ====================
for x, y in zip(swap_time, optimal_trucks):
    ax.annotate(
        str(y),
        xy=(x, y),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11
    )

# ==================== 6. 图例 ====================
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    frameon=False,
    prop=legend_font,
    handlelength=2.8
)

# ==================== 7. 保存图片 ====================
fig.tight_layout()

output_path = Path("换电时间敏感性分析.png").resolve()

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