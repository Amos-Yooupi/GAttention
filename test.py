import matplotlib.pyplot as plt
import numpy as np

# 从表格提取消融实验数据（Time-aware expert对比）
# 数据结构：数据集→指标→是否使用Time-aware expert→[预测长度12, 24, 36]的(均值, 标准差)
pred_lengths = [12, 24, 36]  # 预测长度，与表格一致
datasets = ['PEMS04', 'PEMS08']  # 数据集顺序
metrics = ['MSE', 'MAE']  # 评估指标

# 解析表格数据（均值从表格数值提取，标准差根据表格±后数值计算）
data = {
    'PEMS04': {
        'MSE': {
            'Expert Enabled': ([0.0839, 0.1257, 0.1996], [0.0006, 0.0009, 0.0124]),
            'Expert Disable': ([0.0971, 0.2510, 0.4587], [0.0035, 0.0397, 0.0378])
        },
        'MAE': {
            'Expert Enabled': ([0.2123, 0.2619, 0.3230], [0.0018, 0.0063, 0.0059]),
            'Expert Disable': ([0.2321, 0.3716, 0.5089], [0.0065, 0.0198, 0.0237])
        }
    },
    'PEMS08': {
        'MSE': {
            'Expert Enabled': ([0.0942, 0.1176, 0.1215], [0.0115, 0.0150, 0.0110]),
            'Expert Disable': ([0.1372, 0.3084, 0.5432], [0.0005, 0.0340, 0.0250])
        },
        'MAE': {
            'Expert Enabled': ([0.2207, 0.2544, 0.2549], [0.0115, 0.0180, 0.0150]),
            'Expert Disable': ([0.2655, 0.4183, 0.5581], [0.0070, 0.0310, 0.0200])
        }
    }
}

# 保持1行4子图布局，与原代码尺寸一致
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# 样式配置：区分“使用/不使用Time-aware expert”，延续原代码的视觉权重（加粗线条、清晰标记）
styles = {
    'Expert Enabled': {  # 使用时间感知专家（表格中加粗的最优结果）
        'color': '#2ecc71',  # 绿色：突出最优方案
        'marker': 'o',
        'linestyle': '-',
        'label': 'Expert Enabled',
        'linewidth': 5,
        'markersize': 10
    },
    'Expert Disable': {  # 不使用（替换为Feed Forward）
        'color': '#e74c3c',  # 红色：对比方案
        'marker': 's',
        'linestyle': '--',
        'label': 'Expert Disable',
        'linewidth': 5,
        'markersize': 10
    }
}

# 填充子图数据
subplot_idx = 0
for ds in datasets:
    for metric in metrics:
        ax = axes[subplot_idx]
        # 绘制两种方案的趋势线（带误差棒，反映标准差）
        for expert_status in ['Expert Enabled', 'Expert Disable']:
            means = data[ds][metric][expert_status][0]
            stds = data[ds][metric][expert_status][1]
            ax.errorbar(
                pred_lengths, means, yerr=stds,
                color=styles[expert_status]['color'],
                marker=styles[expert_status]['marker'],
                linestyle=styles[expert_status]['linestyle'],
                capsize=5,  # 误差棒帽子大小
                linewidth=styles[expert_status]['linewidth'],
                markersize=styles[expert_status]['markersize'],
                label=styles[expert_status]['label'],
                elinewidth=4,  # 误差棒线条宽度
                capthick=4     # 误差棒帽子厚度
            )

        # 子图配置：完全对齐原代码的字体、刻度、网格样式
        ax.set_title(f'{ds} - {metric}', fontsize=30, fontweight='bold')
        ax.set_xlabel('Prediction Length', fontsize=26)
        ax.tick_params(
            axis='both',
            labelsize=24,
            direction='in',  # 刻度向内
            length=10,       # 刻度长度
            width=3.5        # 刻度宽度
        )
        ax.set_xticks(pred_lengths)  # x轴仅显示表格中的预测长度
        ax.grid(True, linestyle='--', linewidth=2.5, alpha=0.8)  # 网格线样式
        ax.legend(fontsize=18, frameon=True, edgecolor='gray')  # 图例配置

        subplot_idx += 1

# 布局调整：保持与原代码一致，避免标题/图例被截断
plt.tight_layout()
plt.subplots_adjust(top=0.82)
# 保存图片（高分辨率，适配论文使用）
plt.savefig("time_aware.png", dpi=300, bbox_inches='tight')
print("Done!")
plt.show()