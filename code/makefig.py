import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 设置 Seaborn 样式，让图表更美观
sns.set_style("whitegrid")  # 还可尝试 "darkgrid", "white", "ticks" 等
sns.set_context("talk")     # 调整整体文字大小，"paper"、"notebook"、"talk"、"poster"等

# 2. 指定包含 CSV 文件的文件夹路径
folder_path = "/home/colin/Mamba-UNet-main/model/HSMI/hd95/gai"  # 替换成实际的文件夹路径

# 3. 获取文件夹中所有 CSV 文件的完整路径
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# 4. 读取并合并所有 CSV 文件
data_frames = []
for file in csv_files:
    df = pd.read_csv(file)
    # 去掉 .csv 后缀
    filename = os.path.splitext(os.path.basename(file))[0]
    df['filename'] = filename
    data_frames.append(df)

all_data = pd.concat(data_frames, ignore_index=True)

# 5. 计算每个文件中averageDice的中位数和IQR（四分位距）
stats = all_data.groupby("filename")["hd95_avg"].agg(
    median="median",
    Q1=lambda x: x.quantile(0.25),
    Q3=lambda x: x.quantile(0.75)
)
stats["IQR"] = stats["Q3"] - stats["Q1"]

# 输出每个文件的中位数和IQR
print("每个文件的中位数和 IQR:")
print(stats[["median", "IQR"]])

# 6. 绘制箱型图
plt.figure(figsize=(8, 16))  # 调整画布大小
ax = sns.boxplot(
    x='filename',
    y='hd95_avg',
    data=all_data,
    palette='Set2',     # 可以换成 "Set1", "Set3", "Pastel1" 等
    width=0.6,
    fliersize=3,        # 异常点大小
    linewidth=1.5       # 线条粗细
)

# 叠加散点图以展示数据分布
sns.stripplot(
    x='filename',
    y='hd95_avg',
    data=all_data,
    color='black',
    size=3,
    jitter=True
)

# 7. 调整坐标轴和标题
ax.set_xlabel("文件名", fontsize=12)
ax.set_ylabel("HD95_Average", fontsize=12)
ax.set_title("HSCT", fontsize=14)

# 8. 如果文件名较长，可旋转刻度标签
plt.xticks(rotation=45, ha='right')

# 9. 优化布局并显示图像
plt.tight_layout()

output_dir = "/home/colin/Mamba-UNet-main/model/可视化/箱型图"  # 替换成你想要保存的目录
os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在则创建
output_path = os.path.join(output_dir, "HSCT_HD95_Average_boxplot.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # dpi=300 适合高分辨率输出

plt.show()
