import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def display_h5_images(folder_path):
    # 获取所有h5文件并按文件名排序
    h5_files = glob.glob(os.path.join(folder_path, "*.h5"))
    h5_files.sort()

    # 遍历所有h5文件
    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, 'r') as f:
                # 读取数据集（假设数据集名为'label'）
                data = f[('image')][:]

                # 处理3D数据（取第一个切片）
                if data.ndim == 3:
                    data = data[1]
                elif data.ndim != 2:
                    print(f"跳过 {os.path.basename(h5_path)}: 不支持的维度 {data.ndim}D")
                    continue

                # 数据预处理
                data = np.clip(data, 0, 255).astype(np.uint8)

                # 创建图像显示
                plt.figure(figsize=(8, 6))
                plt.imshow(data, cmap='gray')
                plt.axis('off')
                plt.title(os.path.basename(h5_path))
                plt.show()

        except Exception as e:
            print(f"处理文件 {os.path.basename(h5_path)} 失败: {str(e)}")
            continue


if __name__ == "__main__":
    # 设置包含h5文件的文件夹路径
    target_folder = "/home/colin/Mamba-UNet-main/data/MnMs/data"

    # 检查文件夹是否存在
    if not os.path.isdir(target_folder):
        print(f"错误：文件夹 {target_folder} 不存在")
    else:
        display_h5_images(target_folder)
