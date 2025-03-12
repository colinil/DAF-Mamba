import os
import h5py
import numpy as np

def is_black_label(label_dataset):
    """
    检查label数据集是否为全黑色。
    假设label数据集是二维的，且每个像素值为0表示黑色。
    """
    return np.all(label_dataset[:] == 0)

def delete_black_label_h5_files(folder_path):
    """
    遍历指定文件夹，删除label数据集为全黑色的h5文件。
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                try:
                    with h5py.File(file_path, 'r') as f:
                        if 'label' in f:
                            label_dataset = f['label']
                            if is_black_label(label_dataset):
                                os.remove(file_path)
                                print(f"已删除全黑色label的文件：{file_path}")
                except Exception as e:
                    print(f"处理文件{file_path}时发生错误：{e}")

if __name__ == "__main__":
 folder_path = '/home/colin/Mamba-UNet-main/data/MnMs/valresource'  # 替换为你的文件夹路径
 delete_black_label_h5_files(folder_path)
