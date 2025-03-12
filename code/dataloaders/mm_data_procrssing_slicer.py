import os
import h5py
import numpy as np


def slice_h5_files(input_dir, output_dir, list_file_path):
    """
    将3D HDF5文件切割为2D切片并生成名称列表的函数
    参数：
        input_dir: 输入HDF5文件目录
        output_dir: 切片输出目录
        list_file_path: 切片名称列表文件路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 清空或创建列表文件
    open(list_file_path, 'w').close()

    # 遍历输入目录中的所有HDF5文件
    for filename in os.listdir(input_dir):
        if not filename.endswith('.h5'):
            continue

        file_path = os.path.join(input_dir, filename)

        try:
            with h5py.File(file_path, 'r') as f:
                # 读取图像和标签数据
                image = f['image'][:]
                label = f['label'][:]

                # 验证数据维度
                if image.ndim != 3 or label.ndim != 3:
                    print(f"跳过 {filename}: 无效维度 (应为3D)")
                    continue

                if image.shape != (224, 224, 10) or label.shape != (224, 224, 10):
                    print(f"跳过 {filename}: 尺寸不匹配 (应为224x224x32)")
                    continue

                # 提取基本文件名
                base_name = os.path.splitext(filename)[0]

                # 遍历所有切片
                for slice_idx in range(image.shape[2]):
                    # 生成切片名称和文件路径
                    slice_name = f"{base_name}_slice_{slice_idx:02d}"
                    slice_filename = f"{slice_name}.h5"
                    slice_path = os.path.join(output_dir, slice_filename)

                    # 提取单个切片
                    image_slice = image[:, :, slice_idx]
                    label_slice = label[:, :, slice_idx]

                    # 保存切片文件
                    with h5py.File(slice_path, 'w') as slice_file:
                        slice_file.create_dataset('image',
                                                  data=image_slice.astype(np.float32),
                                                  compression="gzip")
                        slice_file.create_dataset('label',
                                                  data=label_slice.astype(np.uint8),
                                                  compression="gzip")

                    # 写入列表文件（追加模式）
                    with open(list_file_path, 'a') as list_file:
                        list_file.write(f"{slice_name}\n")

            print(f"成功处理: {filename} -> 10个切片")

        except Exception as e:
            print(f"处理 {filename} 时发生错误: {str(e)}")



if __name__ == "__main__":
    # 配置路径
    input_directory = "/home/colin/Mamba-UNet-main/data/MnMs/resource"  # 原始HDF5文件目录
    output_directory = "/home/colin/Mamba-UNet-main/data/MnMs/valresource"  # 切片输出目录
    list_directory="/home/colin/Mamba-UNet-main/data/MnMs"
    list_file = os.path.join(list_directory, "sum.list")  # 列表文件路径

    # 执行切割操作
    slice_h5_files(input_directory, output_directory, list_file)
