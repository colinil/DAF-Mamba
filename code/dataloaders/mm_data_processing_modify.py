import os
import nibabel as nib
import numpy as np
import h5py
from scipy import ndimage


def process_and_save(base_dir, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历基目录下的所有子文件夹
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)

        # 跳过非目录文件
        if not os.path.isdir(folder_path):
            continue

        # 查找需要的文件
        image_path, label_path = None, None
        for f in os.listdir(folder_path):
            if f.endswith('_sa.nii.gz'):
                image_path = os.path.join(folder_path, f)
            elif f.endswith('_sa_gt.nii.gz'):
                label_path = os.path.join(folder_path, f)

        # 检查文件是否存在
        if not image_path or not label_path:
            print(f"Skipping {folder_name}, missing files")
            continue

        try:
            # 加载NIfTI文件
            img_nii = nib.load(image_path)
            label_nii = nib.load(label_path)

            # 获取数据数组
            img_data = img_nii.get_fdata()
            label_data = label_nii.get_fdata()

            # 处理维度：添加时间维度如果是3D
            if img_data.ndim == 3:
                img_data = img_data[..., np.newaxis]
            if label_data.ndim == 3:
                label_data = label_data[..., np.newaxis]

            # 验证维度
            if img_data.ndim != 4 or label_data.ndim != 4:
                raise ValueError("Invalid data dimensions")

            # 计算缩放因子
            original_shape = img_data.shape
            scale_factors = (
                224 / original_shape[0],
                224 / original_shape[1],
                10 / original_shape[2],
                1  # 时间维度保持不变
            )

            # 缩放图像（三次样条插值）
            scaled_img = ndimage.zoom(img_data, scale_factors, order=3)
            # 缩放标签（最近邻插值）
            scaled_label = ndimage.zoom(label_data, scale_factors, order=0)

            # 取第一个时间点并转换为3D
            image_3d = scaled_img[..., 7]
            label_3d = scaled_label[..., 7]

            # 异常值截断（仅图像）
            lower = np.percentile(image_3d, 5)
            upper = np.percentile(image_3d, 95)
            clipped = np.clip(image_3d, lower, upper)

            # 标准化处理
            normalized = (clipped - clipped.mean()) / clipped.std()

            # 创建HDF5文件
            output_path = os.path.join(output_dir, f"{folder_name}.h5")
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('image', data=normalized.astype(np.float32), compression="gzip")
                f.create_dataset('label', data=label_3d.astype(np.uint8), compression="gzip")

            print(f"Processed {folder_name} successfully")

        except Exception as e:
            print(f"Error processing {folder_name}: {str(e)}")


if __name__ == "__main__":
    # 配置路径
    base_directory = "/media/colin/My Passport/数据集/M&Ms心脏CMR分割数据集/数据集/数据集/Training/Labeled"  # 原始数据目录
    output_directory = "/home/colin/Mamba-UNet-main/data/MnMs/resource"  # 输出目录

    # 执行处理
    process_and_save(base_directory, output_directory)

