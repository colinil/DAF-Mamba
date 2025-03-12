import os
import SimpleITK as sitk
import numpy as np
import h5py
from skimage.transform import resize


def process_folder(folder_path, target_shape=(100, 320, 320)):
    """
    处理单个文件夹内的 mhd 图像：
    1. 读取 'image.mhd' 与 'gt_binary.mhd'
    2. 利用 SimpleITK 得到 (depth, height, width) 顺序的 numpy 数组
    3. 统一缩放到 target_shape (100,320,320)
       - 原图采用线性插值（order=1）
       - 标签图像采用最近邻插值（order=0）
    """
    image_path = os.path.join(folder_path, 'image.mhd')
    label_path = os.path.join(folder_path, 'gt_noclip.mhd')

    if not os.path.exists(image_path) or not os.path.exists(label_path):
        print(f"文件不存在或不全: {folder_path}")
        return None, None

    # 读取 mhd 图像，SimpleITK 默认输出 (depth, height, width) 顺序
    image_itk = sitk.ReadImage(image_path)
    label_itk = sitk.ReadImage(label_path)
    image_np = sitk.GetArrayFromImage(image_itk)
    label_np = sitk.GetArrayFromImage(label_itk)

    # 缩放到目标尺寸 (100,320,320)
    image_resized = resize(image_np, target_shape, order=1, preserve_range=True, anti_aliasing=True)
    label_resized = resize(label_np, target_shape, order=0, preserve_range=True, anti_aliasing=False)

    # 数据类型转换
    return image_resized.astype(np.float32), label_resized.astype(np.uint8)


def main(input_dir, output_dir, target_shape=(100, 320, 320)):
    """
    遍历 input_dir 下所有子文件夹：
    - 每个文件夹内的图像数据分别读取、缩放后生成单独的 h5 文件，
      文件名以文件夹名称命名，保存至 output_dir，h5 文件包含 'image' 与 'label' 两个数据集。
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历所有子目录
    for subfolder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, subfolder)
        if os.path.isdir(folder_path):
            print(f"正在处理文件夹: {subfolder}")
            image_array, label_array = process_folder(folder_path, target_shape)
            if image_array is not None and label_array is not None:
                output_h5_path = os.path.join(output_dir, f"{subfolder}.h5")
                with h5py.File(output_h5_path, 'w') as h5f:
                    h5f.create_dataset('image', data=image_array)
                    h5f.create_dataset('label', data=label_array)
                print(f"h5 文件已保存至: {output_h5_path}")
            else:
                print(f"跳过文件夹 {subfolder}，数据缺失。")


if __name__ == '__main__':
    # 修改为实际的输入目录路径，目录下包含多个子文件夹
    input_directory = '/media/colin/My Passport/数据集/HSMI/Testing Set'
    # 修改为输出 h5 文件保存的目录
    output_directory = '/home/colin/Mamba-UNet-main/data/HSMI/data'

    main(input_directory, output_directory)
