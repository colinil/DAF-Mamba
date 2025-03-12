import os
import h5py


def process_h5_file(h5_file_path, output_dir):
    """
    处理单个 h5 文件：
    1. 读取 'image' 与 'label' 数据集，假定其形状为 (100,320,320)。
    2. 遍历深度方向的每个切片（共100片），提取出 (320,320) 的二维切片。
    3. 将每个切片保存为一个新的 h5 文件，文件名格式为 "{原文件名}_slice_{序号}.h5"。
    """
    # 打开原始 h5 文件
    with h5py.File(h5_file_path, 'r') as f:
        image_dataset = f['image']  # shape 应为 (100,320,320)
        label_dataset = f['label']  # shape 应为 (100,320,320)

        # 遍历每个切片
        num_slices = image_dataset.shape[0]
        base_filename = os.path.splitext(os.path.basename(h5_file_path))[0]
        for i in range(num_slices):
            # 提取第 i 个切片
            image_slice = image_dataset[i, :, :]  # (320,320)
            label_slice = label_dataset[i, :, :]  # (320,320)

            # 构造新的文件名，使用三位数字格式化索引
            new_filename = f"{base_filename}_slice_{i:03d}.h5"
            output_h5_path = os.path.join(output_dir, new_filename)

            # 保存切片到新的 h5 文件中
            with h5py.File(output_h5_path, 'w') as new_f:
                new_f.create_dataset('image', data=image_slice)
                new_f.create_dataset('label', data=label_slice)
            print(f"保存切片 {i} 到 {output_h5_path}")


def main(input_dir, output_dir):
    """
    遍历 input_dir 目录下的所有 h5 文件，对每个文件调用 process_h5_file 进行切片保存。
    如果 output_dir 不存在，则自动创建。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历 input_dir 下所有扩展名为 .h5 的文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.h5'):
            h5_file_path = os.path.join(input_dir, filename)
            print(f"正在处理: {h5_file_path}")
            process_h5_file(h5_file_path, output_dir)


if __name__ == '__main__':
    # 修改为实际的 h5 文件所在目录
    input_directory = '/home/colin/Mamba-UNet-main/data/MnMs/resource'
    # 修改为保存切片 h5 文件的目标目录
    output_directory = '/home/colin/Mamba-UNet-main/data/MnMs/valresource'

    main(input_directory, output_directory)
