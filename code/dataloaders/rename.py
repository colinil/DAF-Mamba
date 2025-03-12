import os


def add_suffix_to_h5_files(directory, suffix='_frame06'):
    """
    给指定目录下的所有.h5文件名末尾添加指定后缀。

    参数:
        directory (str): 目标文件夹路径。
        suffix (str): 要添加的后缀，默认为'_frame00'。
    """
    # 获取目录中的所有文件
    files = os.listdir(directory)

    # 遍历文件列表
    for file in files:
        # 检查文件是否以'.h5'结尾
        if file.endswith('.h5'):
            # 构造旧文件路径和新文件路径
            old_file_path = os.path.join(directory, file)
            new_file_name = file.replace('.h5', f'{suffix}.h5')
            new_file_path = os.path.join(directory, new_file_name)

            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"已将文件 '{file}' 重命名为 '{new_file_name}'")


import os


def rename_frame_suffix(directory, old_suffix='_frame00', new_suffix='_frame01'):
    """
    将指定目录下所有.h5文件名末尾的旧后缀替换为新后缀。

    参数:
        directory (str): 目标文件夹路径。
        old_suffix (str): 要替换的旧后缀，默认为'_frame00'。
        new_suffix (str): 替换后的新后缀，默认为'_frame01'。
    """
    # 获取目录中的所有文件
    files = os.listdir(directory)

    # 遍历文件列表
    for file in files:
        # 检查文件是否以'.h5'结尾且包含指定的旧后缀
        if file.endswith('.h5') and old_suffix in file:
            # 构造旧文件路径和新文件路径
            old_file_path = os.path.join(directory, file)
            new_file_name = file.replace(old_suffix, new_suffix)
            new_file_path = os.path.join(directory, new_file_name)

            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"已将文件 '{file}' 重命名为 '{new_file_name}'")


if __name__ == "__main__":
 folder_path = '/home/colin/Mamba-UNet-main/data/MnMs/resource'  # 替换为你的文件夹路径
 add_suffix_to_h5_files(folder_path)
 #rename_frame_suffix(folder_path)
