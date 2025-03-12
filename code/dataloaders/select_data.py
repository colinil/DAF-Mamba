import os
import shutil

# 定义源目录和目标目录
source_dir = "/home/colin/文档/Resources/database_nifti"  # 存放 patient 文件夹的根目录
target_dir = "/home/colin/Mamba-UNet-main/data/Colin3/resource2"  # 指定的目标目录

# 如果目标目录不存在，则创建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历 patient0001 到 patient0500 的所有文件夹
for i in range(1, 501):
    # 构造文件夹名称，例如 patient0001, patient0002, ..., patient0500
    patient_folder = f"patient{i:04d}"
    patient_path = os.path.join(source_dir, patient_folder)

    if os.path.exists(patient_path):
        for filename in os.listdir(patient_path):
            # 检查文件名是否以指定后缀结尾
            if (filename.endswith('_ED.nii.gz') or
                    filename.endswith('_ED_gt.nii.gz') or
                    filename.endswith('_ES.nii.gz') or
                    filename.endswith('_ES_gt.nii.gz')):
                source_file = os.path.join(patient_path, filename)
                target_file = os.path.join(target_dir, filename)

                # 移动文件到目标目录，文件名保持不变
                shutil.move(source_file, target_file)
                print(f"移动文件：{source_file} --> {target_file}")
    else:
        print(f"目录不存在：{patient_path}")
