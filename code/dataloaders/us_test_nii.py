import numpy as np
import nibabel as nib

if __name__ == "__main__":
 image_path = "/media/colin/My Passport/数据集/M&Ms心脏CMR分割数据集/数据集/数据集/Training/Labeled/A1D0Q7/A1D0Q7_sa.nii.gz"
 image_obj = nib.load(image_path)
 image_data = image_obj.get_fdata()

 # 处理4D数据（假设维度顺序为：x, y, z, t）
 height, width, depth, timepoints = image_data.shape  # 解包4个维度
 print(f"Image dimensions - height: {height}, width: {width}, depth: {depth}, timepoints: {timepoints}")

 # 如果要验证数据维度（兼容3D/4D）：
 print("\n完整维度信息:", image_data.shape)
 print("数据类型:", image_data.dtype)

 # 访问第1个时间点的3D数据示例
 first_volume = image_data[..., 0]  # 等同于 image_data[:, :, :, 0]
