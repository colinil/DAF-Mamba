import nibabel as nib
img = nib.load("/media/colin/My Passport/数据集/M&Ms心脏CMR分割数据集/数据集/数据集/Training/Labeled/A0S9V9/A0S9V9_sa.nii.gz")
spacing = img.header.get_zooms()[:2]  # 2D 图像取前两个值
print(spacing)

import SimpleITK as sitk
image = sitk.ReadImage("/media/colin/My Passport/数据集/HSMI/Training Set/a001/image.mhd")
spacing = image.GetSpacing()  # 返回 (x, y, z) 方向的 spacing
print(spacing)