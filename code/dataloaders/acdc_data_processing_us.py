import glob
import os
import h5py
import numpy as np
import SimpleITK as sitk


def resample_image(sitk_image, new_size=(512, 512), is_label=False):
    original_size = sitk_image.GetSize()  # 例如 (width, height) 或 (width, height, depth)

    if len(original_size) == 3:
        new_size_full = list(new_size) + [original_size[2]]
    elif len(original_size) == 2:
        new_size_full = list(new_size)
    else:
        raise ValueError("Unsupported image dimensions: {}".format(len(original_size)))

    original_spacing = sitk_image.GetSpacing()
    if len(original_size) == 3:
        new_spacing = [original_size[i] * original_spacing[i] / new_size_full[i] for i in range(2)]
        new_spacing.append(original_spacing[2])
    elif len(original_size) == 2:
        new_spacing = [original_size[i] * original_spacing[i] / new_size_full[i] for i in range(2)]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size_full)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)

    return resample.Execute(sitk_image)


case_num = 0
mask_path = sorted(glob.glob("/home/colin/Mamba-UNet-main/data/Colin3/resource2/*.nii.gz"))

for case in mask_path:
    # 读取图像，并进行重采样（x,y 方向缩放至512×512，z方向保持不变）
    img_itk = sitk.ReadImage(case)
    img_itk = resample_image(img_itk, new_size=(512, 512), is_label=False)
    image = sitk.GetArrayFromImage(img_itk)  # 得到 numpy 数组，shape 为 (depth, 512, 512)

    msk_path = case.replace("image", "label").replace(".nii.gz", "_gt.nii.gz")
    if os.path.exists(msk_path):
        print(msk_path)
        msk_itk = sitk.ReadImage(msk_path)
        msk_itk = resample_image(msk_itk, new_size=(512, 512), is_label=True)
        mask = sitk.GetArrayFromImage(msk_itk)

        # 对图像进行归一化：均值为0，方差为1
        image = image.astype(np.float32)
        mean = np.mean(image)
        std = np.std(image)
        # 为防止 std 为 0，可以加上一个极小值 epsilon
        epsilon = 1e-8
        image = (image - mean) / (std + epsilon)

        item = case.split("/")[-1].split(".")[0]

        if image.shape != mask.shape:
            print(f"Shape mismatch in case: {item}")
            continue  # 跳过形状不匹配的 case

        # 保存为 HDF5 文件，保存完整的3D数据
        with h5py.File(f'/home/colin/Mamba-UNet-main/data/Colin3/data/{item}.h5', 'w') as f:
            f.create_dataset('image', data=image, compression="gzip")
            f.create_dataset('label', data=mask, compression="gzip")

        case_num += 1
        print(f"Processed case: {item} with shape {image.shape}")

print("\nConversion completed!")
print(f"Total converted 3D volumes: {case_num}")
