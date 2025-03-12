import argparse
import os
import shutil
import csv

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')


def test_single_volume(case, net, test_save_path, FLAGS):
    # 读取h5文件中的图像和标签
    h5f = h5py.File(os.path.join(FLAGS.root_path, "data/{}.h5".format(case)), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)

    # 对每个切片进行预测
    for ind in range(image.shape[0]):
        slice_img = image[ind, :, :]
        x, y = slice_img.shape[0], slice_img.shape[1]
        # 将切片resize到224*224
        slice_resized = zoom(slice_img, (224 / x, 224 / y), order=0)
        input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input_tensor)
            else:
                out_main = net(input_tensor)
            # 计算 softmax 后取 argmax 得到分割结果
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            # 将预测结果resize回原始大小
            pred_slice = zoom(out, (x / 224, y / 224), order=0)
            prediction[ind] = pred_slice

    # 对每个切片计算Dice指标
    slice_dice = []
    for ind in range(image.shape[0]):
        pred_slice = prediction[ind]
        label_slice = label[ind]
        # 二值化处理：只关注类别1
        pred_binary = (pred_slice == 1)
        label_binary = (label_slice == 1)
        # 如果预测和标签均为空，则认为Dice为1.0，否则计算Dice
        if not np.any(pred_binary) and not np.any(label_binary):
            dice_val = 1.0
        else:
            dice_val = metric.binary.dc(pred_binary, label_binary)
        slice_dice.append(dice_val)

    # 保存nii格式图像
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 100))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 100))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 100))
    sitk.WriteImage(prd_itk, os.path.join(test_save_path, case + "_pred.nii.gz"))
    sitk.WriteImage(img_itk, os.path.join(test_save_path, case + "_img.nii.gz"))
    sitk.WriteImage(lab_itk, os.path.join(test_save_path, case + "_gt.nii.gz"))

    # 返回该3D图像中每个切片的Dice值
    return slice_dice


def Inference(FLAGS):
    # 读取测试文件列表
    with open(os.path.join(FLAGS.root_path, 'test.list'), 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])

    snapshot_path = "../model/{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)

    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    print("Loading model from:", save_mode_path)
    net.load_state_dict(torch.load(save_mode_path))
    print("Init weight from {}".format(save_mode_path))
    net.eval()

    # 用于保存每个测试文件的每个切片的Dice值，格式为 (case, slice_index, dice)
    dice_list = []

    for case in tqdm(image_list):
        slice_dice = test_single_volume(case, net, test_save_path, FLAGS)
        for idx, dice_val in enumerate(slice_dice):
            dice_list.append((case, idx, dice_val))

    # 保存Dice结果为CSV文件
    csv_save_path = os.path.join(test_save_path, "dice_results.csv")
    with open(csv_save_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["case", "slice_index", "averageDice"])
        for case, slice_idx, dice in dice_list:
            csv_writer.writerow([case, slice_idx, dice])
    print("Dice results saved to:", csv_save_path)

    return


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    Inference(FLAGS)
