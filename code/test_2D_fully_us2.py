import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
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
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')



def calculate_metric_percase(pred, gt):
    """
    计算单个病例的各项指标，包括：
    Dice, ASD, HD95, IoU, Accuracy, Precision, Sensitivity, Specificity
    """
    pred = pred.copy()
    gt = gt.copy()
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    dice = metric.binary.dc(pred, gt)
    spacing_3d = (1.25, 1.25, 1.37)
    asd = metric.binary.asd(pred, gt, voxelspacing=spacing_3d)
    hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing_3d)
    #asd = metric.binary.asd(pred, gt) if np.any(pred) and np.any(gt) else np.inf
    #hd95 = metric.binary.hd95(pred, gt) if np.any(pred) and np.any(gt) else np.inf

    intersection = np.sum(np.logical_and(pred, gt))
    union = np.sum(np.logical_or(pred, gt))
    iou = intersection / union if union != 0 else 1.0

    TP = intersection
    FP = np.sum(np.logical_and(pred == 1, gt == 0))
    FN = np.sum(np.logical_and(pred == 0, gt == 1))
    TN = np.sum(np.logical_and(pred == 0, gt == 0))
    total = TP + TN + FP + FN

    acc = (TP + TN) / total if total != 0 else 1.0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 1.0
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 1.0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 1.0

    return dice, asd, hd95, iou, acc, precision, sensitivity, specificity



def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(os.path.join(FLAGS.root_path, "data/{}.h5".format(case)), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        # 将 slice resize 到 224*224
        slice = zoom(slice, (224 / x, 224 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            # 计算 softmax 后取 argmax 得到分割结果
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            # 将预测结果 resize 回原来的大小
            pred = zoom(out, (x / 224, y / 224), order=0)
            prediction[ind] = pred

    # 只计算第一个类别的指标（即 prediction == 1）
    first_metric = calculate_metric_percase(prediction == 1, label == 1)

    # 保存 nii 格式图像
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 100))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 100))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 100))
    sitk.WriteImage(prd_itk, os.path.join(test_save_path, case + "_pred.nii.gz"))
    sitk.WriteImage(img_itk, os.path.join(test_save_path, case + "_img.nii.gz"))
    sitk.WriteImage(lab_itk, os.path.join(test_save_path, case + "_gt.nii.gz"))
    return first_metric  # 只返回第一个类别的指标



def Inference(FLAGS):
    with open(os.path.join(FLAGS.root_path, 'test.list'), 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])

    snapshot_path = "../model/{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "../model/{}_{}_labeled/{}_predictions-2/".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)

    save_mode_path = os.path.join(snapshot_path, '{}_best_model_notus.pth'.format(FLAGS.model))
    print("Loading model from:", save_mode_path)
    net.load_state_dict(torch.load(save_mode_path))
    print("Init weight from {}".format(save_mode_path))
    net.eval()

    # 累计第一个类别的各项指标（每个 metric 都是包含8个值的向量）
    first_total = np.zeros(8)
    for case in tqdm(image_list):
        first_metric = test_single_volume(case, net, test_save_path, FLAGS)  # 只返回第一个类别的指标
        first_total += np.asarray(first_metric)

    avg_metric_first = first_total / len(image_list)

    return avg_metric_first  # 只返回第一个类别的平均指标


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric_result = Inference(FLAGS)
    print("第一个类别的平均指标（顺序为 Dice, ASD, HD95, IoU, Acc, Pre, Sen, Spe）：")
    print(metric_result)
