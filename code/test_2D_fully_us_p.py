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
from thop import profile
from torchsummary import summary

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
    # 确保二值化
    pred = pred.copy()  # 避免对原始数据直接修改
    gt = gt.copy()
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    # 已有指标
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)

    # 计算 IoU
    intersection = np.sum(np.logical_and(pred, gt))
    union = np.sum(np.logical_or(pred, gt))
    iou = intersection / union if union != 0 else 1.0

    # 计算混淆矩阵中的各项指标
    TP = intersection
    FP = np.sum(np.logical_and(pred == 1, gt == 0))
    FN = np.sum(np.logical_and(pred == 0, gt == 1))
    TN = np.sum(np.logical_and(pred == 0, gt == 0))
    total = TP + TN + FP + FN

    acc = (TP + TN) / total if total != 0 else 1.0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 1.0
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 1.0  # 又称 Recall
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

    # 分别计算三个类别的指标（假设标签中类别为 1, 2, 3）
    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    # 保存 nii 格式图像
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, os.path.join(test_save_path, case + "_pred.nii.gz"))
    sitk.WriteImage(img_itk, os.path.join(test_save_path, case + "_img.nii.gz"))
    sitk.WriteImage(lab_itk, os.path.join(test_save_path, case + "_gt.nii.gz"))
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(os.path.join(FLAGS.root_path, 'test.list'), 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])

    snapshot_path = "../model/{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "../model/{}_{}_labeled/{}_predictions999-/".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)

    save_mode_path = os.path.join(snapshot_path, '{}_best_model_rate0.8.pth'.format(FLAGS.model))
    print("Loading model from:", save_mode_path)
    net.load_state_dict(torch.load(save_mode_path))
    print("Init weight from {}".format(save_mode_path))
    net.eval()

    # 计算参数量
    total_params = sum(p.numel() for p in net.parameters())
    print(f"模型参数量: {total_params / 1e6:.2f}M")

    # 计算FLOPs
    input_tensor = torch.randn(1, 1, 224, 224).cuda()  # 假设输入尺寸为224x224
    flops, params = profile(net, inputs=(input_tensor,))
    print(f"模型FLOPs: {flops / 1e9:.2f}G")

    # 累计每个类别的各项指标（每个 metric 都是包含8个值的向量）
    first_total = np.zeros(8)
    second_total = np.zeros(8)
    third_total = np.zeros(8)
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric_first = first_total / len(image_list)
    avg_metric_second = second_total / len(image_list)
    avg_metric_third = third_total / len(image_list)
    avg_metric = [avg_metric_first, avg_metric_second, avg_metric_third]
    return avg_metric



if __name__ == '__main__':
    # 假设 parser 已经定义好并解析了命令行参数
    FLAGS = parser.parse_args()
    metric_result = Inference(FLAGS)
    print("各类别平均指标（顺序为 Dice, ASD, HD95, IoU, Acc, Pre, Sen, Spe）：")
    print(metric_result)
    # 若需要计算三个类别的总体平均指标，可如下处理（逐项平均）：
    overall_avg = (np.asarray(metric_result[0]) + np.asarray(metric_result[1]) + np.asarray(metric_result[2])) / 3.0
    print("总体平均指标：")
    print(overall_avg)
