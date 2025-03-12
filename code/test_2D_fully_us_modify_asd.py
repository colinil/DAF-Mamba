import argparse
import os
import shutil
import h5py
import numpy as np
import SimpleITK as sitk
import torch
import csv
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
from networks.net_factory import net_factory
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ACDC', help='数据集根路径')
parser.add_argument('--exp', type=str, default='ACDC/Fully_Supervised', help='实验名称')
parser.add_argument('--model', type=str, default='unet', help='模型名称')
parser.add_argument('--num_classes', type=int, default=4, help='网络输出通道数')
parser.add_argument('--labeled_num', type=int, default=3, help='标记数据数量')

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
    asd = metric.binary.asd(pred, gt) if np.any(pred) and np.any(gt) else np.inf
    hd95 = metric.binary.hd95(pred, gt) if np.any(pred) and np.any(gt) else np.inf

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

def test_single_image(case, net, test_save_path, FLAGS):
    h5f = h5py.File(os.path.join(FLAGS.root_path, "data/{}.h5".format(case)), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)

    x, y = image.shape[0], image.shape[1]
    # 假设模型输入尺寸为224×224
    slice_resized = zoom(image, (224 / x, 224 / y), order=0)
    input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()

    net.eval()
    with torch.no_grad():
        if FLAGS.model == "unet_urds":
            out_main, _, _, _ = net(input_tensor)
        else:
            out_main = net(input_tensor)
        out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / 224, y / 224), order=0)
        prediction = pred

    # 计算各类别的指标（假设标签中类别为1, 2, 3）
    metrics = []
    for i in range(1, FLAGS.num_classes):
        metrics.append(calculate_metric_percase(prediction == i, label == i))

    # 保存预测结果为 nii 格式
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    sitk.WriteImage(prd_itk, os.path.join(test_save_path, case + "_pred.nii.gz"))
    sitk.WriteImage(img_itk, os.path.join(test_save_path, case + "_img.nii.gz"))
    sitk.WriteImage(lab_itk, os.path.join(test_save_path, case + "_gt.nii.gz"))
    return metrics

def Inference(FLAGS):
    with open(os.path.join(FLAGS.root_path, 'test.list'), 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])

    snapshot_path = "../model/{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "../model/{}_{}_labeled/{}_predictions222/".format(FLAGS.exp, FLAGS.labeled_num, FLAGS.model)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)

    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    print("加载模型自:", save_mode_path)
    net.load_state_dict(torch.load(save_mode_path))
    print("从 {} 初始化权重".format(save_mode_path))
    net.eval()

    total_metrics = np.zeros((FLAGS.num_classes - 1, 8))
    # 用于存储每个测试文件的 case 和 ASD 平均值
    asd_list = []
    for case in tqdm(image_list):
        metrics = test_single_image(case, net, test_save_path, FLAGS)
        asd_values = []
        for i in range(FLAGS.num_classes - 1):
            asd_value = metrics[i][0]  # ASD 值在返回指标中的索引 1
            asd_values.append(asd_value)
        # 计算三个类别的平均 ASD
        asd_avg = np.mean(asd_values)
        asd_list.append({"case": case, "Dice_Average": asd_avg})

        for i in range(FLAGS.num_classes - 1):
            total_metrics[i] += np.asarray(metrics[i])

    # 保存 ASD 平均结果到 CSV 文件，只包含 case 和 asd_avg 两栏
    csv_file = os.path.join(test_save_path, "Dice.csv")
    with open(csv_file, mode='w', newline='') as csvfile:
        fieldnames = ["case", "Dice_Average"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in asd_list:
            writer.writerow(entry)
    print(f"ASD 平均结果已保存到: {csv_file}")

    avg_metrics = total_metrics / len(image_list)
    return avg_metrics

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric_result = Inference(FLAGS)
    print("各类别平均指标（顺序为 Dice, ASD, HD95, IoU, Acc, Pre, Sen, Spe）：")
    for i, metrics in enumerate(metric_result):
        print(f"类别 {i + 1}: {metrics}")
    overall_avg = np.mean(metric_result, axis=0)
    print("总体平均指标：")
    print(overall_avg)
