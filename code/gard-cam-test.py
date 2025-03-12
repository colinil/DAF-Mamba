import torch
import torch.nn.functional as F
from torchvision import transforms
import h5py
import numpy as np
import matplotlib.pyplot as plt

from networks.unet import UNet
from networks.vision_mamba import MambaUnet as VIM_seg
from PIL import Image
from config import get_config
import argparse
from scipy.ndimage import zoom
import os
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
import re

parser = argparse.ArgumentParser()
# 参数配置
parser.add_argument('--root_path', type=str, default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str, default='efficient_unet', help='model_name')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
#parser.add_argument('--cfg', type=str, default="../code/configs/vmamba_tiny.yaml", help='path to config file')
parser.add_argument('--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs='+')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'], help='cache mode')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--max_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[512, 512], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=140, help='labeled data')
args = parser.parse_args()

config = get_config(args)

#def print_model_layers(model):
    #for name, layer in model.named_children():
        #print(f"Layer name: {name}, Layer: {layer}")


# Grad-CAM类定义
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.features = None
        self.gradients = None

        self.target_layer.register_forward_hook(self.save_features)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_features(self, module, input, output):
        self.features = output

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def compute_cam(self):
        gradients = self.gradients
        features = self.features

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * features, dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = cam.squeeze(0)
        return cam

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = VIM_seg(config, img_size=args.patch_size, num_classes=args.num_classes).to(device)
model = ViT_seg(config, img_size=args.patch_size,num_classes=args.num_classes).cuda()
#model = net_factory(net_type=args.model ,in_chns=1,class_num=4).to(device)
model.load_from(config)
model.load_state_dict(torch.load("/home/colin/Mamba-UNet-main/model/ACDC/swinunet_140_labeled/swinunet/swinunet_best_model1.pth"))
model.eval()
print(model)
# 获取模型的最后一层





# 加载.h5数据文件
#h5_file_path = '/home/colin/Mamba-UNet-main/data/MnMs/data/A1E9Q1_frame00_slice_04.h5'
#h5_file_path = '/home/colin/Mamba-UNet-main/data/MnMs/data/A2N8V0_frame00_slice_05.h5'
#h5_file_path = '/home/colin/Mamba-UNet-main/data/MnMs/data/D4N6W6_frame00_slice_08.h5'
#h5_file_path = '/home/colin/Mamba-UNet-main/data/MnMs/data/E0O0S0_frame00_slice_06.h5'

#h5_file_path = '/home/colin/Mamba-UNet-main/data/ACDC/data/slices/patient007_frame01_slice_4.h5'
h5_file_path = '/home/colin/Mamba-UNet-main/data/ACDC/data/slices/patient011_frame01_slice_3.h5'
#h5_file_path = '/home/colin/Mamba-UNet-main/data/ACDC/data/slices/patient025_frame02_slice_1.h5'
#h5_file_path = '/home/colin/Mamba-UNet-main/data/ACDC/data/slices/patient040_frame01_slice_1.h5'


match = re.search(r'([A-Za-z0-9_]+)_frame\d+_slice_\d+', h5_file_path)
if match:
    tail_field = match.group(1)

with h5py.File(h5_file_path, 'r') as f:
    image_data = f['image'][...]  # 读取数据集中的图像
    label_data = f['label'][...]  # 读取标签数据
    image = image_data[:, :]  # 选择第一个切片作为示例
    # 缩放图像
    x, y = image.shape[0], image.shape[1]
    # 假设模型输入尺寸为224×224
    image = zoom(image, (224 / x, 224 / y), order=0)
    #image = np.clip(image, 0, 1)  # 处理图像范围
    #image = (image * 255).astype(np.uint8)
    slice = Image.fromarray(image)

# 预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485], std=[0.229]),  # Adjust for single-channel image
])


input_tensor = preprocess(slice).unsqueeze(0).to(device)

# Grad-CAM
#target_layers = model.swin_unet.output
target_layers = model.mamba_unet.output

#target_layers = model.mamba_unet.layers[2].blocks[0].self_attention


#target_layers = model.decoder.out_conv
#target_layers = model.classifier
grad_cam = GradCAM(model, target_layers)
output = model(input_tensor)

# 假设是空的ground truth
ground_truth = torch.zeros_like(output)
loss = F.binary_cross_entropy_with_logits(output, ground_truth)
model.zero_grad()
loss.backward()

# 用来保存三个类别的 Grad-CAM 激活图
cam_images = []

# 分别计算每个类别的 Grad-CAM 并保存
for class_idx in range(0, 4):  # 跳过背景类别，计算类别 1, 2, 3
    grad_cam = GradCAM(model, target_layers)
    output = model(input_tensor)

    # 假设每个类别的目标为1，其他为0
    ground_truth = torch.zeros_like(output)
    ground_truth[0, class_idx, :, :] = 1  # 设置当前类别为1

    loss = F.binary_cross_entropy_with_logits(output, ground_truth)
    model.zero_grad()
    loss.backward()

    cam = grad_cam.compute_cam()
    cam = cam.detach().cpu().numpy()
    cam = np.squeeze(cam)  # 去掉多余的维度

    # 反转热力图的颜色
    cam = np.max(cam) - cam
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    cam_images.append(cam)

    # 合成 Grad-CAM 图像
combined_cam = np.maximum(cam_images[1], np.maximum(cam_images[2], cam_images[3]))


    # 将合成的 Grad-CAM 叠加在原始图像上，并同时显示原图、标签图和预测图
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # 原图
#ax.imshow(image, cmap='gray')
#ax.set_title("Original Image")
#ax.axis('off')

#ax[1].imshow(image, cmap='gray')
#ax[1].imshow(cam_images[1], cmap='jet', alpha=0.5)

#ax[2].imshow(image, cmap='gray')
#ax[2].imshow(cam_images[2], cmap='jet', alpha=0.5)

#ax[3].imshow(image, cmap='gray')
#ax[3].imshow(cam_images[3], cmap='jet', alpha=0.5)
    # 合成的 Grad-CAM 图
ax.imshow(image, cmap='gray')
ax.imshow(combined_cam, cmap='jet', alpha=0.5)  # 用半透明方式叠加
#ax[4].imshow(cam_images[2], cmap='jet', alpha=0.2)
#ax[4].imshow(cam_images[3], cmap='jet', alpha=0.2)
#ax[1].set_title("mambaunet")
ax.axis('off')
#label_data=label_data[ :, :]
#label_resized = zoom(label_data, (224 / label_data.shape[0], 224 / label_data.shape[1]), order=0)
#label_resized = np.clip(label_resized, 0, 1)  # Clip the values to [0, 1] for label consistency

# 标签图
#ax.imshow(label_resized, cmap='gray')
#ax.set_title("Ground Truth Label")
#ax.axis('off')


    # 显示图像
plt.tight_layout()
plt.show()

output_dir =  "/home/colin/Mamba-UNet-main/可视化/cam可视化"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

grad_cam_image_path = os.path.join(output_dir, "grad_cam_{}_{}_50%.png").format(tail_field,args.model)
fig.savefig(grad_cam_image_path)
print(f"Grad-CAM activation map saved to {grad_cam_image_path}")