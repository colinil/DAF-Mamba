import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import h5py
import argparse
from scipy.ndimage import zoom

from networks.vision_mamba import MambaUnet as VIM_seg
from config import get_config
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
# 参数配置（此处省略，和原代码相同）
parser.add_argument('--root_path', type=str, default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
parser.add_argument('--cfg', type=str, default="../code/configs/vmamba_tiny.yaml", help='path to config file')
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

# 新增：输出图像保存目录参数
parser.add_argument('--output_dir', type=str, default='/home/colin/Mamba-UNet-main/model/cam/unet', help='Directory to save output images')

args = parser.parse_args()

# 如果输出目录不存在则创建
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

config = get_config(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = VIM_seg(config, img_size=args.patch_size, num_classes=args.num_classes).to(device)
# model.load_from(config)
model = net_factory(net_type=args.model, in_chns=1, class_num=4).to(device)
model.load_state_dict(torch.load("/home/colin/Mamba-UNet-main/model/MnMs/unet_150_labeled/unet/unet_best_model.pth"))
model.eval()

# 读取测试图
#h5_file_path = '/home/colin/Mamba-UNet-main/data/ACDC/data/slices/patient011_frame01_slice_3.h5'
h5_file_path = '/home/colin/Mamba-UNet-main/data/MnMs/data/A1E9Q1_frame00_slice_04.h5'
with h5py.File(h5_file_path, 'r') as f:
    image_data = f['image'][...]
    label_data = f['label'][...]
    image = image_data[:, :]
    x, y = image.shape[0], image.shape[1]
    image = zoom(image, (224 / x, 224 / y), order=0)
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    slice_img = Image.fromarray(image).convert("L")

plt.figure(figsize=(8, 8))
plt.imshow(slice_img, cmap='gray')
plt.axis('off')
# 保存原始输入图
plt.savefig(os.path.join(args.output_dir, 'slice_image.png'), bbox_inches='tight')
plt.show()

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),  # (1, 224, 224)
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
input_img = transform(slice_img).unsqueeze(0).to(device)
print("Input shape:", input_img.shape)  # [1, 1, 224, 224] 实际会是这样

# ---------------------------
# GradCAM 类 (B,C,H,W)
# ---------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # 注册前向/后向钩子
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        """
        修改点：去掉错误的 (B,H,W,C) -> (B,C,H,W) 转置判断，
        仅保留原始输出即可。
        """
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        """
        同理，去掉错误的转置判断。
        """
        self.gradients = grad_output[0]

    def compute_cam(self):
        # activations, gradients 均是 (B,C,H,W)
        gradients = self.gradients
        activations = self.activations
        # 通道维度做平均
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        # 加权
        cam = (weights * activations).sum(dim=1)  # [B,H,W]
        cam = F.relu(cam)
        # 归一化到 [0,1]
        B, H, W = cam.shape
        cam_flat = cam.view(B, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        cam_max = cam_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

    def __call__(self, input_img, target_class=None):
        output = self.model(input_img)
        # output shape: [B, num_classes, H', W']
        if target_class is None:
            avg_scores = output.mean(dim=(0, 2, 3))
            target_class = avg_scores.argmax().item()

        if isinstance(target_class, list):
            combined_cam = None
            for i, cls in enumerate(target_class):
                self.model.zero_grad()
                score = output[:, cls, :, :].mean()
                score.backward(retain_graph=(i < len(target_class) - 1))
                cam = self.compute_cam()
                if combined_cam is None:
                    combined_cam = cam
                else:
                    combined_cam = torch.max(combined_cam, cam)
            return combined_cam.detach().cpu().numpy()
        else:
            self.model.zero_grad()
            score = output[:, target_class, :, :].mean()
            score.backward()
            cam = self.compute_cam()
            return cam.detach().cpu().numpy()

# ---------------------------
# 注册获取浅层、深层特征
# ---------------------------
activation = {}

def get_activation(name):
    """
    同样去掉 (B,H,W,C) -> (B,C,H,W) 的强制转置。
    """
    def hook(module, input, output):
        activation[name] = output.detach()
    return hook

# 替换为你实际想 hook 的 encoder 部分
model.encoder.down2.register_forward_hook(get_activation('shallow'))
model.encoder.down4.register_forward_hook(get_activation('deep'))

# 运行一次 forward
output = model(input_img)

# 取浅层、深层特征: (B,C,H,W)
shallow_features = activation['shallow']  # (1, C, H, W)
deep_features = activation['deep']        # (1, C, H, W)

# 可视化前64个通道的浅层特征
plt.figure(figsize=(12, 12))
num_channels = shallow_features.shape[1]
for i in range(min(64, num_channels)):
    plt.subplot(8, 8, i + 1)
    plt.imshow(shallow_features[0, i, :, :].cpu().numpy(), cmap='gray')
    plt.axis('off')
plt.suptitle("Shallow Features")
plt.savefig(os.path.join(args.output_dir, 'shallow_features.png'), bbox_inches='tight')
plt.show()

# 可视化前64个通道的深层特征
plt.figure(figsize=(12, 12))
num_channels = deep_features.shape[1]
for i in range(min(64, num_channels)):
    plt.subplot(8, 8, i + 1)
    plt.imshow(deep_features[0, i, :, :].cpu().numpy(), cmap='gray')
    plt.axis('off')
plt.suptitle("Deep Features")
plt.savefig(os.path.join(args.output_dir, 'deep_features.png'), bbox_inches='tight')
plt.show()

# -----------------------
# 计算浅层、深层 Grad-CAM
# -----------------------
gradcam_shallow = GradCAM(model, target_layer=model.encoder.down2)
shallow_cam = gradcam_shallow(input_img, target_class=[1, 2, 3])  # (1,H,W)
shallow_cam = shallow_cam[0]  # -> (H,W)

gradcam_deep = GradCAM(model, target_layer=model.encoder.down4)
deep_cam = gradcam_deep(input_img, target_class=[1, 2, 3])  # (1,H,W)
deep_cam = deep_cam[0]  # -> (H,W)

# -----------------------
# 计算浅层、深层平均激活图
# -----------------------
shallow_avg = shallow_features[0].mean(axis=0).cpu().numpy()  # (H,W)
deep_avg = deep_features[0].mean(axis=0).cpu().numpy()

def normalize_to_255(img):
    mn, mx = img.min(), img.max()
    return np.uint8(255 * (img - mn) / (mx - mn + 1e-8))

shallow_bg = normalize_to_255(shallow_avg)
deep_bg = normalize_to_255(deep_avg)

shallow_cam_uint8 = normalize_to_255(shallow_cam)
deep_cam_uint8 = normalize_to_255(deep_cam)

# -- 关键改动：可选的颜色反转
shallow_cam_uint8_inv = 255 - shallow_cam_uint8
shallow_cam_color = cv2.applyColorMap(shallow_cam_uint8_inv, cv2.COLORMAP_JET)

deep_cam_uint8_inv = 255 - deep_cam_uint8
deep_cam_color = cv2.applyColorMap(deep_cam_uint8_inv, cv2.COLORMAP_JET)

shallow_bg_color = cv2.cvtColor(shallow_bg, cv2.COLOR_GRAY2RGB)
deep_bg_color = cv2.cvtColor(deep_bg, cv2.COLOR_GRAY2RGB)

if shallow_cam_color.shape[:2] != shallow_bg_color.shape[:2]:
    shallow_cam_color = cv2.resize(shallow_cam_color, (shallow_bg_color.shape[1], shallow_bg_color.shape[0]))
if deep_cam_color.shape[:2] != deep_bg_color.shape[:2]:
    deep_cam_color = cv2.resize(deep_cam_color, (deep_bg_color.shape[1], deep_bg_color.shape[0]))

alpha = 0.5
shallow_overlay = cv2.addWeighted(shallow_bg_color, alpha, shallow_cam_color, 1 - alpha, 0)
deep_overlay = cv2.addWeighted(deep_bg_color, alpha, deep_cam_color, 1 - alpha, 0)

plt.figure(figsize=(8, 8))
plt.imshow(shallow_overlay)
plt.axis('off')
plt.title("Shallow Features with CAM Overlay (Inverted Colors, Classes 1,2,3)")
plt.savefig(os.path.join(args.output_dir, 'shallow_overlay.png'), bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(deep_overlay)
plt.axis('off')
plt.title("Deep Features with CAM Overlay (Inverted Colors, Classes 1,2,3)")
plt.savefig(os.path.join(args.output_dir, 'deep_overlay.png'), bbox_inches='tight')
plt.show()

# ---------------------------
# 解码器部分特征可视化
# ---------------------------
model.decoder.up1.register_forward_hook(get_activation('decoder_shallow'))
model.decoder.up3.register_forward_hook(get_activation('decoder_deep'))

# 运行完整的前向传播，这样解码器部分也会执行，从而触发 hook
output = model(input_img)

decoder_shallow_features = activation.get('decoder_shallow')
decoder_deep_features = activation.get('decoder_deep')

if decoder_shallow_features is None or decoder_deep_features is None:
    raise RuntimeError("请检查解码器部分的 hook 是否正确注册，确保模型前向传播经过了相应的模块。")

# 可视化解码器浅层特征（显示前 64 个通道）
plt.figure(figsize=(12, 12))
num_channels = decoder_shallow_features.shape[1]
for i in range(min(64, num_channels)):
    plt.subplot(8, 8, i + 1)
    plt.imshow(decoder_shallow_features[0, i, :, :].cpu().numpy(), cmap='gray')
    plt.axis('off')
plt.suptitle("Decoder Shallow Features")
plt.savefig(os.path.join(args.output_dir, 'decoder_shallow_features.png'), bbox_inches='tight')
plt.show()

# 可视化解码器深层特征（显示前 64 个通道）
plt.figure(figsize=(12, 12))
num_channels = decoder_deep_features.shape[1]
for i in range(min(64, num_channels)):
    plt.subplot(8, 8, i + 1)
    plt.imshow(decoder_deep_features[0, i, :, :].cpu().numpy(), cmap='gray')
    plt.axis('off')
plt.suptitle("Decoder Deep Features")
plt.savefig(os.path.join(args.output_dir, 'decoder_deep_features.png'), bbox_inches='tight')
plt.show()

# 解码器浅层 CAM
gradcam_decoder_shallow = GradCAM(model, target_layer=model.decoder.up1)
decoder_shallow_cam = gradcam_decoder_shallow(input_img, target_class=[1, 2, 3])  # (1,H,W)
decoder_shallow_cam = decoder_shallow_cam[0]  # (H,W)

# 解码器深层 CAM
gradcam_decoder_deep = GradCAM(model, target_layer=model.decoder.up3)
decoder_deep_cam = gradcam_decoder_deep(input_img, target_class=[1, 2, 3])  # (1,H,W)
decoder_deep_cam = decoder_deep_cam[0]

# 对解码器浅层背景激活图（平均每个通道）
decoder_shallow_avg = decoder_shallow_features[0].mean(dim=0).cpu().numpy()  # (H,W)
decoder_deep_avg = decoder_deep_features[0].mean(dim=0).cpu().numpy()       # (H,W)

decoder_shallow_bg = normalize_to_255(decoder_shallow_avg)
decoder_deep_bg = normalize_to_255(decoder_deep_avg)

decoder_shallow_cam_uint8 = normalize_to_255(decoder_shallow_cam)
decoder_deep_cam_uint8 = normalize_to_255(decoder_deep_cam)

# 生成伪彩色图（可选择是否反转颜色）
decoder_shallow_cam_uint8_inv = 255 - decoder_shallow_cam_uint8
decoder_shallow_cam_color = cv2.applyColorMap(decoder_shallow_cam_uint8_inv, cv2.COLORMAP_JET)

decoder_deep_cam_uint8_inv = 255 - decoder_deep_cam_uint8
decoder_deep_cam_color = cv2.applyColorMap(decoder_deep_cam_uint8_inv, cv2.COLORMAP_JET)

decoder_shallow_bg_color = cv2.cvtColor(decoder_shallow_bg, cv2.COLOR_GRAY2RGB)
decoder_deep_bg_color = cv2.cvtColor(decoder_deep_bg, cv2.COLOR_GRAY2RGB)

# 尺寸对齐
if decoder_shallow_cam_color.shape[:2] != decoder_shallow_bg_color.shape[:2]:
    decoder_shallow_cam_color = cv2.resize(decoder_shallow_cam_color,
                                           (decoder_shallow_bg_color.shape[1], decoder_shallow_bg_color.shape[0]))
if decoder_deep_cam_color.shape[:2] != decoder_deep_bg_color.shape[:2]:
    decoder_deep_cam_color = cv2.resize(decoder_deep_cam_color,
                                        (decoder_deep_bg_color.shape[1], decoder_deep_bg_color.shape[0]))

alpha = 0.5
decoder_shallow_overlay = cv2.addWeighted(decoder_shallow_bg_color, alpha,
                                          decoder_shallow_cam_color, 1 - alpha, 0)
decoder_deep_overlay = cv2.addWeighted(decoder_deep_bg_color, alpha,
                                       decoder_deep_cam_color, 1 - alpha, 0)

plt.figure(figsize=(8, 8))
plt.imshow(decoder_shallow_overlay)
plt.axis('off')
plt.title("Decoder Shallow Features with CAM Overlay (Classes 1,2,3)")
plt.savefig(os.path.join(args.output_dir, 'decoder_shallow_overlay.png'), bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(decoder_deep_overlay)
plt.axis('off')
plt.title("Decoder Deep Features with CAM Overlay (Classes 1,2,3)")
plt.savefig(os.path.join(args.output_dir, 'decoder_deep_overlay.png'), bbox_inches='tight')
plt.show()
