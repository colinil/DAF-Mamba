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

# 1. 引入你自己的 Swin-Unet 相关文件
from networks.vision_transformer import SwinUnet as ViT_seg  # <-- 确保此处导入正确
from config import get_config

parser = argparse.ArgumentParser()
# ------------------ 常规配置参数 ------------------
parser.add_argument('--root_path', type=str, default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str, default='gt', help='model_name')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
parser.add_argument('--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml",
                    help='path to config file')
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs='+')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'], help='cache mode')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level')
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

# ------------------ 输出图像保存目录参数 ------------------
parser.add_argument('--output_dir', type=str, default='/home/colin/Mamba-UNet-main/model/cam/swin_unet',
                    help='Directory to save output images')

args = parser.parse_args()
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

config = get_config(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 初始化并加载你的 SwinUnet
model = ViT_seg(config, img_size=args.patch_size, num_classes=args.num_classes).to(device)
model.load_from(config)
model.load_state_dict(
    torch.load("/home/colin/Mamba-UNet-main/model/MnMs/swinunet_150_labeled/swinunet/swinunet_best_model.pth"))
model.eval()

# 3. 读取并预处理测试图
#h5_file_path = '/home/colin/Mamba-UNet-main/data/ACDC/data/slices/patient011_frame01_slice_3.h5'
h5_file_path = '/home/colin/Mamba-UNet-main/data/MnMs/data/A1E9Q1_frame00_slice_04.h5'
with h5py.File(h5_file_path, 'r') as f:
    image_data = f['image'][...]
    label_data = f['label'][...]
    image = image_data[:, :]
    x, y = image.shape
    # 缩放到 224×224
    image = zoom(image, (224 / x, 224 / y), order=0)
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    slice_img = Image.fromarray(image).convert("RGB")

plt.figure(figsize=(8, 8))
plt.imshow(slice_img)
plt.axis('off')
plt.savefig(os.path.join(args.output_dir, 'slice_image.png'), bbox_inches='tight')
plt.show()

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_img = transform(slice_img).unsqueeze(0).to(device)
print("Input shape:", input_img.shape)  # [1, 3, 224, 224]


# --------------------------- #
#   辅助函数：将 [B, N, C] reshape 为 [B, C, H, W]
# --------------------------- #
def maybe_reshape_tokens_to_map(out):
    """
    如果 out 是 [B, N, C] 且 N 为完全平方数，则将其 reshape 成 [B, C, H, W]，
    否则直接返回原始张量。
    """
    if out.dim() == 3:
        B, N, C = out.shape
        H = int(np.sqrt(N))
        if H * H == N:
            # 先变为 [B, H, H, C]，再 permute 成 [B, C, H, H]
            out = out.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()
    return out


# --------------------------- #
#   4. 在 Hook 中手动 reshape
# --------------------------- #
activation = {}


def get_activation(name):
    """
    钩子函数：如果输出是 [B, N, C] 且 N 为完全平方数，则自动 reshape 成 [B, C, H, W]，
    否则直接保存。
    """

    def hook(module, inp, out):
        if out is None:
            activation[name] = None
            return

        if out.dim() == 4:
            activation[name] = out.detach()
        elif out.dim() == 3:
            activation[name] = maybe_reshape_tokens_to_map(out).detach()
        else:
            activation[name] = out.detach()

    return hook


# --------------------------- #
#   GradCAM 类
# --------------------------- #
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
        if output is None:
            self.activations = None
        else:
            # 如果输出是 [B, N, C] 则转换为 [B, C, H, W]
            self.activations = maybe_reshape_tokens_to_map(output)

    def backward_hook(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if grad is None:
            self.gradients = None
        else:
            # 同样处理梯度
            self.gradients = maybe_reshape_tokens_to_map(grad)

    def compute_cam(self):
        gradients = self.gradients
        activations = self.activations

        if gradients is None or activations is None:
            return None

        # 求每个通道的权重
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = F.relu(cam)

        # 归一化
        B, H, W = cam.shape
        cam_flat = cam.view(B, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        cam_max = cam_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # 上采样到 224x224
        cam = F.interpolate(cam.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False)
        return cam.squeeze(1)

    def __call__(self, input_img, target_class=None):
        output = self.model(input_img)
        # output: [B, num_classes, H', W']
        if target_class is None:
            avg_scores = output.mean(dim=(0, 2, 3))
            target_class = avg_scores.argmax().item()

        if isinstance(target_class, list):
            combined_cam = None
            for i, cls in enumerate(target_class):
                self.model.zero_grad()
                score = output[:, cls, :, :].mean()
                score.backward(retain_graph=(i < len(target_class) - 1))
                single_cam = self.compute_cam()
                if single_cam is None:
                    continue
                if combined_cam is None:
                    combined_cam = single_cam
                else:
                    combined_cam = torch.max(combined_cam, single_cam)
            if combined_cam is not None:
                return combined_cam.detach().cpu().numpy()
            else:
                return None
        else:
            self.model.zero_grad()
            score = output[:, target_class, :, :].mean()
            score.backward()
            cam = self.compute_cam()
            if cam is not None:
                return cam.detach().cpu().numpy()
            else:
                return None


# --------------------------- #
#   5. 注册 Hook: 编码器/解码器浅深层
# --------------------------- #
model.swin_unet.patch_embed.register_forward_hook(get_activation('patch_embed'))  # 可视需要是否保留
model.swin_unet.layers[0].register_forward_hook(get_activation('shallow'))  # stage0 => 56×56（token数为784）
model.swin_unet.layers[-1].register_forward_hook(get_activation('deep'))  # stage3 => 7×7（token数为49）

model.swin_unet.layers_up[0].register_forward_hook(get_activation('decoder_shallow'))
model.swin_unet.layers_up[-1].register_forward_hook(get_activation('decoder_deep'))

# ---------------------------
#   6. 先跑 encoder 部分
# ---------------------------
features, x_downsample = model.swin_unet.forward_features(input_img)

shallow_features = activation['shallow']  # 现在应是 [B, C, H, W]
deep_features = activation['deep']


def get_channel(features, i):
    return features[0, i, :, :]


def visualize_features_64channels(feat_tensor, title, save_name):
    if feat_tensor is None:
        print(f"{title} is None, skip.")
        return
    plt.figure(figsize=(12, 12))
    C = feat_tensor.shape[1]
    for i in range(min(64, C)):
        plt.subplot(8, 8, i + 1)
        ch_data = get_channel(feat_tensor, i).cpu().numpy()
        plt.imshow(ch_data, cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.savefig(os.path.join(args.output_dir, save_name), bbox_inches='tight')
    plt.show()


visualize_features_64channels(shallow_features, "Shallow Features (Encoder)", 'shallow_features.png')
visualize_features_64channels(deep_features, "Deep Features (Encoder)", 'deep_features.png')

# ---------------------------
#   7. Encoder Grad-CAM
# ---------------------------
gradcam_shallow = GradCAM(model.swin_unet, target_layer=model.swin_unet.layers[0])
shallow_cam = gradcam_shallow(input_img, target_class=[1, 2, 3])
if shallow_cam is not None:
    shallow_cam = shallow_cam[0]  # 取 batch=0
else:
    shallow_cam = np.zeros((224, 224), dtype=np.float32)

gradcam_deep = GradCAM(model.swin_unet, target_layer=model.swin_unet.layers[-1])
deep_cam = gradcam_deep(input_img, target_class=[1, 2, 3])
if deep_cam is not None:
    deep_cam = deep_cam[0]
else:
    deep_cam = np.zeros((224, 224), dtype=np.float32)


def normalize_to_255(img):
    mn, mx = img.min(), img.max()
    return np.uint8(255 * (img - mn) / (mx - mn + 1e-8))


# 叠加显示 Encoder 部分
if shallow_features is not None:
    shallow_avg = shallow_features[0].mean(dim=0).cpu().numpy()
    shallow_bg = normalize_to_255(shallow_avg)
    shallow_bg = cv2.resize(shallow_bg, (224, 224), interpolation=cv2.INTER_LINEAR)
    shallow_bg_color = cv2.cvtColor(shallow_bg, cv2.COLOR_GRAY2RGB)

    shallow_cam_uint8 = normalize_to_255(shallow_cam)
    shallow_cam_inv = 255 - shallow_cam_uint8
    shallow_cam_color = cv2.applyColorMap(shallow_cam_inv, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(shallow_bg_color, 0.5, shallow_cam_color, 0.5, 0)
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title("Encoder Shallow Features with CAM Overlay")
    plt.savefig(os.path.join(args.output_dir, 'shallow_overlay.png'), bbox_inches='tight')
    plt.show()

if deep_features is not None:
    deep_avg = deep_features[0].mean(dim=0).cpu().numpy()
    deep_bg = normalize_to_255(deep_avg)
    deep_cam_uint8 = normalize_to_255(deep_cam)
    deep_cam_inv = 255 - deep_cam_uint8
    deep_cam_color = cv2.applyColorMap(deep_cam_inv, cv2.COLORMAP_JET)
    deep_bg_color = cv2.cvtColor(deep_bg, cv2.COLOR_GRAY2RGB)
    if deep_cam_color.shape[:2] != deep_bg_color.shape[:2]:
        deep_cam_color = cv2.resize(deep_cam_color, (deep_bg_color.shape[1], deep_bg_color.shape[0]))
    overlay = cv2.addWeighted(deep_bg_color, 0.5, deep_cam_color, 0.5, 0)
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title("Encoder Deep Features with CAM Overlay")
    plt.savefig(os.path.join(args.output_dir, 'deep_overlay.png'), bbox_inches='tight')
    plt.show()

# ---------------------------
#   8. Decoder 部分
# ---------------------------
output = model(input_img)  # 这一步会触发 layers_up 的 hook

decoder_shallow_features = activation['decoder_shallow']
decoder_deep_features = activation['decoder_deep']

visualize_features_64channels(decoder_shallow_features, "Decoder Shallow Features", 'decoder_shallow_features.png')
visualize_features_64channels(decoder_deep_features, "Decoder Deep Features", 'decoder_deep_features.png')

# GradCAM on Decoder
gradcam_decoder_shallow = GradCAM(model.swin_unet, target_layer=model.swin_unet.layers_up[0])
dshallow_cam = gradcam_decoder_shallow(input_img, target_class=[1, 2, 3])
if dshallow_cam is not None:
    dshallow_cam = dshallow_cam[0]
else:
    dshallow_cam = np.zeros((224, 224), dtype=np.float32)

gradcam_decoder_deep = GradCAM(model.swin_unet, target_layer=model.swin_unet.layers_up[-1])
ddeep_cam = gradcam_decoder_deep(input_img, target_class=[1, 2, 3])
if ddeep_cam is not None:
    ddeep_cam = ddeep_cam[0]
else:
    ddeep_cam = np.zeros((224, 224), dtype=np.float32)

if decoder_shallow_features is not None:
    dshallow_avg = decoder_shallow_features[0].mean(dim=0).cpu().numpy()
    dshallow_bg = normalize_to_255(dshallow_avg)
    dshallow_cam_uint8 = normalize_to_255(dshallow_cam)
    dshallow_cam_inv = 255 - dshallow_cam_uint8
    dshallow_cam_color = cv2.applyColorMap(dshallow_cam_inv, cv2.COLORMAP_JET)
    dshallow_bg_color = cv2.cvtColor(dshallow_bg, cv2.COLOR_GRAY2RGB)
    if dshallow_cam_color.shape[:2] != dshallow_bg_color.shape[:2]:
        dshallow_cam_color = cv2.resize(dshallow_cam_color, (dshallow_bg_color.shape[1], dshallow_bg_color.shape[0]))
    overlay = cv2.addWeighted(dshallow_bg_color, 0.5, dshallow_cam_color, 0.5, 0)
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title("Decoder Shallow Features with CAM Overlay")
    plt.savefig(os.path.join(args.output_dir, 'decoder_shallow_overlay.png'), bbox_inches='tight')
    plt.show()

if decoder_deep_features is not None:
    ddeep_avg = decoder_deep_features[0].mean(dim=0).cpu().numpy()
    ddeep_bg = normalize_to_255(ddeep_avg)
    ddeep_cam_uint8 = normalize_to_255(ddeep_cam)
    ddeep_cam_inv = 255 - ddeep_cam_uint8
    ddeep_cam_color = cv2.applyColorMap(ddeep_cam_inv, cv2.COLORMAP_JET)
    ddeep_bg_color = cv2.cvtColor(ddeep_bg, cv2.COLOR_GRAY2RGB)
    if ddeep_cam_color.shape[:2] != ddeep_bg_color.shape[:2]:
        ddeep_cam_color = cv2.resize(ddeep_cam_color, (ddeep_bg_color.shape[1], ddeep_bg_color.shape[0]))
    overlay = cv2.addWeighted(ddeep_bg_color, 0.5, ddeep_cam_color, 0.5, 0)
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title("Decoder Deep Features with CAM Overlay")
    plt.savefig(os.path.join(args.output_dir, 'decoder_deep_overlay.png'), bbox_inches='tight')
    plt.show()

print("Done visualization.")
