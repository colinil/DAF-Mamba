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

# 假设 Efficient-UNet 的实现已在 efficient_unet 模块中
from config import get_config
from networks.net_factory import net_factory
# 若不再需要 net_factory，则可删除其导入

parser = argparse.ArgumentParser()
# 参数配置（与原代码相同）
parser.add_argument('--root_path', type=str, default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str, default='efficient_unet', help='model_name')
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
parser.add_argument('--output_dir', type=str, default='/home/colin/Mamba-UNet-main/model/cam/efficient_unet', help='Directory to save output images')

args = parser.parse_args()

# 如果输出目录不存在则创建
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# 如果需要配置文件，也可调用 get_config(args)
config = get_config(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = VIM_seg(config, img_size=args.patch_size, num_classes=args.num_classes).to(device)
# model.load_from(config)
model = net_factory(net_type=args.model, in_chns=1, class_num=4).to(device)
model.load_state_dict(torch.load("/home/colin/Mamba-UNet-main/model/MnMs/efficient_unet_150_labeled/efficient_unet/efficient_unet_best_model.pth"))
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
plt.savefig(os.path.join(args.output_dir, 'slice_image.png'), bbox_inches='tight')
plt.show()

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),  # (1, 224, 224)
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
input_img = transform(slice_img).unsqueeze(0).to(device)
print("Input shape:", input_img.shape)  # 预期形状: [1, 1, 224, 224]

# ---------------------------
# GradCAM 类 (B,C,H,W)
# ---------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # 注册前向和后向钩子
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def compute_cam(self):
        gradients = self.gradients  # (B,C,H,W)
        activations = self.activations  # (B,C,H,W)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = (weights * activations).sum(dim=1)  # [B,H,W]
        cam = F.relu(cam)
        B, H, W = cam.shape
        cam_flat = cam.view(B, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        cam_max = cam_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

    def __call__(self, input_img, target_class=None):
        output = self.model(input_img)
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
# 注册获取编码器与解码器部分特征
# ---------------------------
activation = {}

def get_activation(name):
    def hook(module, input, output):
        activation[name] = output.detach()
    return hook

# 对于 Efficient-UNet 的编码器：
# 选择 _blocks[2] 作为浅层特征， _blocks[-1] 作为深层特征
model.encoder._blocks[2].register_forward_hook(get_activation('encoder_shallow'))
model.encoder._blocks[-1].register_forward_hook(get_activation('encoder_deep'))

# 运行一次 forward 得到编码器特征
output = model(input_img)
encoder_shallow_features = activation['encoder_shallow']  # (B,C,H,W)
encoder_deep_features = activation['encoder_deep']        # (B,C,H,W)

# 可视化前64个通道的编码器浅层特征
plt.figure(figsize=(12, 12))
num_channels = encoder_shallow_features.shape[1]
for i in range(min(64, num_channels)):
    plt.subplot(8, 8, i + 1)
    plt.imshow(encoder_shallow_features[0, i, :, :].cpu().numpy(), cmap='gray')
    plt.axis('off')
plt.suptitle("Encoder Shallow Features")
plt.savefig(os.path.join(args.output_dir, 'encoder_shallow_features.png'), bbox_inches='tight')
plt.show()

# 可视化前64个通道的编码器深层特征
plt.figure(figsize=(12, 12))
num_channels = encoder_deep_features.shape[1]
for i in range(min(64, num_channels)):
    plt.subplot(8, 8, i + 1)
    plt.imshow(encoder_deep_features[0, i, :, :].cpu().numpy(), cmap='gray')
    plt.axis('off')
plt.suptitle("Encoder Deep Features")
plt.savefig(os.path.join(args.output_dir, 'encoder_deep_features.png'), bbox_inches='tight')
plt.show()

# -----------------------
# 计算编码器部分 Grad-CAM
# -----------------------
# 使用 _blocks[2] 和 _blocks[-1] 分别计算浅层和深层的 Grad-CAM
gradcam_encoder_shallow = GradCAM(model, target_layer=model.encoder._blocks[2])
encoder_shallow_cam = gradcam_encoder_shallow(input_img, target_class=[1, 2, 3])[0]  # (H,W)

gradcam_encoder_deep = GradCAM(model, target_layer=model.encoder._blocks[-1])
encoder_deep_cam = gradcam_encoder_deep(input_img, target_class=[1, 2, 3])[0]  # (H,W)

# -----------------------
# 计算编码器平均激活图并归一化
# -----------------------
def normalize_to_255(img):
    mn, mx = img.min(), img.max()
    return np.uint8(255 * (img - mn) / (mx - mn + 1e-8))

encoder_shallow_avg = encoder_shallow_features[0].mean(dim=0).cpu().numpy()
encoder_deep_avg = encoder_deep_features[0].mean(dim=0).cpu().numpy()

encoder_shallow_bg = normalize_to_255(encoder_shallow_avg)
encoder_deep_bg = normalize_to_255(encoder_deep_avg)

encoder_shallow_cam_uint8 = normalize_to_255(encoder_shallow_cam)
encoder_deep_cam_uint8 = normalize_to_255(encoder_deep_cam)

# 可选颜色反转
encoder_shallow_cam_uint8_inv = 255 - encoder_shallow_cam_uint8
encoder_shallow_cam_color = cv2.applyColorMap(encoder_shallow_cam_uint8_inv, cv2.COLORMAP_JET)

encoder_deep_cam_uint8_inv = 255 - encoder_deep_cam_uint8
encoder_deep_cam_color = cv2.applyColorMap(encoder_deep_cam_uint8_inv, cv2.COLORMAP_JET)

encoder_shallow_bg_color = cv2.cvtColor(encoder_shallow_bg, cv2.COLOR_GRAY2RGB)
encoder_deep_bg_color = cv2.cvtColor(encoder_deep_bg, cv2.COLOR_GRAY2RGB)

if encoder_shallow_cam_color.shape[:2] != encoder_shallow_bg_color.shape[:2]:
    encoder_shallow_cam_color = cv2.resize(encoder_shallow_cam_color, (encoder_shallow_bg_color.shape[1], encoder_shallow_bg_color.shape[0]))
if encoder_deep_cam_color.shape[:2] != encoder_deep_bg_color.shape[:2]:
    encoder_deep_cam_color = cv2.resize(encoder_deep_cam_color, (encoder_deep_bg_color.shape[1], encoder_deep_bg_color.shape[0]))

alpha = 0.5
encoder_shallow_overlay = cv2.addWeighted(encoder_shallow_bg_color, alpha, encoder_shallow_cam_color, 1 - alpha, 0)
encoder_deep_overlay = cv2.addWeighted(encoder_deep_bg_color, alpha, encoder_deep_cam_color, 1 - alpha, 0)

plt.figure(figsize=(8, 8))
plt.imshow(encoder_shallow_overlay)
plt.axis('off')
plt.title("Encoder Shallow Features with CAM Overlay (Inverted Colors, Classes 1,2,3)")
plt.savefig(os.path.join(args.output_dir, 'encoder_shallow_overlay.png'), bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(encoder_deep_overlay)
plt.axis('off')
plt.title("Encoder Deep Features with CAM Overlay (Inverted Colors, Classes 1,2,3)")
plt.savefig(os.path.join(args.output_dir, 'encoder_deep_overlay.png'), bbox_inches='tight')
plt.show()

# ---------------------------
# 注册解码器部分特征 Hook
# ---------------------------
# 假设解码器为 UnetDecoder，且其 blocks 存储在 model.decoder.blocks 中
model.decoder.blocks[1].register_forward_hook(get_activation('decoder_shallow'))
model.decoder.blocks[3].register_forward_hook(get_activation('decoder_deep'))

# 运行一次完整前向传播以触发解码器 hook
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

# -----------------------
# 解码器部分 Grad-CAM
# -----------------------
gradcam_decoder_shallow = GradCAM(model, target_layer=model.decoder.blocks[1])
decoder_shallow_cam = gradcam_decoder_shallow(input_img, target_class=[1, 2, 3])[0]

gradcam_decoder_deep = GradCAM(model, target_layer=model.decoder.blocks[3])
decoder_deep_cam = gradcam_decoder_deep(input_img, target_class=[1, 2, 3])[0]

# 计算解码器平均激活图并归一化
decoder_shallow_avg = decoder_shallow_features[0].mean(dim=0).cpu().numpy()
decoder_deep_avg = decoder_deep_features[0].mean(dim=0).cpu().numpy()

decoder_shallow_bg = normalize_to_255(decoder_shallow_avg)
decoder_deep_bg = normalize_to_255(decoder_deep_avg)

decoder_shallow_cam_uint8 = normalize_to_255(decoder_shallow_cam)
decoder_deep_cam_uint8 = normalize_to_255(decoder_deep_cam)

decoder_shallow_cam_uint8_inv = 255 - decoder_shallow_cam_uint8
decoder_shallow_cam_color = cv2.applyColorMap(decoder_shallow_cam_uint8_inv, cv2.COLORMAP_JET)

decoder_deep_cam_uint8_inv = 255 - decoder_deep_cam_uint8
decoder_deep_cam_color = cv2.applyColorMap(decoder_deep_cam_uint8_inv, cv2.COLORMAP_JET)

decoder_shallow_bg_color = cv2.cvtColor(decoder_shallow_bg, cv2.COLOR_GRAY2RGB)
decoder_deep_bg_color = cv2.cvtColor(decoder_deep_bg, cv2.COLOR_GRAY2RGB)

if decoder_shallow_cam_color.shape[:2] != decoder_shallow_bg_color.shape[:2]:
    decoder_shallow_cam_color = cv2.resize(decoder_shallow_cam_color,
                                           (decoder_shallow_bg_color.shape[1], decoder_shallow_bg_color.shape[0]))
if decoder_deep_cam_color.shape[:2] != decoder_deep_bg_color.shape[:2]:
    decoder_deep_cam_color = cv2.resize(decoder_deep_cam_color,
                                        (decoder_deep_bg_color.shape[1], decoder_deep_bg_color.shape[0]))

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
