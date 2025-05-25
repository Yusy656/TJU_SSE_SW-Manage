import os
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from skimage.measure import shannon_entropy

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'best_model_epoch42.pth')

# -------------------- 数据预处理与增强 --------------------
class RealisticFireSmokeAugmentation:
    """改进的物理增强模块"""

    def __init__(self, fire_prob=0.3, smoke_prob=0.4):
        self.fire_prob = fire_prob
        self.smoke_prob = smoke_prob

    def __call__(self, img):
        img = img.astype(np.float32)
        h, w = img.shape  # 在函数开头统一获取图像尺寸
        # 烟雾粒子扩散模拟
        if np.random.rand() < self.smoke_prob:
            h, w = img.shape
            smoke = np.zeros((h, w), dtype=np.float32)
            for _ in range(np.random.randint(3, 8)):
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                radius = np.random.randint(20, 100)
                intensity = np.random.uniform(0.3, 0.8)
                smoke = cv2.circle(smoke, (x, y), radius, intensity, -1)
            smoke = cv2.GaussianBlur(smoke, (151, 151), 0)
            img = cv2.addWeighted(img, 0.7, smoke, 0.3, 0)

        # 不规则火源生成
        if np.random.rand() < self.fire_prob:
            pts = np.random.rand(5, 2) * np.array([[w * 0.3, h * 0.3]]) + np.array([[w * 0.3, h * 0.3]])
            pts = pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], 255)

        return np.clip(img, 0, 255).astype(np.uint8)


def physics_based_enhancement(img):
    """兼容性增强处理"""
    # CLAHE直方图均衡
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    # 替代导向滤波
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


# -------------------- 数据集类 --------------------
class FireSmokeDataset(Dataset):
    def __init__(self, thermal_dir, ir_dir, transform=None):
        self.thermal_dir = thermal_dir
        self.ir_dir = ir_dir
        self.thermal_files = sorted([f for f in os.listdir(thermal_dir) if f.endswith(('.png', '.jpg'))])
        self.ir_files = sorted([f for f in os.listdir(ir_dir) if f.endswith(('.png', '.jpg'))])
        assert len(self.thermal_files) == len(self.ir_files), "数据不匹配"
        self.transform = transform
        self.augment = RealisticFireSmokeAugmentation()

    def __len__(self):
        return len(self.thermal_files)

    def __getitem__(self, idx):
        try:
            # 读取热成像
            thermal_path = os.path.join(self.thermal_dir, self.thermal_files[idx])
            thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
            if thermal is None:
                raise FileNotFoundError(f"无法读取热成像文件：{thermal_path}")
            thermal = physics_based_enhancement(thermal)

            # 读取红外
            ir_path = os.path.join(self.ir_dir, self.ir_files[idx])
            ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
            if ir is None:
                raise FileNotFoundError(f"无法读取红外文件：{ir_path}")
            ir = physics_based_enhancement(ir)

            # 数据增强
            thermal = self.augment(thermal)
            ir = self.augment(ir)

            # 转换为Tensor
            if self.transform:
                thermal = self.transform(thermal)
                ir = self.transform(ir)
            else:
                thermal = torch.from_numpy(thermal).float().unsqueeze(0) / 255.0
                ir = torch.from_numpy(ir).float().unsqueeze(0) / 255.0

            return thermal, ir

        except Exception as e:
            print(f"处理数据{idx}出错: {str(e)}")
            return self[np.random.randint(len(self))]  # 返回随机样本替代


# -------------------- 改进网络架构 --------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.GELU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.conv(x)


class EnhancedEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            ResidualBlock(64)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            ResidualBlock(128)
        )
        self.attn = ChannelAttention(128)

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        return self.attn(x)


class EnhancedFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.thermal_enc = EnhancedEncoder()
        self.ir_enc = EnhancedEncoder()

        self.fusion_attn = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 2, 1),
            nn.Softmax(dim=1)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            ResidualBlock(32),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, thermal, ir):
        t_feat = self.thermal_enc(thermal)
        i_feat = self.ir_enc(ir)

        attn = self.fusion_attn(torch.cat([t_feat, i_feat], dim=1))
        fused = attn[:, 0:1] * t_feat + attn[:, 1:2] * i_feat

        return self.decoder(fused)


# -------------------- 损失函数 --------------------
class SSIM(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window = self._gaussian_window(window_size, sigma)
        self.window_size = window_size

    def _gaussian_window(self, size, sigma):
        gauss = torch.Tensor([np.exp(-(x - size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(size)])
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        return window.unsqueeze(0).unsqueeze(0)

    def forward(self, img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        window = self.window.to(img1.device)
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=1)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=1)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=1) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=1) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


class AdvancedFusionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIM()
        self.register_buffer('sobel_x',
                             torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y',
                             torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

    def gradient_loss(self, pred, target):
        grad_pred_x = F.conv2d(pred, self.sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, self.sobel_y, padding=1)
        grad_target_x = F.conv2d(target, self.sobel_x, padding=1)
        grad_target_y = F.conv2d(target, self.sobel_y, padding=1)
        return F.l1_loss(grad_pred_x, grad_target_x) + F.l1_loss(grad_pred_y, grad_target_y)

    def forward(self, fused, thermal, ir):
        # 动态目标生成
        thermal_mask = (thermal > 0.7).float()
        target = thermal_mask * thermal + (1 - thermal_mask) * ir

        # 多目标损失
        l_content = self.l1(fused, target)
        l_ssim = 1 - self.ssim(fused, target)
        l_grad = self.gradient_loss(fused, ir)

        return 0.5 * l_content + 0.3 * l_ssim + 0.2 * l_grad


# -------------------- 训练流程 --------------------
def train():
    # 数据集配置
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    dataset = FireSmokeDataset(
        thermal_dir="/root/autodl-tmp/Real-Time-Image-Dehazing-Using-Deep-Learning-PyTorch--main/clear",
        ir_dir="/root/autodl-tmp/Real-Time-Image-Dehazing-Using-Deep-Learning-PyTorch--main/hazy",
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )

    # 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedFusionModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    criterion = AdvancedFusionLoss().to(device)
    scaler = GradScaler()

    # 训练监控
    best_loss = float('inf')
    loss_history = []

    for epoch in range(50):
        model.train()
        epoch_loss = 0.0

        with tqdm(dataloader, desc=f'Epoch {epoch + 1}/50', unit='batch') as pbar:
            for thermal, ir in pbar:
                thermal = thermal.to(device, non_blocking=True)
                ir = ir.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast():
                    fused = model(thermal, ir)
                    loss = criterion(fused, thermal, ir)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                avg_loss = epoch_loss / len(pbar)
                pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})

            scheduler.step()

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), f'best_model_epoch{epoch + 1}.pth')

            # 记录损失曲线
            loss_history.append(avg_loss)
            if (epoch + 1) % 5 == 0:
                plt.figure(figsize=(10, 6))
                plt.plot(loss_history, 'b-o', linewidth=2)
                plt.title('Training Loss Curve', fontsize=14)
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.grid(True)
                plt.savefig(f'loss_curve_epoch_{epoch + 1}.png', dpi=300, bbox_inches='tight')
                plt.close()


def inference_single_image(thermal_path, ir_path, output_path, model_path= MODEL_PATH, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """单张图像推理函数"""
    try:
        # 初始化模型
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        model = EnhancedFusionModel().to(device)
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # 读取图像
        thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

        if thermal is None or ir_img is None:
            raise ValueError("无法读取输入图像，请检查路径是否正确")

        # 物理增强处理
        t_proc = physics_based_enhancement(thermal)
        i_proc = physics_based_enhancement(ir_img)

        # 转换为Tensor
        t_tensor = torch.from_numpy(t_proc).float().unsqueeze(0).unsqueeze(0) / 255.0
        i_tensor = torch.from_numpy(i_proc).float().unsqueeze(0).unsqueeze(0) / 255.0

        # 推理
        with torch.no_grad():
            fused = model(t_tensor.to(device), i_tensor.to(device))
        
        # 后处理
        fused_img = (fused.squeeze().cpu().numpy() * 255).astype(np.uint8)
        
        # 确保输出图像尺寸与输入一致
        h, w = t_proc.shape
        fused_img = cv2.resize(fused_img, (w, h), interpolation=cv2.INTER_LINEAR)

        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, fused_img)
        
        # 计算评估指标
        ent = shannon_entropy(fused_img)
        sf = spatial_frequency(fused_img)
        std_f = fused_img.std()
        mi_t = compute_mi(fused_img, t_proc)
        mi_i = compute_mi(fused_img, i_proc)

        metrics = {
            'entropy': float(ent),
            'spatial_frequency': float(sf),
            'std_deviation': float(std_f),
            'mi_thermal': float(mi_t),
            'mi_ir': float(mi_i)
        }

        print(f"融合结果已保存至：{output_path}")
        print("\n=== 评估指标 ===")
        print(f"熵值:              {ent:.2f}")
        print(f"空间频率:          {sf:.2f}")
        print(f"标准差:            {std_f:.2f}")
        print(f"互信息(融合↔热成像): {mi_t:.2f}")
        print(f"互信息(融合↔红外):   {mi_i:.2f}")

        return metrics

    except Exception as e:
        print(f"处理失败: {str(e)}")
        return None

def spatial_frequency(img):
    img = img.astype(np.float32)
    rf = np.diff(img, axis=1)
    cf = np.diff(img, axis=0)
    rf_mean = np.sqrt((rf ** 2).mean())
    cf_mean = np.sqrt((cf ** 2).mean())
    return math.sqrt(rf_mean ** 2 + cf_mean ** 2)


def mutual_information(hgram):
    pxy = hgram / float(np.sum(hgram))
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def compute_mi(img1, img2, bins=256):
    hgram, _, _ = np.histogram2d(
        img1.ravel(), img2.ravel(),
        bins=bins
    )
    return mutual_information(hgram)


def inference_batch(
        thermal_dir: str,
        ir_dir: str,
        output_dir: str,
        model_path: str = "./best_model_epoch42.pth",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    批量推理，输出并汇总以下指标：
      - 无参考：Entropy, Spatial Frequency, Std
      - 源–融相关性：MI(fused, thermal), MI(fused, ir)
    """
    # 模型加载（指定 weights_only=True）
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = EnhancedFusionModel().to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    thermal_files = sorted([f for f in os.listdir(thermal_dir) if f.lower().endswith(('.png', '.jpg'))])
    ir_files = sorted([f for f in os.listdir(ir_dir) if f.lower().endswith(('.png', '.jpg'))])
    os.makedirs(output_dir, exist_ok=True)

    ent_list, sf_list, std_list = [], [], []
    mi_t_list, mi_i_list = [], []

    for t_name, i_name in tqdm(zip(thermal_files, ir_files),
                               total=min(len(thermal_files), len(ir_files)),
                               desc='Batch Inference'):
        # 读取
        t_path, i_path = os.path.join(thermal_dir, t_name), os.path.join(ir_dir, i_name)
        thermal = cv2.imread(t_path, cv2.IMREAD_GRAYSCALE)
        ir_img = cv2.imread(i_path, cv2.IMREAD_GRAYSCALE)
        if thermal is None or ir_img is None:
            print(f"跳过损坏文件：{t_name} 或 {i_name}")
            continue

        # 物理增强
        t_proc = physics_based_enhancement(thermal)
        i_proc = physics_based_enhancement(ir_img)

        # Tensor 化
        t_tensor = torch.from_numpy(t_proc).float().unsqueeze(0).unsqueeze(0) / 255.0
        i_tensor = torch.from_numpy(i_proc).float().unsqueeze(0).unsqueeze(0) / 255.0

        # 推理
        with torch.no_grad():
            fused = model(t_tensor.to(device), i_tensor.to(device))
        fused_img = (fused.squeeze().cpu().numpy() * 255).astype(np.uint8)

        # —— 关键改动：对齐尺寸 ——
        # 有时 conv/convtranspose 会改变维度，resize 回原图大小
        h, w = t_proc.shape
        fused_img = cv2.resize(fused_img, (w, h), interpolation=cv2.INTER_LINEAR)

        # 保存
        out_name = f"fused_{t_name}"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, fused_img)

        # 计算指标
        ent = shannon_entropy(fused_img)
        sf = spatial_frequency(fused_img)
        std_f = fused_img.std()
        mi_t = compute_mi(fused_img, t_proc)
        mi_i = compute_mi(fused_img, i_proc)

        ent_list.append(ent)
        sf_list.append(sf)
        std_list.append(std_f)
        mi_t_list.append(mi_t)
        mi_i_list.append(mi_i)

        print(f"[{t_name}] Ent={ent:.2f}, SF={sf:.2f}, Std={std_f:.2f}, MI_T={mi_t:.2f}, MI_I={mi_i:.2f}")

    if ent_list:
        print("\n=== Average Metrics ===")
        print(f"Entropy:            {np.mean(ent_list):.2f}")
        print(f"Spatial Frequency:  {np.mean(sf_list):.2f}")
        print(f"Std Deviation:      {np.mean(std_list):.2f}")
        print(f"MI (fused↔thermal): {np.mean(mi_t_list):.2f}")
        print(f"MI (fused↔ir):      {np.mean(mi_i_list):.2f}")


if __name__ == '__main__':
    # train()
    # 使用示例
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # 单张图像推理
    # inference_single_image(
    #     thermal_path="frame_00000650.jpg",
    #     ir_path="frame_00000650 (1).jpg",
    #     model_path="best_model_epoch39.pth",
    #     output_path="new_fused_result.jpg",
    #     device=device
    # )

    # 批量推理
    inference_batch(
        thermal_dir="/root/autodl-tmp/pipeline/frames/processed_ir_smoked3",
        ir_dir="/root/autodl-tmp/pipeline/frames/processed_rgb_smoked3",
        model_path="../best_model_epoch34.pth",
        output_dir="/root/autodl-tmp/pipeline/fusion_results_5",
        device=device
    )
