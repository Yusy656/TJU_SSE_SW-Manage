import os
import pytest
import numpy as np
import torch
import psutil
import cv2
import tempfile
import shutil
from pathlib import Path
from torchvision import transforms
from fusion import (
    EnhancedFusionModel, FireSmokeDataset, spatial_frequency, compute_mi,
    RealisticFireSmokeAugmentation, physics_based_enhancement,
    ChannelAttention, ResidualBlock, EnhancedEncoder,
    SSIM, AdvancedFusionLoss, inference_single_image, inference_batch
)

# 创建临时测试数据
@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        thermal_dir = Path(tmpdir) / "thermal"
        ir_dir = Path(tmpdir) / "ir"
        thermal_dir.mkdir()
        ir_dir.mkdir()
        
        # 创建测试图像
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        for i in range(5):
            cv2.imwrite(str(thermal_dir / f"frame_{i:08d}.jpg"), img)
            cv2.imwrite(str(ir_dir / f"frame_{i:08d}.jpg"), img)
        
        yield tmpdir
        
# 1. 数据预处理与增强测试
def test_realistic_fire_smoke_augmentation():
    aug = RealisticFireSmokeAugmentation(fire_prob=1.0, smoke_prob=1.0)
    img = np.ones((64, 64), dtype=np.uint8) * 128
    
    # 测试烟雾和火焰增强
    result = aug(img)
    assert result.shape == (64, 64)
    assert result.dtype == np.uint8
    assert not np.array_equal(result, img)  
    
    # 测试概率为0时
    aug = RealisticFireSmokeAugmentation(fire_prob=0.0, smoke_prob=0.0)
    result = aug(img)
    assert np.array_equal(result, img) 

def test_physics_based_enhancement():
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    enhanced = physics_based_enhancement(img)
    assert enhanced.shape == (64, 64)
    assert enhanced.dtype == np.uint8
    assert not np.array_equal(enhanced, img) 

# 2. 数据集测试
def test_dataset(temp_data_dir, capsys):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    
    dataset = FireSmokeDataset(
        thermal_dir=str(Path(temp_data_dir) / "thermal"),
        ir_dir=str(Path(temp_data_dir) / "ir"),
        transform=transform
    )
    
    # 测试长度
    assert len(dataset) == 5
    
    # 测试正常getitem
    thermal, ir = dataset[0]
    assert isinstance(thermal, torch.Tensor)
    assert isinstance(ir, torch.Tensor)
    assert thermal.shape == (1, 64, 64)
    assert ir.shape == (1, 64, 64)
    
    # 测试错误处理 - 应该返回随机样本而不是抛出异常
    dataset.thermal_files.append("non_existent.jpg")
    thermal, ir = dataset[len(dataset)-1]  # 应该返回随机样本
    assert isinstance(thermal, torch.Tensor)
    assert isinstance(ir, torch.Tensor)
    
    # 检查是否打印了错误信息
    captured = capsys.readouterr()
    assert "无法读取热成像文件" in captured.out

    # 测试transform为None的情况
    dataset_no_transform = FireSmokeDataset(
        thermal_dir=str(Path(temp_data_dir) / "thermal"),
        ir_dir=str(Path(temp_data_dir) / "ir"),
        transform=None
    )
    thermal, ir = dataset_no_transform[0]
    assert isinstance(thermal, torch.Tensor)
    assert isinstance(ir, torch.Tensor)
    assert thermal.shape == (1, 64, 64)
    assert ir.shape == (1, 64, 64)

def test_realistic_fire_smoke_augmentation_combinations():
    aug = RealisticFireSmokeAugmentation()
    img = np.ones((64, 64), dtype=np.uint8) * 128
    
    # 测试不同概率组合
    combinations = [
        (1.0, 0.0),  # 只有火焰
        (0.0, 1.0),  # 只有烟雾
        (1.0, 1.0),  # 两者都有
        (0.0, 0.0)   # 两者都没有
    ]
    
    for fire_prob, smoke_prob in combinations:
        aug = RealisticFireSmokeAugmentation(fire_prob=fire_prob, smoke_prob=smoke_prob)
        result = aug(img)
        assert result.shape == (64, 64)
        assert result.dtype == np.uint8

def test_advanced_fusion_loss_edge_cases():
    criterion = AdvancedFusionLoss()
    
    # 测试全零输入
    zero_input = torch.zeros(2, 1, 64, 64)
    loss = criterion(zero_input, zero_input, zero_input)
    assert loss.item() >= 0
    
    # 测试全一输入
    ones_input = torch.ones(2, 1, 64, 64)
    loss = criterion(ones_input, ones_input, ones_input)
    assert loss.item() >= 0
    
    # 测试随机噪声输入
    noise_input = torch.rand(2, 1, 64, 64)
    loss = criterion(noise_input, noise_input, noise_input)
    assert loss.item() >= 0

def test_inference_error_handling(temp_data_dir):
    model_path = "non_existent_model.pth"
    thermal_path = str(Path(temp_data_dir) / "thermal/frame_00000000.jpg")
    ir_path = str(Path(temp_data_dir) / "ir/frame_00000000.jpg")
    output_path = str(Path(temp_data_dir) / "fused.jpg")
    
    # 测试模型文件不存在
    result = inference_single_image(
        thermal_path=thermal_path,
        ir_path=ir_path,
        output_path=output_path,
        model_path=model_path
    )
    assert result is None, "模型文件不存在时应返回None"
    
    # 测试输入图像不存在
    result = inference_single_image(
        thermal_path="non_existent.jpg",
        ir_path=ir_path,
        output_path=output_path,
        model_path="best_model_epoch31.pth"
    )
    assert result is None, "输入图像不存在时应返回None"

def test_spatial_registration_extended():
    # 创建测试图像 - 使用更复杂的图案
    img = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(img, (32, 32), 10, 255, -1)
    cv2.rectangle(img, (10, 10), (20, 20), 255, -1)  
    cv2.line(img, (40, 40), (50, 50), 255, 2) 
    
    # 测试不同的偏移量
    offsets = [1, 2, 4, 8]
    base_mi = compute_mi(img, img)
    
    for offset in offsets:
        # 水平偏移
        img_shift_h = np.roll(img, shift=offset, axis=1)
        mi_h = compute_mi(img, img_shift_h)
        assert mi_h <= base_mi, f"水平偏移{offset}像素的互信息应不大于原图"
        
        # 垂直偏移
        img_shift_v = np.roll(img, shift=offset, axis=0)
        mi_v = compute_mi(img, img_shift_v)
        assert mi_v <= base_mi, f"垂直偏移{offset}像素的互信息应不大于原图"
        
        # 旋转 - 使用更大的旋转角度和更复杂的变换
        if offset > 1:
            # 旋转 + 缩放
            M = cv2.getRotationMatrix2D((32, 32), offset * 10, 1.2)
            img_rot = cv2.warpAffine(img, M, (64, 64))
            mi_rot = compute_mi(img, img_rot)
            assert mi_rot < base_mi, f"旋转{offset*10}度并缩放1.2倍的互信息应小于原图"

# 3. 网络模块测试
def test_channel_attention():
    ca = ChannelAttention(64)
    x = torch.randn(2, 64, 32, 32)
    out = ca(x)
    assert out.shape == x.shape
    assert not torch.equal(out, x)  # 应该有注意力加权

def test_residual_block():
    rb = ResidualBlock(64)
    x = torch.randn(2, 64, 32, 32)
    out = rb(x)
    assert out.shape == x.shape
    assert not torch.equal(out, x)  # 应该有残差连接

def test_enhanced_encoder():
    encoder = EnhancedEncoder(in_channels=1, out_channels=128)
    x = torch.randn(2, 1, 64, 64)
    out = encoder(x)
    assert out.shape == (2, 128, 16, 16)  # 两次下采样

# 4. 损失函数测试
def test_ssim():
    ssim = SSIM()
    img1 = torch.randn(2, 1, 64, 64)
    img2 = torch.randn(2, 1, 64, 64)
    loss = ssim(img1, img2)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() <= 1.0  # SSIM值应在[0,1]范围内

def test_advanced_fusion_loss():
    criterion = AdvancedFusionLoss()
    fused = torch.randn(2, 1, 64, 64)
    thermal = torch.randn(2, 1, 64, 64)
    ir = torch.randn(2, 1, 64, 64)
    loss = criterion(fused, thermal, ir)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0  # 损失值应为正

# 5. 推理函数测试
def test_inference_single_image(temp_data_dir):
    model_path = "best_model_epoch31.pth"
    if not os.path.exists(model_path):
        pytest.skip("模型文件不存在，跳过测试")
    
    thermal_path = str(Path(temp_data_dir) / "thermal/frame_00000000.jpg")
    ir_path = str(Path(temp_data_dir) / "ir/frame_00000000.jpg")
    output_path = str(Path(temp_data_dir) / "fused.jpg")
    
    metrics = inference_single_image(
        thermal_path=thermal_path,
        ir_path=ir_path,
        output_path=output_path,
        model_path=model_path
    )
    
    assert os.path.exists(output_path)
    assert isinstance(metrics, dict)
    assert "entropy" in metrics
    assert "spatial_frequency" in metrics

def test_inference_batch(temp_data_dir):
    model_path = "best_model_epoch31.pth"
    if not os.path.exists(model_path):
        pytest.skip("模型文件不存在，跳过测试")
    
    output_dir = str(Path(temp_data_dir) / "output")
    os.makedirs(output_dir, exist_ok=True)
    
    inference_batch(
        thermal_dir=str(Path(temp_data_dir) / "thermal"),
        ir_dir=str(Path(temp_data_dir) / "ir"),
        output_dir=output_dir,
        model_path=model_path
    )
    
    # 检查输出文件
    assert len(os.listdir(output_dir)) > 0

# 原有的测试用例保持不变
def test_time_sync_bias():
    files = [f"frame_{i:08d}.jpg" for i in range(100, 110)]
    ir_files = files
    rgb_files = [f"frame_{i+np.random.randint(-3,4):08d}.jpg" for i in range(100, 110)]
    ir_times = [int(f.split('_')[1].split('.')[0]) for f in ir_files]
    rgb_times = [int(f.split('_')[1].split('.')[0]) for f in rgb_files]
    max_bias = max([abs(a-b) for a,b in zip(ir_times, rgb_times)])
    assert max_bias <= 3, f"时间同步偏差超出3秒: {max_bias}秒"

def test_spatial_registration():
    img = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(img, (32, 32), 10, 255, -1)
    offset = 2
    img_shift = np.roll(img, shift=offset, axis=1)
    mi = compute_mi(img, img_shift)
    sf1 = spatial_frequency(img)
    sf2 = spatial_frequency(img_shift)
    assert mi < compute_mi(img, img), "配准误差评估异常"
    assert abs(sf1-sf2) < 1e-3, "空间频率应基本一致"

def test_fusion_accuracy():
    model = EnhancedFusionModel()
    model.eval()
    batch = 8
    x1 = torch.rand(batch, 1, 64, 64)
    x2 = torch.rand(batch, 1, 64, 64)
    with torch.no_grad():
        y_pred = model(x1, x2)
    assert y_pred.shape == (batch, 1, 64, 64)
    assert not torch.isnan(y_pred).any()

def test_modality_failure():
    model = EnhancedFusionModel()
    model.eval()
    x1 = torch.rand(1, 1, 64, 64)
    x2 = torch.zeros(1, 1, 64, 64)
    with torch.no_grad():
        y_pred = model(x1, x2)
    assert not torch.isnan(y_pred).any()
    assert y_pred.abs().sum() > 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="无GPU环境")
def test_fusion_memory_usage():
    model = EnhancedFusionModel().cuda()
    x1 = torch.rand(1, 1, 256, 256).cuda()
    x2 = torch.rand(1, 1, 256, 256).cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(x1, x2)
    mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    assert mem <= 2048, f"GPU内存占用超2GB: {mem:.2f}MB"

def test_fusion_cpu_memory_usage():
    import gc
    gc.collect()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    model = EnhancedFusionModel()
    x1 = torch.rand(1, 1, 256, 256)
    x2 = torch.rand(1, 1, 256, 256)
    with torch.no_grad():
        _ = model(x1, x2)
    mem_after = process.memory_info().rss
    mem_used = (mem_after - mem_before) / 1024 / 1024
    assert mem_used <= 2048, f"CPU内存占用超2GB: {mem_used:.2f}MB"

def test_dataset_error_handling(temp_data_dir):
    """测试数据集的错误处理机制"""
    # 创建一个空目录
    empty_dir = Path(temp_data_dir) / "empty"
    empty_dir.mkdir()
    
    # 测试目录为空的情况
    dataset = FireSmokeDataset(
        thermal_dir=str(empty_dir),
        ir_dir=str(empty_dir),
        transform=None
    )
    assert len(dataset) == 0
    
    # 测试文件数量不匹配的情况
    thermal_dir = Path(temp_data_dir) / "thermal"
    ir_dir = Path(temp_data_dir) / "ir"
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    cv2.imwrite(str(thermal_dir / "extra.jpg"), img)
    
    with pytest.raises(AssertionError, match="数据不匹配"):
        FireSmokeDataset(
            thermal_dir=str(thermal_dir),
            ir_dir=str(ir_dir)
        )

def test_advanced_fusion_loss_gradient():
    """测试高级融合损失的梯度计算"""
    criterion = AdvancedFusionLoss()
    
    # 测试简单的梯度
    pred = torch.zeros(1, 1, 64, 64)
    target = torch.zeros(1, 1, 64, 64)
    pred[0, 0, 32:, 32:] = 1.0 
    target[0, 0, 32:, 32:] = 1.0
    
    grad_loss = criterion.gradient_loss(pred, target)
    assert grad_loss.item() == 0.0  
    
    # 测试不同的梯度
    target[0, 0, :32, :32] = 1.0  
    grad_loss = criterion.gradient_loss(pred, target)
    assert grad_loss.item() > 0.0 

def test_ssim_edge_cases():
    """测试SSIM的边界情况"""
    ssim = SSIM(window_size=5, sigma=1.0) 
    
    # 测试完全相同的图像
    img1 = torch.ones(1, 1, 32, 32)
    assert ssim(img1, img1).item() == 1.0 
    
    # 测试完全不同的图像
    img2 = torch.zeros(1, 1, 32, 32)
    similarity = ssim(img1, img2).item()
    assert 0.0 <= similarity <= 1.0  
    
    # 测试随机噪声图像
    noise1 = torch.rand(1, 1, 32, 32)
    noise2 = torch.rand(1, 1, 32, 32)
    similarity = ssim(noise1, noise2).item()
    assert 0.0 <= similarity <= 1.0

def test_fusion_model_shapes():
    """测试融合模型在不同输入尺寸下的行为"""
    model = EnhancedFusionModel()
    model.eval()
    
    # 测试不同的输入尺寸
    sizes = [(64, 64), (128, 128), (256, 256)]
    for h, w in sizes:
        thermal = torch.rand(1, 1, h, w)
        ir = torch.rand(1, 1, h, w)
        with torch.no_grad():
            output = model(thermal, ir)
        assert output.shape == (1, 1, h, w), f"输出尺寸应为{(1, 1, h, w)}，但得到{output.shape}"
        assert torch.all((output >= 0) & (output <= 1)), "输出应在[0,1]范围内"

def test_mutual_information_properties():
    """测试互信息的基本性质"""
    # 创建测试图像
    img1 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    
    # 测试自身的互信息
    mi_self = compute_mi(img1, img1)
    assert mi_self >= 0, "互信息应该非负"
    
    # 测试对称性
    mi_12 = compute_mi(img1, img2)
    mi_21 = compute_mi(img2, img1)
    assert np.abs(mi_12 - mi_21) < 1e-10, "互信息应该具有对称性"
    
    # 测试不同bins的影响
    mi_256 = compute_mi(img1, img2, bins=256)
    mi_128 = compute_mi(img1, img2, bins=128)
    assert mi_256 != mi_128, "不同的bins应该产生不同的互信息值"
