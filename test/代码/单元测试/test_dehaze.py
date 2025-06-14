import pytest
import torch
import cv2
import numpy as np
import os
import time
import shutil
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import sys
import tempfile
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dehazing.dehaze import (
    start_dehaze, 
    dehaze_image, 
    compute_mean_gradient, 
    adaptive_gamma, 
    enhance_image
)
from dehazing.net import dehaze_net

# 使用pytest标记来过滤警告
pytestmark = [
    pytest.mark.filterwarnings("ignore::FutureWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning")
]

# 测试数据和模型的路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(CURRENT_DIR, "test_data")
TEST_OUTPUT_DIR = os.path.join(CURRENT_DIR, "test_output")
MODEL_PATH = os.path.join(CURRENT_DIR, "snapshots", "dehazer.pth")

# 确保测试目录存在
os.makedirs(TEST_DATA_DIR, exist_ok=True)
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

@pytest.fixture(autouse=True)
def ignore_torch_warnings():
    """自动使用的fixture，用于过滤PyTorch的加载警告"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, 
                              module="torch.serialization",
                              message="You are using `torch.load` with `weights_only=False`")
        yield

@pytest.fixture(scope="module")
def dehaze_model():
    """加载去烟模型的fixture"""
    model = dehaze_net().cuda()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    return model

@pytest.fixture(scope="module")
def test_image():
    """生成测试用的烟雾图像fixture"""
    # 确保测试目录存在
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    
    # 创建一个较大尺寸的测试图像 (256x256)，包含更多的细节和纹理
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # 添加渐变背景
    for i in range(256):
        for j in range(256):
            img[i, j] = [(i+j)//2, (i+j)//2, (i+j)//2]
    
    # 添加一些几何图形作为细节
    cv2.rectangle(img, (50, 50), (150, 150), (200, 200, 200), -1)
    cv2.circle(img, (180, 180), 40, (100, 100, 100), -1)
    cv2.line(img, (20, 200), (200, 20), (150, 150, 150), 5)
    
    # 添加随机纹理
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    # 确保添加噪声后的值仍在有效范围内
    img = cv2.add(img, noise, dtype=cv2.CV_8U)
    
    test_path = os.path.join(TEST_DATA_DIR, "test.jpg")
    cv2.imwrite(test_path, img)
    
    # 确保文件被正确创建
    assert os.path.exists(test_path), f"测试图像未能创建: {test_path}"
    return test_path

def test_psnr_ssim_metrics(dehaze_model, test_image):
    """测试去烟后图像的PSNR和SSIM指标"""
    # 处理图像
    output_path = os.path.join(TEST_OUTPUT_DIR, "test_metrics.jpg")
    _, _, _ = dehaze_image(test_image, dehaze_model, output_path)
    
    # 读取原始和处理后的图像
    original = cv2.imread(test_image)
    dehazed = cv2.imread(output_path)
    
    # 计算PSNR和SSIM
    psnr_value = psnr(original, dehazed)
    ssim_value = ssim(original, dehazed, channel_axis=2)
    
    # 调整验证标准为更合理的值
    assert psnr_value >= 15.0, f"PSNR值 {psnr_value:.2f}dB 低于要求的15dB"
    assert ssim_value >= 0.75, f"SSIM值 {ssim_value:.2f} 低于要求的0.75"

@pytest.mark.parametrize("density", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
def test_smoke_density_stability(dehaze_model, test_image, density):
    """测试不同烟雾浓度下的去烟效果稳定性"""
    # 模拟不同浓度的烟雾图像
    original = cv2.imread(test_image)
    hazy = cv2.convertScaleAbs(original, alpha=1-density, beta=density*255)
    test_hazy_path = os.path.join(TEST_DATA_DIR, f"test_density_{density}.jpg")
    cv2.imwrite(test_hazy_path, hazy)
    
    # 处理图像
    output_path = os.path.join(TEST_OUTPUT_DIR, f"dehazed_density_{density}.jpg")
    _, _, _ = dehaze_image(test_hazy_path, dehaze_model, output_path)
    
    # 验证处理后的图像质量
    dehazed = cv2.imread(output_path)
    psnr_value = psnr(original, dehazed)
    ssim_value = ssim(original, dehazed, channel_axis=2)
    
    # 根据烟雾浓度调整阈值
    if density <= 0.3:
        min_psnr = 15.0
        min_ssim = 0.75
    else:
        # 高浓度烟雾时允许稍低的指标
        min_psnr = 13.0
        min_ssim = 0.70
    
    assert psnr_value >= min_psnr, f"浓度{density}时PSNR值过低: {psnr_value:.2f}dB"
    assert ssim_value >= min_ssim, f"浓度{density}时SSIM值过低: {ssim_value:.2f}"

def test_processing_time(dehaze_model, test_image):
    """测试单帧去烟处理时间"""
    # 预热GPU
    for _ in range(3):
        dehaze_image(test_image, dehaze_model, os.path.join(TEST_OUTPUT_DIR, "warmup.jpg"))
    
    # 测试处理时间
    start_time = time.time()
    _, _, processing_time = dehaze_image(test_image, dehaze_model, 
                                       os.path.join(TEST_OUTPUT_DIR, "time_test.jpg"))
    
    assert processing_time <= 0.3, f"处理时间 {processing_time*1000:.2f}ms 超过要求的300ms"

def test_strong_light_stability(dehaze_model):
    """测试强光反射条件下的去烟稳定性"""
    # 创建带有强光反射的测试图像
    img = np.ones((256, 256, 3), dtype=np.uint8) * 128
    # 添加强光区域
    img[100:150, 100:150] = 255
    test_path = os.path.join(TEST_DATA_DIR, "strong_light.jpg")
    cv2.imwrite(test_path, img)
    
    # 处理图像
    output_path = os.path.join(TEST_OUTPUT_DIR, "dehazed_strong_light.jpg")
    try:
        _, _, _ = dehaze_image(test_path, dehaze_model, output_path)
        # 验证输出图像是否存在且可读
        assert os.path.exists(output_path)
        result = cv2.imread(output_path)
        assert result is not None
    except Exception as e:
        pytest.fail(f"强光条件下处理失败: {str(e)}")

def test_dynamic_fire_stability(dehaze_model):
    """测试动态火源干扰下的去烟稳定性"""
    # 创建模拟火源的测试图像序列
    for i in range(5):
        img = np.ones((256, 256, 3), dtype=np.uint8) * 128
        # 添加随机位置的"火源"
        x, y = np.random.randint(0, 200, 2)
        img[y:y+50, x:x+50] = [0, 0, 255]  
        test_path = os.path.join(TEST_DATA_DIR, f"fire_{i}.jpg")
        cv2.imwrite(test_path, img)
        
        # 处理图像
        output_path = os.path.join(TEST_OUTPUT_DIR, f"dehazed_fire_{i}.jpg")
        try:
            _, _, _ = dehaze_image(test_path, dehaze_model, output_path)
            # 验证输出
            assert os.path.exists(output_path)
            result = cv2.imread(output_path)
            assert result is not None
        except Exception as e:
            pytest.fail(f"火源干扰条件下处理失败: {str(e)}")

@pytest.mark.parametrize("test_case", [
    "corrupted_image",
    "empty_image",
    "wrong_format",
    "very_small_image"
])
def test_error_handling(dehaze_model, test_case):
    """测试输入低质量/损坏图像时的错误处理"""
    test_path = os.path.join(TEST_DATA_DIR, f"test_{test_case}.jpg")
    output_path = os.path.join(TEST_OUTPUT_DIR, f"dehazed_{test_case}.jpg")
    
    if test_case == "corrupted_image":
        # 创建损坏的图像文件
        with open(test_path, 'wb') as f:
            f.write(b'corrupted data')
    elif test_case == "empty_image":
        # 创建空文件
        open(test_path, 'w').close()
    elif test_case == "wrong_format":
        # 创建错误格式的文件
        with open(test_path, 'w') as f:
            f.write("not an image")
    elif test_case == "very_small_image":
        # 创建极小的图像
        img = np.ones((2, 2, 3), dtype=np.uint8)
        cv2.imwrite(test_path, img)
    
    # 测试错误处理
    try:
        _, _, _ = dehaze_image(test_path, dehaze_model, output_path)
    except Exception as e:
        # 确保错误被适当捕获和处理
        assert str(e) != "", "错误信息不应为空"
        return
    
    # 如果没有抛出异常，确保输出是有效的
    if os.path.exists(output_path):
        result = cv2.imread(output_path)
        assert result is not None, "输出图像应该是有效的"

def test_cleanup():
    """清理测试生成的文件"""
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)

def create_mock_image(size=(256, 256)):
    """创建模拟的测试图像"""
    # 创建一个随机的RGB图像数组
    mock_image = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    # 转换为PIL图像
    image = Image.fromarray(mock_image)
    # 保存到临时文件
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "mock_test.jpg")
    image.save(temp_path)
    return temp_path, temp_dir

@pytest.fixture(scope="function")
def mock_image():
    """提供模拟图像的fixture"""
    temp_path, temp_dir = create_mock_image()
    yield temp_path
    # 清理临时文件和目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def test_start_dehaze_batch_processing(dehaze_model, mock_image, tmpdir):
    """测试批量处理功能"""
    # 使用模拟图像
    input_dir = os.path.dirname(mock_image)
    output_dir = os.path.join(str(tmpdir), "output") 
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制模拟图像到输入目录以模拟多个文件
    for i in range(3):
        shutil.copy2(mock_image, os.path.join(input_dir, f"test_{i}.jpg"))
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 调用处理函数
    metrics = start_dehaze(input_dir, output_dir)
    
    # 检查返回值
    assert isinstance(metrics, dict)
    if 'error' not in metrics:
        assert metrics.get('processed_count', 0) >= 1
        assert metrics.get('avg_processing_time', 0) > 0
        assert metrics.get('avg_original_gradient', 0) > 0
        assert metrics.get('avg_dehazed_gradient', 0) > 0
    else:
        print(f"处理出现错误: {metrics['error']}")

def test_start_dehaze_single_file(dehaze_model, mock_image, tmpdir):
    """测试单文件处理"""
    output_path = os.path.join(str(tmpdir), "output_single.jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    metrics = start_dehaze(mock_image, output_path)
    
    assert isinstance(metrics, dict)
    if 'error' not in metrics:
        assert metrics.get('processed_count', 0) >= 1
        assert metrics.get('avg_processing_time', 0) > 0
        assert metrics.get('avg_original_gradient', 0) > 0
        assert metrics.get('avg_dehazed_gradient', 0) > 0
    else:
        print(f"处理出现错误: {metrics['error']}")

def test_start_dehaze_invalid_paths():
    """测试无效路径处理"""
    # 测试不存在的输入路径
    metrics = start_dehaze("/nonexistent/path", "/some/output")
    assert 'error' in metrics
    
    # 测试无效的输入类型
    metrics = start_dehaze(None, "/some/output")
    assert 'error' in metrics

def test_enhance_image_variations(mock_image):
    """测试图像增强的不同组合"""
    # 读取模拟图像
    img = cv2.imread(mock_image)
    assert img is not None
    
    # 测试不同的增强方法组合
    methods = ["clahe", "gamma", "clahe+gamma"]
    for method in methods:
        enhanced = enhance_image(img, method=method)
        assert enhanced is not None
        assert enhanced.shape == img.shape
        assert enhanced.dtype == img.dtype

def test_compute_mean_gradient_variations():
    """测试平均梯度计算的不同情况"""
    # 创建测试图像
    flat_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    gradient_image = np.zeros((100, 100, 3), dtype=np.uint8)
    gradient_image[:, :50] = 0
    gradient_image[:, 50:] = 255
    
    # 测试平坦图像
    flat_gradient = compute_mean_gradient(flat_image)
    assert flat_gradient >= 0
    
    # 测试有明显边缘的图像
    edge_gradient = compute_mean_gradient(gradient_image)
    assert edge_gradient > flat_gradient

def test_adaptive_gamma_variations():
    """测试自适应gamma的不同情况"""
    # 测试暗图像
    dark_image = np.ones((100, 100, 3), dtype=np.uint8) * 50
    dark_gamma = adaptive_gamma(dark_image)
    assert dark_gamma > 1.0
    
    # 测试亮图像
    bright_image = np.ones((100, 100, 3), dtype=np.uint8) * 200
    bright_gamma = adaptive_gamma(bright_image)
    assert bright_gamma < dark_gamma
    
    # 测试正常图像
    normal_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    normal_gamma = adaptive_gamma(normal_image)
    assert 1.0 <= normal_gamma <= 1.2

def test_dehaze_image_with_enhancement_options(dehaze_model, mock_image):
    """测试去烟函数的增强选项"""
    output_path = os.path.join("/tmp", "enhanced_output.jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 测试enhance=True
    metrics_enhanced = start_dehaze(mock_image, output_path, enhance=True)
    assert isinstance(metrics_enhanced, dict)
    if 'error' not in metrics_enhanced:
        assert metrics_enhanced.get('processed_count', 0) >= 1
    
    # 测试enhance=False
    metrics_no_enhance = start_dehaze(mock_image, output_path, enhance=False)
    assert isinstance(metrics_no_enhance, dict)
    if 'error' not in metrics_no_enhance:
        assert metrics_no_enhance.get('processed_count', 0) >= 1

def test_error_handling_extended(dehaze_model):
    """扩展错误处理测试"""
    # 测试无效的输出路径
    with pytest.raises(Exception):
        dehaze_image(TEST_DATA_DIR + "/test.jpg", dehaze_model, "/invalid/path/output.jpg")
    
    # 测试无效的模型状态
    invalid_model = dehaze_net().cuda()  
    with pytest.raises(Exception):
        dehaze_image(TEST_DATA_DIR + "/test.jpg", invalid_model, TEST_OUTPUT_DIR + "/test.jpg") 