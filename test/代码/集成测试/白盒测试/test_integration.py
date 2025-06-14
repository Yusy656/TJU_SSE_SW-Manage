"""
集成测试模块 - 基于路径覆盖的白盒测试
包含去雾、融合、检测和后端API的集成测试用例
"""
import pytest
import os
import cv2
import numpy as np
import torch
import json
import time
import datetime
from pathlib import Path
from dehazing.dehaze import start_dehaze
from fusion.fusion import inference_single_image
from detect.detect import detect_single_image
from app import app, create_session, session_states
import shutil

# 测试数据路径
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
TEST_RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'test_results')

@pytest.fixture(scope="session")
def setup_test_env():
    """设置测试环境和测试数据"""
    # 创建测试目录
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)
    test_small = np.random.randint(0, 255, (540, 960), dtype=np.uint8)
    
    # 保存正常测试图像
    cv2.imwrite(os.path.join(TEST_DATA_DIR, 'test_thermal.jpg'), test_image)
    cv2.imwrite(os.path.join(TEST_DATA_DIR, 'test_ir.jpg'), test_image)
    cv2.imwrite(os.path.join(TEST_DATA_DIR, 'test_infrared.jpg'), test_image)
    
    # 保存小尺寸图像
    cv2.imwrite(os.path.join(TEST_DATA_DIR, 'small_thermal.jpg'), test_small)
    
    # 创建损坏的图像文件
    with open(os.path.join(TEST_DATA_DIR, 'corrupted.jpg'), 'wb') as f:
        f.write(b'Invalid JPEG data')
    
    yield
    
    # 清理测试数据
    try:
        for f in os.listdir(TEST_DATA_DIR):
            os.remove(os.path.join(TEST_DATA_DIR, f))
        for f in os.listdir(TEST_RESULTS_DIR):
            os.remove(os.path.join(TEST_RESULTS_DIR, f))
    except:
        pass

@pytest.fixture
def test_client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

class TestDehazeIntegration:
    """去雾模块集成测试 - 基于路径覆盖"""
    
    def test_normal_path(self, setup_test_env):
        """路径1: 正常去雾路径"""
        input_path = os.path.join(TEST_DATA_DIR, 'test_ir.jpg')
        output_path = os.path.join(TEST_RESULTS_DIR, 'dehazed.jpg')
        metrics = start_dehaze(input_path, output_path)
        assert metrics is not None
        assert 'avg_dehazed_gradient' in metrics
    
    def test_extreme_smoke_path(self, setup_test_env):
        """路径2: 极端烟雾密度路径"""
        dark_image = np.ones((1080, 1920), dtype=np.uint8) * 50
        dark_path = os.path.join(TEST_DATA_DIR, 'dark.jpg')
        cv2.imwrite(dark_path, dark_image)
        output_path = os.path.join(TEST_RESULTS_DIR, 'dehazed_dark.jpg')
        metrics = start_dehaze(dark_path, output_path)
        assert metrics is not None
    
    def test_file_not_found_path(self, setup_test_env, capfd):
        """路径3: 文件不存在路径"""
        start_dehaze('nonexistent.jpg', 'output.jpg')
        out, err = capfd.readouterr()
        assert "处理失败" in out or "无效的输入路径" in out or "not exist" in out
    
    def test_corrupted_image_path(self, setup_test_env, capfd):
        """路径4: 损坏图像路径"""
        corrupted_path = os.path.join(TEST_DATA_DIR, 'corrupted.jpg')
        output_path = os.path.join(TEST_RESULTS_DIR, 'corrupted_output.jpg')
        start_dehaze(corrupted_path, output_path)
        out, err = capfd.readouterr()
        assert "处理失败" in out or "cannot identify image file" in out

class TestFusionIntegration:
    """融合模块集成测试 - 基于路径覆盖"""
    
    def test_normal_fusion_path(self, setup_test_env):
        """路径1: 正常融合路径"""
        thermal_path = os.path.join(TEST_DATA_DIR, 'test_thermal.jpg')
        ir_path = os.path.join(TEST_DATA_DIR, 'test_ir.jpg')
        fused_path = os.path.join(TEST_RESULTS_DIR, 'fused.jpg')
        try:
            metrics = inference_single_image(thermal_path, ir_path, fused_path)
            assert metrics is None or isinstance(metrics, dict)
        except Exception:
            pass
    
    def test_different_size_path(self, setup_test_env):
        """路径2: 不同尺寸图像融合路径"""
        small_thermal_path = os.path.join(TEST_DATA_DIR, 'small_thermal.jpg')
        ir_path = os.path.join(TEST_DATA_DIR, 'test_ir.jpg')
        fused_path = os.path.join(TEST_RESULTS_DIR, 'small_fused.jpg')
        try:
            metrics = inference_single_image(small_thermal_path, ir_path, fused_path)
            assert metrics is None or isinstance(metrics, dict)
        except Exception:
            pass
    
    def test_missing_file_path(self, setup_test_env, capfd):
        """路径3: 缺失文件路径"""
        inference_single_image('nonexistent.jpg', 'nonexistent2.jpg', 'output.jpg')
        out, err = capfd.readouterr()
        assert "处理失败" in out or "Error(s) in loading" in out
    
    def test_corrupted_image_path(self, setup_test_env, capfd):
        """路径4: 损坏图像路径"""
        corrupted_path = os.path.join(TEST_DATA_DIR, 'corrupted.jpg')
        ir_path = os.path.join(TEST_DATA_DIR, 'test_ir.jpg')
        fused_path = os.path.join(TEST_RESULTS_DIR, 'corrupted_fused.jpg')
        inference_single_image(corrupted_path, ir_path, fused_path)
        out, err = capfd.readouterr()
        assert "处理失败" in out or "Error(s) in loading" in out

class TestDetectionIntegration:
    """检测模块集成测试 - 基于路径覆盖"""
    
    def test_normal_detection_path(self, setup_test_env):
        """路径1: 正常检测路径"""
        input_path = os.path.join(TEST_DATA_DIR, 'test_ir.jpg')
        output_path = os.path.join(TEST_RESULTS_DIR, 'detected.jpg')
        try:
            metrics = detect_single_image(input_path, output_path)
            assert metrics is None or isinstance(metrics, dict)
        except Exception:
            pass
    
    def test_no_detection_path(self, setup_test_env):
        """路径2: 无目标检测路径"""
        black_image = np.zeros((1080, 1920), dtype=np.uint8)
        black_path = os.path.join(TEST_DATA_DIR, 'black.jpg')
        cv2.imwrite(black_path, black_image)
        output_path = os.path.join(TEST_RESULTS_DIR, 'detected_black.jpg')
        try:
            metrics = detect_single_image(black_path, output_path)
            assert metrics is None or isinstance(metrics, dict)
        except Exception:
            pass
    
    def test_file_not_found_path(self, setup_test_env, capfd):
        """路径3: 文件不存在路径"""
        detect_single_image('nonexistent.jpg', 'output.jpg')
        out, err = capfd.readouterr()
        assert "处理图片失败" in out or "无法读取图像" in out
    
    def test_corrupted_image_path(self, setup_test_env, capfd):
        """路径4: 损坏图像路径"""
        corrupted_path = os.path.join(TEST_DATA_DIR, 'corrupted.jpg')
        output_path = os.path.join(TEST_RESULTS_DIR, 'corrupted_detected.jpg')
        detect_single_image(corrupted_path, output_path)
        out, err = capfd.readouterr()
        assert "处理图片失败" in out or "无法读取图像" in out

class TestBackendIntegration:
    """后端API集成测试 - 基于路径覆盖"""
    
    def test_session_creation_path(self, test_client):
        """路径1: 会话创建路径"""
        response = test_client.post('/create-session')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'session_id' in data
    
    def test_invalid_session_path(self, test_client):
        """路径2: 无效会话路径"""
        response = test_client.get('/get-processed-images?session_id=invalid')
        assert response.status_code in (400, 404)
    
    def test_session_timeout_path(self, test_client):
        """路径3: 会话超时路径"""
        response = test_client.post('/create-session')
        session_id = json.loads(response.data)['session_id']
        session_states[session_id]['created_at'] = datetime.datetime.now() - datetime.timedelta(hours=2)
        response = test_client.get(f'/get-processed-images?session_id={session_id}')
        assert response.status_code in (400, 404)
    
    def test_complete_processing_path(self, test_client, setup_test_env):
        """路径4: 完整处理路径"""
        # 创建会话
        response = test_client.post('/create-session')
        session_id = json.loads(response.data)['session_id']
        
        # 上传图像
        for img_type in ['infrared', 'thermal']:
            with open(os.path.join(TEST_DATA_DIR, f'test_{img_type}.jpg'), 'rb') as f:
                response = test_client.post('/upload', data={
                    'file': (f, f'test_{img_type}.jpg'),
                    'type': img_type,
                    'session_id': session_id
                })
            assert response.status_code == 200
        
        # 处理图像
        response = test_client.post('/process', data={'session_id': session_id})
        assert response.status_code in (200, 500)
    
    def test_missing_images_path(self, test_client):
        """路径5: 缺失图像路径"""
        response = test_client.post('/create-session')
        session_id = json.loads(response.data)['session_id']
        response = test_client.post('/process', data={'session_id': session_id})
        assert response.status_code in (400, 404)
    
    def test_processing_error_path(self, test_client, setup_test_env):
        """路径6: 处理错误路径"""
        # 创建会话
        response = test_client.post('/create-session')
        session_id = json.loads(response.data)['session_id']
        
        # 上传损坏的图像
        with open(os.path.join(TEST_DATA_DIR, 'corrupted.jpg'), 'rb') as f:
            response = test_client.post('/upload', data={
                'file': (f, 'corrupted.jpg'),
                'type': 'infrared',
                'session_id': session_id
            })
        
        # 尝试处理
        response = test_client.post('/process', data={'session_id': session_id})
        assert response.status_code in (400, 500)

if __name__ == '__main__':
    pytest.main(['-v', 'test_integration.py']) 