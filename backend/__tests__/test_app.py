import unittest
import json
import os
from app import app, session_states
import tempfile
import shutil
import cv2
import numpy as np
from PIL import Image
import io

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        # 创建临时目录用于测试
        self.test_dir = tempfile.mkdtemp()
        app.config['THERMAL_UPLOAD_FOLDER'] = os.path.join(self.test_dir, 'thermal')
        app.config['INFRARED_UPLOAD_FOLDER'] = os.path.join(self.test_dir, 'infrared')
        app.config['DEHAZED_RESULTS_FOLDER'] = os.path.join(self.test_dir, 'dehazed')
        app.config['FUSED_RESULTS_FOLDER'] = os.path.join(self.test_dir, 'fused')
        app.config['DETECTED_RESULTS_FOLDER'] = os.path.join(self.test_dir, 'detected')
        
        # 创建必要的目录
        for folder in [app.config['THERMAL_UPLOAD_FOLDER'],
                      app.config['INFRARED_UPLOAD_FOLDER'],
                      app.config['DEHAZED_RESULTS_FOLDER'],
                      app.config['FUSED_RESULTS_FOLDER'],
                      app.config['DETECTED_RESULTS_FOLDER']]:
            os.makedirs(folder, exist_ok=True)

        # 创建测试用的图像文件
        self.create_test_images()

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.test_dir)
        # 清理会话状态
        session_states.clear()

    def create_test_images(self):
        # 创建测试用的热成像图像
        thermal_img = np.zeros((100, 100, 3), dtype=np.uint8)
        thermal_img[..., 0] = 255  # 红色通道
        cv2.imwrite(os.path.join(app.config['THERMAL_UPLOAD_FOLDER'], 'test_thermal.jpg'), thermal_img)

        # 创建测试用的红外图像
        ir_img = np.zeros((100, 100, 3), dtype=np.uint8)
        ir_img[..., 1] = 255  # 绿色通道
        cv2.imwrite(os.path.join(app.config['INFRARED_UPLOAD_FOLDER'], 'test_ir.jpg'), ir_img)

    def test_index(self):
        """测试根路由"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), 'Backend is running!')

    def test_create_session(self):
        """测试会话创建"""
        response = self.app.post('/create-session')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('session_id', data)
        self.assertIn('message', data)
        self.assertEqual(data['message'], 'Session created successfully')

    def test_upload_image_valid(self):
        """测试有效图像上传"""
        # 创建会话
        session_response = self.app.post('/create-session')
        session_data = json.loads(session_response.data)
        session_id = session_data['session_id']

        # 上传热成像图像
        with open(os.path.join(app.config['THERMAL_UPLOAD_FOLDER'], 'test_thermal.jpg'), 'rb') as f:
            response = self.app.post('/upload', data={
                'file': (f, 'test_thermal.jpg'),
                'type': 'thermal',
                'session_id': session_id
            })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('filename', data)

        # 上传红外图像
        with open(os.path.join(app.config['INFRARED_UPLOAD_FOLDER'], 'test_ir.jpg'), 'rb') as f:
            response = self.app.post('/upload', data={
                'file': (f, 'test_ir.jpg'),
                'type': 'infrared',
                'session_id': session_id
            })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('filename', data)

    def test_upload_image_invalid_session(self):
        """测试无效会话ID上传"""
        response = self.app.post('/upload', data={
            'file': (b'test', 'test.jpg'),
            'type': 'thermal',
            'session_id': 'invalid_session'
        })
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_upload_image_missing_file(self):
        """测试缺少文件上传"""
        session_response = self.app.post('/create-session')
        session_data = json.loads(session_response.data)
        session_id = session_data['session_id']

        response = self.app.post('/upload', data={
            'type': 'thermal',
            'session_id': session_id
        })
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_upload_image_invalid_type(self):
        """测试无效图像类型上传"""
        session_response = self.app.post('/create-session')
        session_data = json.loads(session_response.data)
        session_id = session_data['session_id']

        response = self.app.post('/upload', data={
            'file': (b'test', 'test.jpg'),
            'type': 'invalid_type',
            'session_id': session_id
        })
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_process_pipeline_valid(self):
        """测试完整的处理流程"""
        # 创建会话
        session_response = self.app.post('/create-session')
        session_data = json.loads(session_response.data)
        session_id = session_data['session_id']

        # 上传两种图像
        with open(os.path.join(app.config['THERMAL_UPLOAD_FOLDER'], 'test_thermal.jpg'), 'rb') as f:
            self.app.post('/upload', data={
                'file': (f, 'test_thermal.jpg'),
                'type': 'thermal',
                'session_id': session_id
            })

        with open(os.path.join(app.config['INFRARED_UPLOAD_FOLDER'], 'test_ir.jpg'), 'rb') as f:
            self.app.post('/upload', data={
                'file': (f, 'test_ir.jpg'),
                'type': 'infrared',
                'session_id': session_id
            })

        # 执行处理
        response = self.app.post('/process', data={
            'session_id': session_id
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('message', data)

    def test_process_pipeline_invalid_session(self):
        """测试无效会话的处理流程"""
        response = self.app.post('/process', data={
            'session_id': 'invalid_session'
        })
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_get_processed_images(self):
        """测试获取处理后的图像"""
        # 创建会话并上传图像
        session_response = self.app.post('/create-session')
        session_data = json.loads(session_response.data)
        session_id = session_data['session_id']

        # 上传图像并处理
        with open(os.path.join(app.config['THERMAL_UPLOAD_FOLDER'], 'test_thermal.jpg'), 'rb') as f:
            self.app.post('/upload', data={
                'file': (f, 'test_thermal.jpg'),
                'type': 'thermal',
                'session_id': session_id
            })

        with open(os.path.join(app.config['INFRARED_UPLOAD_FOLDER'], 'test_ir.jpg'), 'rb') as f:
            self.app.post('/upload', data={
                'file': (f, 'test_ir.jpg'),
                'type': 'infrared',
                'session_id': session_id
            })

        self.app.post('/process', data={'session_id': session_id})

        # 获取处理后的图像
        response = self.app.get(f'/get-processed-images?session_id={session_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # 验证返回的数据结构
        self.assertIn('combinedImage', data)
        self.assertIn('dehazingImage', data)
        self.assertIn('fusingImage', data)
        self.assertIn('metrics', data)
        self.assertIn('status', data)
        
        # 验证状态
        self.assertEqual(data['status'], 'completed')
        
        # 验证图像路径
        self.assertTrue(data['combinedImage'].startswith('/results/'))
        self.assertTrue(data['dehazingImage'].startswith('/results/'))
        self.assertTrue(data['fusingImage'].startswith('/results/'))
        
        # 验证指标数据
        metrics = data['metrics']
        self.assertIn('dehazing', metrics)
        self.assertIn('fusion', metrics)
        self.assertIn('detection', metrics)
        self.assertIn('resource_usage', metrics)

    def test_get_processed_images_invalid_session(self):
        """测试获取无效会话的处理图像"""
        response = self.app.get('/get-processed-images?session_id=invalid_session')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_system_metrics(self):
        """测试系统指标获取"""
        response = self.app.get('/system-metrics')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('cpu', data)
        self.assertIn('memory', data)
        self.assertIn('gpu', data)

if __name__ == '__main__':
    unittest.main() 