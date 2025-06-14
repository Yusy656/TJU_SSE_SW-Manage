import os
import unittest
import cv2
import numpy as np
import requests
import json
from pathlib import Path
from PIL import Image
import io

class TestImageProcessingBlackBox(unittest.TestCase):
    """图像处理系统黑盒测试 - 使用等价类划分方法"""

    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 设置测试目录
        cls.test_dir = Path(__file__).parent / "test_data"
        cls.input_dir = cls.test_dir / "input"
        cls.output_dir = cls.test_dir / "output"
        
        # 创建测试目录
        for dir_path in [cls.input_dir, cls.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # API端点
        cls.base_url = "http://localhost:5000"
        
    def setUp(self):
        """每个测试用例前的准备工作"""
        # 创建会话
        response = requests.post(f"{self.base_url}/create-session")
        self.assertTrue(response.status_code == 200, "创建会话失败")
        self.session_id = response.json()['session_id']
        
    def generate_test_image(self, size=(640, 480), format='jpg', is_noise=False, is_blur=False):
        """生成测试图像"""
        # 生成随机图像
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        
        # 添加噪声
        if is_noise:
            noise = np.random.normal(0, 25, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
        # 添加模糊
        if is_blur:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            
        # 保存图像
        file_path = str(self.input_dir / f"test_image_{hash(str(size))}_{format}.{format}")
        if format.lower() == 'jpg':
            cv2.imwrite(file_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        elif format.lower() == 'png':
            cv2.imwrite(file_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            # 使用PIL处理其他格式
            Image.fromarray(img).save(file_path, format=format.upper())
            
        return file_path
        
    def process_image_pair(self, thermal_path, ir_path):
        """通过API处理图像对"""
        try:
            # 上传热成像图
            with open(thermal_path, 'rb') as f:
                response = requests.post(
                    f"{self.base_url}/upload",
                    files={'file': f},
                    data={
                        'type': 'thermal',
                        'session_id': self.session_id
                    }
                )
                self.assertTrue(response.status_code == 200, "上传热成像失败")
                
            # 上传红外图
            with open(ir_path, 'rb') as f:
                response = requests.post(
                    f"{self.base_url}/upload",
                    files={'file': f},
                    data={
                        'type': 'infrared',
                        'session_id': self.session_id
                    }
                )
                self.assertTrue(response.status_code == 200, "上传红外图像失败")
                
            # 处理图像
            response = requests.post(
                f"{self.base_url}/process",
                data={'session_id': self.session_id}
            )
            
            if response.status_code != 200:
                return False, None
                
            # 获取处理结果
            response = requests.get(
                f"{self.base_url}/get-processed-images",
                params={'session_id': self.session_id}
            )
            
            if response.status_code != 200:
                return False, None
                
            result = response.json()
            if result['status'] != 'completed':
                return False, None
                
            return True, result
            
        except Exception as e:
            print(f"处理失败: {str(e)}")
            return False, None
            
    # ===== 有效等价类测试 =====
    def test_EC1_standard_image(self):
        """EC1-有效等价类：标准图像(640x480, JPG格式)"""
        thermal_path = self.generate_test_image(size=(640, 480), format='jpg')
        ir_path = self.generate_test_image(size=(640, 480), format='jpg')
        
        success, result = self.process_image_pair(thermal_path, ir_path)
        self.assertTrue(success, "标准图像处理应该成功")
        self.assertIsNotNone(result, "应该返回处理结果")
        
    def test_EC2_hd_image(self):
        """EC2-有效等价类：高清图像(1920x1080, JPG格式)"""
        thermal_path = self.generate_test_image(size=(1920, 1080), format='jpg')
        ir_path = self.generate_test_image(size=(1920, 1080), format='jpg')
        
        success, result = self.process_image_pair(thermal_path, ir_path)
        self.assertTrue(success, "高清图像处理应该成功")
        self.assertIsNotNone(result, "应该返回处理结果")
        
    def test_EC3_png_format(self):
        """EC3-有效等价类：PNG格式图像"""
        thermal_path = self.generate_test_image(format='png')
        ir_path = self.generate_test_image(format='png')
        
        success, result = self.process_image_pair(thermal_path, ir_path)
        self.assertTrue(success, "PNG格式图像处理应该成功")
        self.assertIsNotNone(result, "应该返回处理结果")
        
    def test_EC4_noisy_image(self):
        """EC4-有效等价类：带噪声图像"""
        thermal_path = self.generate_test_image(is_noise=True)
        ir_path = self.generate_test_image(is_noise=True)
        
        success, result = self.process_image_pair(thermal_path, ir_path)
        self.assertTrue(success, "带噪声图像处理应该成功")
        self.assertIsNotNone(result, "应该返回处理结果")
        
    # ===== 无效等价类测试 =====
    def test_EC5_size_mismatch(self):
        """EC5-无效等价类：尺寸不匹配"""
        thermal_path = self.generate_test_image(size=(640, 480))
        ir_path = self.generate_test_image(size=(800, 600))
        
        success, _ = self.process_image_pair(thermal_path, ir_path)
        self.assertFalse(success, "尺寸不匹配应该处理失败")
        
    def test_EC6_invalid_format(self):
        """EC6-无效等价类：不支持的图像格式"""
        # 创建GIF文件
        thermal_path = str(self.input_dir / "test_thermal.gif")
        ir_path = str(self.input_dir / "test_ir.gif")
        
        img = Image.new('RGB', (100, 100), color='red')
        img.save(thermal_path, 'GIF')
        img.save(ir_path, 'GIF')
        
        success, _ = self.process_image_pair(thermal_path, ir_path)
        self.assertFalse(success, "不支持的格式应该处理失败")
        
    def test_EC7_empty_file(self):
        """EC7-无效等价类：空文件"""
        thermal_path = str(self.input_dir / "empty_thermal.jpg")
        ir_path = str(self.input_dir / "empty_ir.jpg")
        
        # 创建空文件
        Path(thermal_path).touch()
        Path(ir_path).touch()
        
        success, _ = self.process_image_pair(thermal_path, ir_path)
        self.assertFalse(success, "空文件应该处理失败")
        
    def test_EC8_corrupted_file(self):
        """EC8-无效等价类：损坏的文件"""
        thermal_path = str(self.input_dir / "corrupted_thermal.jpg")
        ir_path = str(self.input_dir / "corrupted_ir.jpg")
        
        # 创建损坏的文件
        with open(thermal_path, 'wb') as f:
            f.write(b'corrupted data')
        with open(ir_path, 'wb') as f:
            f.write(b'corrupted data')
            
        success, _ = self.process_image_pair(thermal_path, ir_path)
        self.assertFalse(success, "损坏的文件应该处理失败")
        
    # ===== 边界值测试 =====
    def test_EC9_min_size(self):
        """EC9-边界值：最小允许尺寸(1x1)"""
        thermal_path = self.generate_test_image(size=(1, 1))
        ir_path = self.generate_test_image(size=(1, 1))
        
        success, _ = self.process_image_pair(thermal_path, ir_path)
        self.assertFalse(success, "最小尺寸应该处理失败")
        
    def test_EC10_max_size(self):
        """EC10-边界值：最大允许尺寸(8192x8192)"""
        thermal_path = self.generate_test_image(size=(8192, 8192))
        ir_path = self.generate_test_image(size=(8192, 8192))
        
        success, _ = self.process_image_pair(thermal_path, ir_path)
        self.assertFalse(success, "超大尺寸应该处理失败")
        
    def test_EC11_zero_byte(self):
        """EC11-边界值：0字节文件"""
        thermal_path = str(self.input_dir / "zero_thermal.jpg")
        ir_path = str(self.input_dir / "zero_ir.jpg")
        
        # 创建0字节文件
        Path(thermal_path).touch()
        Path(ir_path).touch()
        
        success, _ = self.process_image_pair(thermal_path, ir_path)
        self.assertFalse(success, "0字节文件应该处理失败")
        
    def test_EC12_large_aspect_ratio(self):
        """EC12-边界值：极端宽高比(16:1)"""
        thermal_path = self.generate_test_image(size=(1600, 100))
        ir_path = self.generate_test_image(size=(1600, 100))
        
        success, _ = self.process_image_pair(thermal_path, ir_path)
        self.assertFalse(success, "极端宽高比应该处理失败")

if __name__ == '__main__':
    unittest.main() 