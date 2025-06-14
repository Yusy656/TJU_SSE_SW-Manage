import unittest
import requests
import os
import cv2
import numpy as np
import shutil
from pathlib import Path

class TestImageProcessingBoundary(unittest.TestCase):
    """使用边界值分析法进行黑盒测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.base_url = "http://localhost:5000"
        cls.test_dir = Path(__file__).parent / "test_data"
        # 如果目录已存在,先删除
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
    def setUp(self):
        """每个测试前创建新会话"""
        response = requests.post(f"{self.base_url}/create-session")
        self.assertTrue(response.ok)
        self.session_id = response.json()["session_id"]
        
    def generate_test_image(self, size, pixel_value=128):
        """生成测试图像"""
        try:
            img = np.full((*size, 3), pixel_value, dtype=np.uint8)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            path = str(self.test_dir / f"test_{size[0]}x{size[1]}_{pixel_value}.jpg")
            cv2.imwrite(path, img_gray)
            return path
        except Exception as e:
            self.fail(f"生成测试图像失败: {str(e)}")
        
    def process_image_pair(self, thermal_path, ir_path):
        """通过API处理图像对"""
        try:
            # 上传热成像图
            with open(thermal_path, 'rb') as f:
                response = requests.post(
                    f"{self.base_url}/upload",
                    files={'file': f},
                    data={'type': 'thermal', 'session_id': self.session_id}
                )
                if not response.ok:
                    return False, f"热成像上传失败: {response.text}"
                    
            # 上传红外图
            with open(ir_path, 'rb') as f:
                response = requests.post(
                    f"{self.base_url}/upload",
                    files={'file': f},
                    data={'type': 'infrared', 'session_id': self.session_id}
                )
                if not response.ok:
                    return False, f"红外图上传失败: {response.text}"
                    
            # 处理图像
            response = requests.post(
                f"{self.base_url}/process",
                data={'session_id': self.session_id}
            )
            if not response.ok:
                return False, f"处理失败: {response.text}"
                
            # 获取结果
            response = requests.get(
                f"{self.base_url}/get-processed-images",
                params={'session_id': self.session_id}
            )
            if not response.ok:
                return False, f"获取结果失败: {response.text}"
                
            result = response.json()
            return result["status"] == "completed", result.get("error", None)
            
        except Exception as e:
            return False, str(e)
            
    def test_BVA01_minimum_size(self):
        """BVA01: 最小尺寸边界值测试"""
        sizes = [(31, 31), (32, 32), (33, 33)]
        
        for size in sizes:
            with self.subTest(size=size):
                thermal_path = self.generate_test_image(size)
                ir_path = self.generate_test_image(size)
                
                success, error = self.process_image_pair(thermal_path, ir_path)
                
                if size[0] < 32:
                    self.assertFalse(success, f"尺寸{size}应该处理失败,但成功了。错误信息:{error}")
                else:
                    self.assertTrue(success, f"尺寸{size}应该处理成功,但失败了。错误信息:{error}")
                    
    def test_BVA02_maximum_size(self):
        """BVA02: 最大尺寸边界值测试"""
        sizes = [(4095, 4095), (4096, 4096), (4097, 4097)]
        
        for size in sizes:
            with self.subTest(size=size):
                thermal_path = self.generate_test_image(size)
                ir_path = self.generate_test_image(size)
                
                success, error = self.process_image_pair(thermal_path, ir_path)
                
                if size[0] > 4096:
                    self.assertFalse(success, f"尺寸{size}应该处理失败,但成功了。错误信息:{error}")
                else:
                    self.assertTrue(success, f"尺寸{size}应该处理成功,但失败了。错误信息:{error}")
                    
    def test_BVA03_pixel_values(self):
        """BVA03: 像素值边界值测试"""
        size = (64, 64)
        pixel_values = [-1, 0, 1, 127, 128, 129, 254, 255, 256]
        
        for value in pixel_values:
            with self.subTest(pixel_value=value):
                try:
                    thermal_path = self.generate_test_image(size, value)
                    ir_path = self.generate_test_image(size, value)
                    
                    success, error = self.process_image_pair(thermal_path, ir_path)
                    
                    if 0 <= value <= 255:
                        self.assertTrue(success, f"像素值{value}应该处理成功,但失败了。错误信息:{error}")
                    else:
                        self.assertFalse(success, f"像素值{value}应该处理失败,但成功了。错误信息:{error}")
                except Exception as e:
                    if 0 <= value <= 255:
                        self.fail(f"像素值{value}应该能正常保存和处理,但出错: {str(e)}")
                    
    def test_BVA04_aspect_ratio(self):
        """BVA04: 宽高比边界值测试"""
        base_size = 256
        ratios = [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16]
        
        for ratio in ratios:
            with self.subTest(ratio=ratio):
                if ratio >= 1:
                    size = (int(base_size * ratio), base_size)
                else:
                    size = (base_size, int(base_size / ratio))
                    
                thermal_path = self.generate_test_image(size)
                ir_path = self.generate_test_image(size)
                
                success, error = self.process_image_pair(thermal_path, ir_path)
                
                if 1/8 <= ratio <= 8:
                    self.assertTrue(success, f"宽高比{ratio}应该处理成功,但失败了。错误信息:{error}")
                else:
                    self.assertFalse(success, f"宽高比{ratio}应该处理失败,但成功了。错误信息:{error}")
                    
    def test_BVA05_image_quality(self):
        """BVA05: 图像质量边界值测试"""
        size = (64, 64)
        quality_levels = [0, 1, 50, 95, 100]
        
        for quality in quality_levels:
            with self.subTest(quality=quality):
                img = np.full((*size, 3), 128, dtype=np.uint8)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                thermal_path = str(self.test_dir / f"thermal_quality_{quality}.jpg")
                ir_path = str(self.test_dir / f"ir_quality_{quality}.jpg")
                
                cv2.imwrite(thermal_path, img_gray, [cv2.IMWRITE_JPEG_QUALITY, quality])
                cv2.imwrite(ir_path, img_gray, [cv2.IMWRITE_JPEG_QUALITY, quality])
                
                success, error = self.process_image_pair(thermal_path, ir_path)
                
                if quality > 0:
                    self.assertTrue(success, f"质量级别{quality}应该处理成功,但失败了。错误信息:{error}")
                else:
                    self.assertFalse(success, f"质量级别{quality}应该处理失败,但成功了。错误信息:{error}")
                    
    def test_BVA06_file_size(self):
        """BVA06: 文件大小边界值测试"""
        size = (64, 64)
        compression_levels = [0, 1, 5, 9]
        
        for level in compression_levels:
            with self.subTest(compression_level=level):
                img = np.full((*size, 3), 128, dtype=np.uint8)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                thermal_path = str(self.test_dir / f"thermal_compress_{level}.png")
                ir_path = str(self.test_dir / f"ir_compress_{level}.png")
                
                cv2.imwrite(thermal_path, img_gray, [cv2.IMWRITE_PNG_COMPRESSION, level])
                cv2.imwrite(ir_path, img_gray, [cv2.IMWRITE_PNG_COMPRESSION, level])
                
                success, error = self.process_image_pair(thermal_path, ir_path)
                self.assertTrue(success, f"压缩级别{level}应该处理成功,但失败了。错误信息:{error}")
                
    def test_BVA07_empty_file(self):
        """BVA07: 空文件测试"""
        thermal_path = str(self.test_dir / "thermal_empty.jpg")
        ir_path = str(self.test_dir / "ir_empty.jpg")
        
        # 创建空文件
        Path(thermal_path).touch()
        Path(ir_path).touch()
        
        success, error = self.process_image_pair(thermal_path, ir_path)
        self.assertFalse(success, f"空文件应该处理失败,但成功了。错误信息:{error}")
        
    def tearDown(self):
        """清理测试数据"""
        # 删除所有测试文件
        for file in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"清理文件失败: {file_path}, 错误: {str(e)}")
                
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        try:
            shutil.rmtree(cls.test_dir)
        except Exception as e:
            print(f"清理测试目录失败: {str(e)}")

if __name__ == '__main__':
    unittest.main() 