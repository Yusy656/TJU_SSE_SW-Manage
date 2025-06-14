"""
图像处理系统性能测试脚本
使用Locust框架进行负载测试，模拟多用户并发访问场景
"""
from locust import HttpUser, task, between
import os
import random

class ImageProcessUser(HttpUser):
    wait_time = between(1, 3)
    # 热成像目录和红外（可见光）目录
    thermal_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../raw_data/image'))
    ir_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../raw_data/images'))
    # 只取两边都存在的同名图片编号
    thermal_files = set(f for f in os.listdir(thermal_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    ir_files = set(f for f in os.listdir(ir_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    common_files = list(thermal_files & ir_files)
    if len(common_files) < 1:
        raise RuntimeError('raw_data/image 和 raw_data/images 目录下没有同名图片，无法进行性能测试！')

    @task
    def full_pipeline(self):
        # 1. 创建会话
        resp = self.client.post("/create-session")
        if resp.status_code != 200:
            return
        session_id = resp.json().get("session_id")
        if not session_id:
            return

        # 随机选一个编号
        filename = random.choice(self.common_files)
        thermal_path = os.path.join(self.thermal_dir, filename)
        ir_path = os.path.join(self.ir_dir, filename)

        # 2. 上传红外（可见光）
        with open(ir_path, "rb") as f:
            resp = self.client.post("/upload", data={"type": "infrared", "session_id": session_id}, files={"file": f})
            if resp.status_code != 200:
                return

        # 3. 上传热成像
        with open(thermal_path, "rb") as f:
            resp = self.client.post("/upload", data={"type": "thermal", "session_id": session_id}, files={"file": f})
            if resp.status_code != 200:
                return

        # 4. 处理
        resp = self.client.post("/process", data={"session_id": session_id})
        if resp.status_code != 200:
            return

        # 5. 获取结果
        self.client.get(f"/get-processed-images?session_id={session_id}") 