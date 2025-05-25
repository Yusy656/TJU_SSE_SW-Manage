import unittest
import json
import os
from app import app, session_states
import tempfile
import shutil

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

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.test_dir)
        # 清理会话状态
        session_states.clear()

    def test_index(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), 'Backend is running!')

    def test_create_session(self):
        response = self.app.post('/create-session')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('session_id', data)
        self.assertIn('message', data)
        self.assertEqual(data['message'], 'Session created successfully')

    def test_upload_image_invalid_session(self):
        response = self.app.post('/upload', data={
            'file': (b'test', 'test.jpg'),
            'type': 'thermal',
            'session_id': 'invalid_session'
        })
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_upload_image_missing_file(self):
        # 先创建会话
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

    def test_process_pipeline_invalid_session(self):
        response = self.app.post('/process', data={
            'session_id': 'invalid_session'
        })
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_get_processed_images_invalid_session(self):
        response = self.app.get('/get-processed-images?session_id=invalid_session')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main() 