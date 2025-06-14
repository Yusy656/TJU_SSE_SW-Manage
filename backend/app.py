from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from datetime import datetime
from dehazing.dehaze import start_dehaze
from fusion.fusion import inference_single_image
from detect.detect import main, detect_single_image
from config import Config, create_directories
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import time
import cv2
import psutil
import pynvml
import json
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:8080"],  # 你的前端开发服务器地址
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
app.config.from_object(Config)

# 配置日志
def setup_logger():
    # 创建logs目录（如果不存在）
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器（按大小轮转）
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    
    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 初始化日志记录器
logger = setup_logger()

# Create all required directories
create_directories()
logger.info("应用初始化完成，所有必要的目录已创建")

# 存储会话状态
session_states = {}

def create_session():
    session_id = str(uuid.uuid4())
    session_states[session_id] = {
        'created_at': datetime.now(),
        'thermal_image': None,
        'infrared_image': None,
        'status': 'initialized'
    }
    logger.info(f"创建新会话: {session_id}")
    return session_id

@app.route('/')
def index():
    logger.info("访问首页")
    return "Backend is running!"

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            logger.error("上传失败：请求中没有文件")
            return jsonify({'error': 'No file part in the request'}), 400
        if 'type' not in request.form:
            logger.error("上传失败：未提供图像类型")
            return jsonify({'error': 'No image type provided'}), 400
        if 'session_id' not in request.form:
            logger.error("上传失败：未提供会话ID")
            return jsonify({'error': 'No session ID provided'}), 400

        file = request.files['file']
        image_type = request.form['type']
        session_id = request.form['session_id']

        logger.info(f"开始上传图像 - 会话ID: {session_id}, 类型: {image_type}")

        if session_id not in session_states:
            logger.error(f"无效的会话ID: {session_id}")
            return jsonify({'error': 'Invalid session ID'}), 400

        if file.filename == '':
            logger.error("上传失败：未选择文件")
            return jsonify({'error': 'No selected file'}), 400

        allowed_types = ['infrared', 'thermal']
        if image_type not in allowed_types:
            logger.error(f"无效的图像类型: {image_type}")
            return jsonify({'error': f'Invalid image type: {image_type}. Allowed types are {allowed_types}'}), 400

        if file:
            upload_folder = app.config[f'{image_type.upper()}_UPLOAD_FOLDER']
            file_extension = os.path.splitext(file.filename)[1]
            new_filename = f'{session_id}_{image_type}{file_extension}'
            file_path = os.path.join(upload_folder, new_filename)
            file.save(file_path)
            
            # 更新会话状态
            session_states[session_id][f'{image_type}_image'] = new_filename
            
            logger.info(f"图像上传成功 - 文件名: {new_filename}")
            return jsonify({
                'message': f'{image_type} image uploaded successfully',
                'filename': new_filename,
                'type': image_type,
                'session_id': session_idme
            }), 200
    except Exception as e:
        logger.error(f"上传过程中发生错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/create-session', methods=['POST'])
def create_new_session():
    session_id = create_session()
    logger.info(f"创建新会话: {session_id}")
    return jsonify({
        'session_id': session_id,
        'message': 'Session created successfully'
    }), 200

@app.route('/process', methods=['POST'])
def start_process_pipeline():
    try:
        total_start_time = time.time()
        logger.info("开始处理流程")
        
        # 记录开始时的系统资源使用情况
        start_metrics = {
            'cpu': psutil.cpu_percent(interval=1),
            'memory': psutil.virtual_memory().percent,
            'gpu': []
        }
        
        # GPU监控初始化
        logger.info("初始化GPU监控")
        try:
            pynvml.nvmlInit()
            logger.info("NVIDIA驱动初始化成功")
            
            device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"检测到 {device_count} 个GPU设备")
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                try:
                    device_name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(device_name, bytes):
                        device_name = device_name.decode('utf-8')
                    elif isinstance(device_name, str):
                        device_name = device_name
                    else:
                        device_name = f"GPU-{i}"
                except:
                    device_name = f"GPU-{i}"
                
                logger.info(f"GPU {i}: {device_name}")
                
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                logger.info(f"GPU {i} 当前功耗: {power_usage:.2f}W")
                logger.info(f"GPU {i} 内存使用: {memory_info.used/1024/1024:.2f}MB / {memory_info.total/1024/1024:.2f}MB")
                
                start_metrics['gpu'].append({
                    'id': i,
                    'name': device_name,
                    'power_usage': power_usage,
                    'memory_used': memory_info.used,
                    'memory_total': memory_info.total
                })
        except pynvml.NVMLError as e:
            error_msg = f"NVIDIA驱动错误: {str(e)}"
            logger.error(error_msg)
            start_metrics['gpu'] = {'error': error_msg}
        except Exception as e:
            error_msg = f"GPU监控错误: {str(e)}"
            logger.error(error_msg)
            start_metrics['gpu'] = {'error': error_msg}

        session_id = request.form.get('session_id')
        if not session_id or session_id not in session_states:
            logger.error(f"无效的会话ID: {session_id}")
            return jsonify({'error': 'Invalid session ID'}), 400

        session = session_states[session_id]
        if not session['thermal_image'] or not session['infrared_image']:
            logger.error("缺少必要的图像文件")
            return jsonify({'error': 'Missing required images'}), 400

        # 获取输入文件路径
        thermal_path = os.path.join(app.config['THERMAL_UPLOAD_FOLDER'], session['thermal_image'])
        ir_path = os.path.join(app.config['INFRARED_UPLOAD_FOLDER'], session['infrared_image'])
        
        # 确保输入文件存在
        if not os.path.exists(thermal_path) or not os.path.exists(ir_path):
            logger.error("输入图像文件不存在")
            raise FileNotFoundError("输入图像文件不存在")
        
        # 更新会话状态
        session['status'] = 'processing'
        logger.info(f"会话 {session_id} 状态更新为处理中")
        
        # Dehazing
        logger.info("开始去雾处理")
        ir_img = cv2.imread(ir_path)
        if ir_img is None:
            logger.error("无法读取红外图像")
            raise FileNotFoundError("无法读取红外图像")
        logger.info(f"输入图像格式: {ir_img.shape[1]}x{ir_img.shape[0]} (宽x高), 通道数: {ir_img.shape[2]}")
        
        dehaze_start_time = time.time()
        dehazed_ir_path = os.path.join(app.config['DEHAZED_RESULTS_FOLDER'], f'{session_id}_dehazed_infrared.jpg')
        dehaze_metrics = start_dehaze(input_path=ir_path, output_path=dehazed_ir_path)
        if not dehaze_metrics:
            logger.error("去雾处理失败")
            raise Exception("去雾处理失败")
        dehaze_end_time = time.time()
        
        # 读取并打印去雾后图像格式
        dehazed_img = cv2.imread(dehazed_ir_path)
        if dehazed_img is None:
            logger.error("无法读取去雾后的图像")
            raise FileNotFoundError("无法读取去雾后的图像")
        logger.info(f"去雾后图像格式: {dehazed_img.shape[1]}x{dehazed_img.shape[0]} (宽x高), 通道数: {dehazed_img.shape[2]}")
        logger.info(f"去雾处理用时: {dehaze_end_time - dehaze_start_time:.4f} 秒")
        
        # Fusion
        logger.info("开始图像融合")
        thermal_img = cv2.imread(thermal_path)
        if thermal_img is None:
            logger.error("无法读取热成像")
            raise FileNotFoundError("无法读取热成像")
        logger.info(f"热成像格式: {thermal_img.shape[1]}x{thermal_img.shape[0]} (宽x高), 通道数: {thermal_img.shape[2]}")
        
        fusion_start_time = time.time()
        fused_path = os.path.join(app.config['FUSED_RESULTS_FOLDER'], f'{session_id}_fused_image.jpg')
        fusion_metrics = inference_single_image(
            thermal_path=thermal_path, 
            ir_path=dehazed_ir_path, 
            output_path=fused_path
        )
        if not fusion_metrics:
            logger.error("图像融合失败")
            raise Exception("图像融合失败")
        fusion_end_time = time.time()
        
        # 读取并打印融合后图像格式
        fused_img = cv2.imread(fused_path)
        if fused_img is None:
            logger.error("无法读取融合后的图像")
            raise FileNotFoundError("无法读取融合后的图像")
        logger.info(f"融合后图像格式: {fused_img.shape[1]}x{fused_img.shape[0]} (宽x高), 通道数: {fused_img.shape[2]}")
        logger.info(f"图像融合用时: {fusion_end_time - fusion_start_time:.4f} 秒")
        logger.info(f"去雾到融合的间隔时间: {fusion_start_time - dehaze_end_time:.4f} 秒")
            
        # Detect
        logger.info("开始目标检测")
        detect_start_time = time.time()
        detected_path = os.path.join(app.config['DETECTED_RESULTS_FOLDER'], f'{session_id}_detected_image.jpg')
        detection_metrics = detect_single_image(
            img_path=fused_path,
            output_path=detected_path
        )
        if not detection_metrics:
            logger.error("目标检测失败")
            raise Exception("目标检测失败")
        detect_end_time = time.time()
        
        # 读取并打印检测后图像格式
        detected_img = cv2.imread(detected_path)
        if detected_img is None:
            logger.error("无法读取检测后的图像")
            raise FileNotFoundError("无法读取检测后的图像")
        logger.info(f"检测后图像格式: {detected_img.shape[1]}x{detected_img.shape[0]} (宽x高), 通道数: {detected_img.shape[2]}")
        logger.info(f"目标检测用时: {detect_end_time - detect_start_time:.4f} 秒")
        logger.info(f"融合到检测的间隔时间: {detect_start_time - fusion_end_time:.4f} 秒")
        
        # 记录结束时的系统资源使用情况
        end_metrics = {
            'cpu': psutil.cpu_percent(interval=1),
            'memory': psutil.virtual_memory().percent,
            'gpu': []
        }
        
        # GPU监控结束状态
        logger.info("记录GPU结束状态")
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                try:
                    device_name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(device_name, bytes):
                        device_name = device_name.decode('utf-8')
                    elif isinstance(device_name, str):
                        device_name = device_name
                    else:
                        device_name = f"GPU-{i}"
                except:
                    device_name = f"GPU-{i}"
                
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                logger.info(f"GPU {i} 结束功耗: {power_usage:.2f}W")
                logger.info(f"GPU {i} 结束内存: {memory_info.used/1024/1024:.2f}MB / {memory_info.total/1024/1024:.2f}MB")
                
                end_metrics['gpu'].append({
                    'id': i,
                    'name': device_name,
                    'power_usage': power_usage,
                    'memory_used': memory_info.used,
                    'memory_total': memory_info.total
                })
        except pynvml.NVMLError as e:
            error_msg = f"NVIDIA驱动错误: {str(e)}"
            logger.error(error_msg)
            end_metrics['gpu'] = {'error': error_msg}
        except Exception as e:
            error_msg = f"GPU监控错误: {str(e)}"
            logger.error(error_msg)
            end_metrics['gpu'] = {'error': error_msg}
            
        # 计算资源使用差异
        resource_metrics = {
            'cpu_usage_change': end_metrics['cpu'] - start_metrics['cpu'],
            'memory_usage_change': end_metrics['memory'] - start_metrics['memory'],
            'gpu_metrics': []
        }
        
        # 计算GPU资源使用差异
        logger.info("计算GPU资源使用统计")
        if isinstance(start_metrics['gpu'], list) and isinstance(end_metrics['gpu'], list):
            for start_gpu, end_gpu in zip(start_metrics['gpu'], end_metrics['gpu']):
                gpu_metric = {
                    'gpu_id': start_gpu['id'],
                    'gpu_name': start_gpu['name'],
                    'power_usage_change': end_gpu['power_usage'] - start_gpu['power_usage'],
                    'memory_usage_change': end_gpu['memory_used'] - start_gpu['memory_used'],
                    'start_power': start_gpu['power_usage'],
                    'end_power': end_gpu['power_usage'],
                    'start_memory': start_gpu['memory_used'],
                    'end_memory': end_gpu['memory_used']
                }
                logger.info(f"GPU {start_gpu['id']} ({start_gpu['name']}) 功耗变化: {gpu_metric['power_usage_change']:.2f}W")
                logger.info(f"GPU {start_gpu['id']} ({start_gpu['name']}) 内存变化: {gpu_metric['memory_usage_change']/1024/1024:.2f}MB")
                resource_metrics['gpu_metrics'].append(gpu_metric)
        else:
            logger.warning("无法计算GPU资源使用差异")
            logger.warning(f"开始状态: {start_metrics['gpu']}")
            logger.warning(f"结束状态: {end_metrics['gpu']}")
        
        # 计算总用时
        total_end_time = time.time()
        logger.info(f"处理完成，总用时: {total_end_time - total_start_time:.4f} 秒")
        
        # 更新会话状态
        session['status'] = 'completed'
        session['results'] = {
            'dehazed_image': f'{session_id}_dehazed_infrared.jpg',
            'fused_image': f'{session_id}_fused_image.jpg',
            'detected_image': f'{session_id}_detected_image.jpg',
            'metrics': {
                'dehazing': dehaze_metrics,
                'fusion': fusion_metrics,
                'detection': detection_metrics,
                'resource_usage': resource_metrics
            }
        }
        
        logger.info(f"会话 {session_id} 处理完成")
        return jsonify({
            'message': 'Processing completed',
            'session_id': session_id,
            'metrics': {
                'dehazing': dehaze_metrics,
                'fusion': fusion_metrics,
                'detection': detection_metrics,
                'resource_usage': resource_metrics
            }
        }), 200
        
    except Exception as e:
        error_msg = f"处理过程中发生错误: {str(e)}"
        logger.error(error_msg)
        if session_id in session_states:
            session_states[session_id]['status'] = 'failed'
        return jsonify({
            'message': 'Processing failed',
            'error': str(e)
        }), 500

@app.route('/get-processed-images', methods=['GET'])
def get_processed_images():
    try:
        session_id = request.args.get('session_id')
        if not session_id or session_id not in session_states:
            return jsonify({'error': 'Invalid session ID'}), 400

        session = session_states[session_id]
        
        if session['status'] == 'processing':
            return jsonify({
                'dehazingImage': None,
                'fusingImage': None,
                'combinedImage': None,
                'status': 'processing',
                'metrics': None
            }), 200
            
        if session['status'] == 'failed':
            return jsonify({
                'status': 'failed',
                'error': 'Processing failed',
                'metrics': None
            }), 200
            
        if session['status'] == 'completed' and 'results' in session:
            return jsonify({
                'dehazingImage': f'/results/{session["results"]["dehazed_image"]}',
                'fusingImage': f'/results/{session["results"]["fused_image"]}',
                'combinedImage': f'/results/{session["results"]["detected_image"]}',
                'status': 'completed',
                'metrics': session['results']['metrics']
            }), 200
            
        return jsonify({
            'status': 'not_found',
            'message': 'No processed images found for this session',
            'metrics': None
        }), 404
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error',
            'metrics': None
        }), 500

# 添加静态文件服务路由
@app.route('/results/<path:filename>')
def serve_result_image(filename):
    try:
        # 根据文件名判断应该从哪个目录提供文件
        if 'detected' in filename:
            directory = app.config['DETECTED_RESULTS_FOLDER']
        elif 'fused' in filename:
            directory = app.config['FUSED_RESULTS_FOLDER']
        else:
            directory = app.config['DEHAZED_RESULTS_FOLDER']
            
        return send_from_directory(directory, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

def cleanup_expired_sessions():
    """清理过期的会话"""
    logger.info("开始清理过期会话")
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, session in session_states.items():
        # 如果会话超过1小时未完成，则清理
        if (current_time - session['created_at']).total_seconds() > 3600:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        logger.info(f"清理过期会话: {session_id}")
        # 清理相关文件
        session = session_states[session_id]
        for image_type in ['thermal', 'infrared']:
            if session.get(f'{image_type}_image'):
                file_path = os.path.join(
                    app.config[f'{image_type.upper()}_UPLOAD_FOLDER'],
                    session[f'{image_type}_image']
                )
                try:
                    os.remove(file_path)
                    logger.info(f"删除文件: {file_path}")
                except Exception as e:
                    logger.error(f"删除文件失败 {file_path}: {str(e)}")
        
        # 清理结果文件
        if 'results' in session:
            for result_type in ['dehazed', 'fused', 'detected']:
                file_path = os.path.join(
                    app.config[f'{result_type.upper()}_RESULTS_FOLDER'],
                    session['results'][f'{result_type}_image']
                )
                try:
                    os.remove(file_path)
                    logger.info(f"删除结果文件: {file_path}")
                except Exception as e:
                    logger.error(f"删除结果文件失败 {file_path}: {str(e)}")
        
        # 删除会话
        del session_states[session_id]
        logger.info(f"会话 {session_id} 已删除")

# 创建调度器
scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_expired_sessions, 'interval', hours=1)
scheduler.start()
logger.info("会话清理调度器已启动")

# 在应用关闭时停止调度器
atexit.register(lambda: scheduler.shutdown())
logger.info("应用已启动，监听端口 5000")

def check_session_status(session_id):
    """检查会话状态是否有效"""
    if session_id not in session_states:
        return False
    
    session = session_states[session_id]
    # 检查会话是否过期
    if (datetime.now() - session['created_at']).total_seconds() > 3600:
        cleanup_expired_sessions()
        return False
    
    return True

@app.route('/system-metrics', methods=['GET'])
def get_system_metrics():
    try:
        metrics = {
            'cpu': {
                'usage_percent': psutil.cpu_percent(interval=1),
                'count': psutil.cpu_count(),
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'used': psutil.virtual_memory().used,
                'percent': psutil.virtual_memory().percent
            },
            'gpu': []
        }
        
        # 尝试获取GPU信息
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_info = {
                    'id': i,
                    'name': pynvml.nvmlDeviceGetName(handle).decode('utf-8'),
                    'memory': {
                        'total': info.total,
                        'used': info.used,
                        'free': info.free,
                        'percent': (info.used / info.total) * 100
                    },
                    'temperature': pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                    'power_usage': pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                }
                metrics['gpu'].append(gpu_info)
        except Exception as e:
            metrics['gpu'] = {'error': str(e)}
            
        return jsonify(metrics), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 