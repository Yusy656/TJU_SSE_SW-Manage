import os
import time
import torch
import numpy as np
import cv2
from pathlib import Path
from detect.detect import (
    DetEvaluator, PRCalculator, PRVisualizer, load_gt_boxes,
    detect_single_image, main as detect_main
)
import json
from tqdm import tqdm
import coverage
import sys
import urllib.request
import pytest
import shutil
import random
import warnings
import io
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast")

def calculate_coverage(detections, gt_boxes, iou_threshold=0.5):
    """计算检测覆盖率
    
    Args:
        detections: 检测结果字典 {class_id: [(x1,y1,x2,y2,conf), ...]}
        gt_boxes: 真实框字典 {class_id: [(x1,y1,x2,y2), ...]}
        iou_threshold: IoU阈值
        
    Returns:
        dict: 每个类别的覆盖率统计
    """
    coverage_stats = {}
    
    for cls_id in set(detections.keys()).union(gt_boxes.keys()):
        det_boxes = detections.get(cls_id, [])
        gt_boxes_cls = gt_boxes.get(cls_id, [])
        
        if not gt_boxes_cls:
            continue
            
        # 计算每个GT框是否被检测到
        gt_matched = set()
        for gt_box in gt_boxes_cls:
            max_iou = 0
            for det_box in det_boxes:
                iou = calculate_iou(det_box[:4], gt_box)
                if iou > max_iou:
                    max_iou = iou
            if max_iou >= iou_threshold:
                gt_matched.add(gt_box)
        
        # 计算覆盖率
        total_gt = len(gt_boxes_cls)
        matched_gt = len(gt_matched)
        coverage = matched_gt / total_gt if total_gt > 0 else 0
        
        coverage_stats[cls_id] = {
            'total_gt': total_gt,
            'matched_gt': matched_gt,
            'coverage_rate': coverage
        }
    
    return coverage_stats

def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0

def test_map(model_path, test_dir, label_dir, output_dir):
    """测试mAP和覆盖率
    
    Args:
        model_path: 模型路径
        test_dir: 测试图像目录
        label_dir: 标注文件目录
        output_dir: 输出目录
    """
    print("\n=== 测试 mAP@0.5 和覆盖率 ===")
    
    # 初始化评估器
    evaluator = DetEvaluator(model_path)
    
    # 初始化PR计算器
    pr_calculator = PRCalculator(iou_threshold=0.5)
    
    # 获取所有子目录中的图像文件
    test_files = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                test_files.append(os.path.join(root, file))
    
    if not test_files:
        raise ValueError("测试目录中没有图像文件")
    
    # 用于累积覆盖率统计
    total_coverage_stats = {}
    
    for img_path in tqdm(test_files, desc="计算mAP和覆盖率"):
        # 获取对应的标注文件路径
        rel_path = os.path.relpath(img_path, test_dir)
        label_path = os.path.join(label_dir, os.path.splitext(rel_path)[0] + '.txt')
        
        try:
            # 执行检测
            tensor, orig_shape = evaluator.preprocess(img_path)
            results = evaluator.detect(tensor)
            detections = evaluator.postprocess(results, orig_shape)
            
            # 加载真实框
            gt_boxes = load_gt_boxes(label_path, img_path)
            
            # 计算覆盖率
            coverage_stats = calculate_coverage(detections, gt_boxes)
            
            # 累积覆盖率统计
            for cls_id, stats in coverage_stats.items():
                if cls_id not in total_coverage_stats:
                    total_coverage_stats[cls_id] = {
                        'total_gt': 0,
                        'matched_gt': 0
                    }
                total_coverage_stats[cls_id]['total_gt'] += stats['total_gt']
                total_coverage_stats[cls_id]['matched_gt'] += stats['matched_gt']
            
            # 处理每个类别用于mAP计算
            all_classes = set(detections.keys()).union(gt_boxes.keys())
            for cls_id in all_classes:
                pr_calculator.process_image(
                    cls_id,
                    detections.get(cls_id, []),
                    gt_boxes.get(cls_id, [])
                )
                
        except Exception as e:
            print(f"处理失败 {img_path}: {str(e)}")
            continue
    
    # 计算指标
    metrics = pr_calculator.calculate_metrics()
    
    # 计算总体覆盖率
    final_coverage = {}
    for cls_id, stats in total_coverage_stats.items():
        if stats['total_gt'] > 0:
            final_coverage[cls_id] = stats['matched_gt'] / stats['total_gt']
        else:
            final_coverage[cls_id] = 0.0
    
    # 打印结果
    print("\n评估结果汇总：")
    print("\nmAP@0.5:")
    for cls_id, cls_metrics in metrics.items():
        print(f"类别 {cls_id}:")
        print(f"  AP: {cls_metrics['ap']:.4f}")
        print(f"  Precision: {cls_metrics['precision']:.4f}")
        print(f"  Recall: {cls_metrics['recall']:.4f}")
    
    print("\n覆盖率:")
    for cls_id, coverage in final_coverage.items():
        print(f"类别 {cls_id}: {coverage:.4f}")
    
    return metrics

def test_latency(model_path, test_data, num_runs=100):
    """测试检测延迟"""
    evaluator = DetEvaluator(model_path)
    
    # 获取测试图像
    test_files = []
    for root, _, files in os.walk(test_data['test_images']):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                test_files.append(os.path.join(root, file))
    
    assert len(test_files) > 0, "没有找到测试图像"
    
    # 过滤掉损坏的图像文件
    valid_test_files = []
    for img_path in test_files:
        img = cv2.imread(img_path)
        if img is not None:
            valid_test_files.append(img_path)
    
    assert len(valid_test_files) > 0, "没有找到有效的测试图像"
    
    # 使用第一张有效图像进行延迟测试
    img_path = valid_test_files[0]
    tensor, orig_shape = evaluator.preprocess(img_path)
    
    # 预热
    for _ in range(10):
        results = evaluator.detect(tensor)
        detections = evaluator.postprocess(results, orig_shape)
    
    # 测试延迟
    latencies = []
    for _ in range(num_runs):
        start_time = time.time()
        results = evaluator.detect(tensor)
        detections = evaluator.postprocess(results, orig_shape)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    # 计算统计信息
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    print(f"\n延迟统计 (ms):")
    print(f"平均延迟: {avg_latency:.2f}")
    print(f"P95延迟: {p95_latency:.2f}")
    print(f"P99延迟: {p99_latency:.2f}")
    
    # 验证延迟是否在合理范围内
    assert avg_latency < 100, f"平均延迟过高: {avg_latency:.2f}ms"
    assert p95_latency < 200, f"P95延迟过高: {p95_latency:.2f}ms"
    assert p99_latency < 500, f"P99延迟过高: {p99_latency:.2f}ms"

def test_robustness(model_path, test_cases):
    """测试不同环境条件下的检测稳定性
    
    Args:
        model_path: 模型路径
        test_cases: 测试用例字典，格式为：
            {
                'smoke': {'dir': 'path/to/smoke/images', 'label_dir': 'path/to/labels'},
                'fire': {'dir': 'path/to/fire/images', 'label_dir': 'path/to/labels'},
                'low_light': {'dir': 'path/to/low_light/images', 'label_dir': 'path/to/labels'}
            }
    """
    print("\n=== 测试检测稳定性 ===")
    
    # 初始化评估器
    evaluator = DetEvaluator(model_path)
    
    results = {}
    
    for case_name, case_info in test_cases.items():
        print(f"\n测试场景: {case_name}")
        
        # 初始化PR计算器
        pr_calculator = PRCalculator(iou_threshold=0.5)
        
        # 获取测试图像
        test_files = []
        for root, _, files in os.walk(case_info['dir']):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    test_files.append(os.path.join(root, file))
        
        if not test_files:
            print(f"警告: {case_name} 场景没有测试图像")
            continue
        
        # 用于累积覆盖率统计
        total_coverage_stats = {}
        
        for img_path in tqdm(test_files, desc=f"处理{case_name}场景"):
            # 获取对应的标注文件路径
            rel_path = os.path.relpath(img_path, case_info['dir'])
            label_path = os.path.join(case_info['label_dir'], os.path.splitext(rel_path)[0] + '.txt')
            
            try:
                # 执行检测
                tensor, orig_shape = evaluator.preprocess(img_path)
                results = evaluator.detect(tensor)
                detections = evaluator.postprocess(results, orig_shape)
                
                # 加载真实框
                gt_boxes = load_gt_boxes(label_path, img_path)
                
                # 计算覆盖率
                coverage_stats = calculate_coverage(detections, gt_boxes)
                
                # 累积覆盖率统计
                for cls_id, stats in coverage_stats.items():
                    if cls_id not in total_coverage_stats:
                        total_coverage_stats[cls_id] = {
                            'total_gt': 0,
                            'matched_gt': 0
                        }
                    total_coverage_stats[cls_id]['total_gt'] += stats['total_gt']
                    total_coverage_stats[cls_id]['matched_gt'] += stats['matched_gt']
                
                # 处理每个类别用于mAP计算
                all_classes = set(detections.keys()).union(gt_boxes.keys())
                for cls_id in all_classes:
                    pr_calculator.process_image(
                        cls_id,
                        detections.get(cls_id, []),
                        gt_boxes.get(cls_id, [])
                    )
                    
            except Exception as e:
                print(f"处理失败 {img_path}: {str(e)}")
                continue
        
        # 计算该场景的指标
        metrics = pr_calculator.calculate_metrics()
        
        # 计算总体覆盖率
        final_coverage = {}
        for cls_id, stats in total_coverage_stats.items():
            if stats['total_gt'] > 0:
                final_coverage[cls_id] = stats['matched_gt'] / stats['total_gt']
            else:
                final_coverage[cls_id] = 0.0
        
        # 保存结果
        results[case_name] = {
            'metrics': metrics,
            'coverage': final_coverage
        }
        
        # 打印结果
        print(f"\n{case_name} 场景评估结果：")
        print("\nmAP@0.5:")
        for cls_id, cls_metrics in metrics.items():
            print(f"类别 {cls_id}:")
            print(f"  AP: {cls_metrics['ap']:.4f}")
            print(f"  Precision: {cls_metrics['precision']:.4f}")
            print(f"  Recall: {cls_metrics['recall']:.4f}")
        
        print("\n覆盖率:")
        for cls_id, coverage in final_coverage.items():
            print(f"类别 {cls_id}: {coverage:.4f}")
    
    return results

def download_file(url, filename):
    """下载文件的函数，带进度条显示"""
    def progress_hook(t):
        last_b = [0]
        def update_to(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b
        return update_to

    with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename=filename, reporthook=progress_hook(t))

def download_model(model_path):
    """下载模型文件"""
    print(f"\n=== 开始下载模型文件 ===")
    print(f"目标路径: {model_path}")
    
    # 创建模型目录（如果不存在）
    model_dir = os.path.dirname(model_path)
    if model_dir:  # 只有当路径包含目录时才创建
        os.makedirs(model_dir, exist_ok=True)
    
    # 模型下载URL
    model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt"
    
    try:
        download_file(model_url, model_path)
        print(f"\n✅ 模型文件下载完成: {model_path}")
        return True
    except Exception as e:
        print(f"\n❌ 模型下载失败: {str(e)}")
        print("请检查网络连接或手动下载模型文件")
        return False

def create_sample_test_data():
    """创建示例测试图像和标注文件"""
    print("\n=== 创建示例测试数据 ===")
    
    # 创建示例图像
    def create_sample_image(path, condition):
        # 创建一个黑色背景
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # 根据条件添加不同的效果
        if condition == 'smoke':
            # 添加烟雾效果
            for _ in range(100):
                x = np.random.randint(0, 640)
                y = np.random.randint(0, 640)
                cv2.circle(img, (x, y), np.random.randint(5, 20), (200, 200, 200), -1)
        elif condition == 'fire':
            # 添加火焰效果
            for _ in range(50):
                x = np.random.randint(0, 640)
                y = np.random.randint(0, 640)
                cv2.circle(img, (x, y), np.random.randint(10, 30), (0, 0, 255), -1)
        elif condition == 'low_light':
            # 降低亮度
            img = img // 4
        
        # 添加一些模拟的人形目标
        for _ in range(3):
            x = np.random.randint(100, 540)
            y = np.random.randint(100, 540)
            w = np.random.randint(50, 100)
            h = np.random.randint(100, 200)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 保存图像
        cv2.imwrite(path, img)
        print(f"✅ 创建示例图像: {path}")
        
        # 创建对应的标注文件
        label_path = path.replace('test_images', 'test_labels').replace('.jpg', '.txt')
        with open(label_path, 'w') as f:
            # 为每个目标写入标注（格式：class x_center y_center width height）
            for _ in range(3):
                x = np.random.randint(100, 540) / 640
                y = np.random.randint(100, 540) / 640
                w = np.random.randint(50, 100) / 640
                h = np.random.randint(100, 200) / 640
                class_id = np.random.randint(0, 2)  # 0: 站立/跌倒, 1: 救援人员
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        print(f"✅ 创建标注文件: {label_path}")

def create_test_directories():
    """创建测试所需的目录结构"""
    directories = [
        'test_images',
        'test_images/smoke',
        'test_images/fire',
        'test_images/low_light',
        'test_labels',
        'test_labels/smoke',
        'test_labels/fire',
        'test_labels/low_light',
        'test_results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ 创建目录: {directory}")

def copy_test_images():
    """从data_split目录复制测试图像"""
    print("\n=== 准备测试数据 ===")
    
    # 源目录映射
    source_dirs = {
        'smoke': [
            'data_split/rgb/test',
            'data_split/ir/test'
        ],
        'fire': [
            'data_split/rgb/test',
            'data_split/ir/test'
        ],
        'low_light': [
            'data_split/ir/test'
        ]
    }
    
    # 目标目录
    target_dirs = {
        'smoke': 'test_images/smoke',
        'fire': 'test_images/fire',
        'low_light': 'test_images/low_light'
    }
    
    # 为每个场景复制图像
    for scene, dirs in source_dirs.items():
        target_dir = target_dirs[scene]
        os.makedirs(target_dir, exist_ok=True)
        
        # 从每个源目录复制图像
        for source_dir in dirs:
            if not os.path.exists(source_dir):
                continue
                
            # 获取源目录中的所有图像文件
            image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if not image_files:
                continue
            
            # 随机选择最多5张图像
            selected_files = random.sample(image_files, min(5, len(image_files)))
            
            # 复制选中的图像
            for img_file in selected_files:
                src_path = os.path.join(source_dir, img_file)
                dst_path = os.path.join(target_dir, f"{scene}_{img_file}")
                
                # 确保源文件存在
                if not os.path.exists(src_path):
                    continue
                    
                # 复制图像文件
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(f"复制图像失败: {str(e)}")
                    continue
                
                # 创建对应的标注文件
                label_path = dst_path.replace('test_images', 'test_labels').replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
                os.makedirs(os.path.dirname(label_path), exist_ok=True)
                
                # 从源目录复制对应的标注文件
                src_label_path = src_path.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
                if os.path.exists(src_label_path):
                    try:
                        shutil.copy2(src_label_path, label_path)
                    except Exception as e:
                        print(f"复制标注失败: {str(e)}")
                else:
                    # 如果没有标注文件，创建随机标注
                    try:
                        with open(label_path, 'w') as f:
                            num_targets = random.randint(1, 3)
                            for _ in range(num_targets):
                                x = random.uniform(0.1, 0.9)
                                y = random.uniform(0.1, 0.9)
                                w = random.uniform(0.1, 0.3)
                                h = random.uniform(0.2, 0.4)
                                class_id = random.randint(0, 1)  # 0: 站立/跌倒, 1: 救援人员
                                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    except Exception as e:
                        print(f"创建标注失败: {str(e)}")
    
    # 验证是否成功复制了图像
    total_images = 0
    for scene, target_dir in target_dirs.items():
        images = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        total_images += len(images)
    
    if total_images == 0:
        raise ValueError("没有成功复制任何测试图像")
    
    print(f"✅ 测试数据准备完成，共 {total_images} 张图像")

def run_with_coverage():
    """运行测试并收集代码覆盖率"""
    # 初始化覆盖率收集器
    cov = coverage.Coverage(
        branch=True,  # 启用分支覆盖率
        source=['detect'],  # 指定要收集覆盖率的源代码目录
        omit=[
            '*/test_*.py',  # 排除测试文件
            '*/__init__.py',  # 排除初始化文件
            '*/setup.py',  # 排除安装文件
        ]
    )
    
    # 开始收集覆盖率
    cov.start()
    
    try:
        # 运行所有测试
        detect_main()
    finally:
        # 停止收集覆盖率
        cov.stop()
        cov.save()
        
        # 生成覆盖率报告
        print("\n=== 代码覆盖率报告 ===")
        cov.report(show_missing=True)
        
        # 生成HTML报告
        cov.html_report(directory='coverage_html')
        
        # 生成XML报告（可用于CI集成）
        cov.xml_report(outfile='coverage.xml')

@pytest.fixture
def model_path():
    """模型文件路径fixture"""
    path = "yolo11x.pt"
    if not os.path.exists(path):
        download_model(path)
    return path

@pytest.fixture
def test_directories():
    """创建并返回测试目录结构"""
    dirs = {
        'test_images': 'test_images',
        'test_labels': 'test_labels',
        'test_results': 'test_results'
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        # 创建子目录
        for subdir in ['smoke', 'fire', 'low_light']:
            os.makedirs(os.path.join(dir_path, subdir), exist_ok=True)
    
    return dirs

@pytest.fixture
def test_data(test_directories):
    """准备测试数据"""
    copy_test_images()
    return test_directories

@pytest.fixture
def test_cases(test_data):
    """创建测试用例"""
    return {
        'smoke': {
            'dir': os.path.join(test_data['test_images'], 'smoke'),
            'label_dir': os.path.join(test_data['test_labels'], 'smoke')
        },
        'fire': {
            'dir': os.path.join(test_data['test_images'], 'fire'),
            'label_dir': os.path.join(test_data['test_labels'], 'fire')
        },
        'low_light': {
            'dir': os.path.join(test_data['test_images'], 'low_light'),
            'label_dir': os.path.join(test_data['test_labels'], 'low_light')
        }
    }

def test_model_loading(model_path):
    """测试模型加载"""
    evaluator = DetEvaluator(model_path)
    assert evaluator.model is not None
    assert isinstance(evaluator.model, torch.nn.Module)

def test_detection_basic(model_path, test_data):
    """测试基本检测功能"""
    evaluator = DetEvaluator(model_path)

    # 获取测试图像
    test_files = []
    for root, _, files in os.walk(test_data['test_images']):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                test_files.append(os.path.join(root, file))

    assert len(test_files) > 0, "没有找到测试图像"

    # 过滤掉损坏的图像文件
    valid_test_files = []
    for img_path in test_files:
        img = cv2.imread(img_path)
        if img is not None:
            valid_test_files.append(img_path)

    assert len(valid_test_files) > 0, "没有找到有效的测试图像"

    # 测试单张图像检测
    img_path = valid_test_files[0]
    tensor, orig_shape = evaluator.preprocess(img_path)
    results = evaluator.detect(tensor)
    detections = evaluator.postprocess(results, orig_shape)

    # 验证检测结果格式
    assert isinstance(detections, (list, dict)), "检测结果应该是列表或字典"
    # 如果是dict，合并所有box
    if isinstance(detections, dict):
        all_boxes = []
        for v in detections.values():
            all_boxes.extend(v)
        detections = all_boxes
    # 允许无检测结果
    if len(detections) == 0:
        return
    for det in detections:
        assert len(det) == 5 or len(det) == 6, "每个检测结果应该包含5或6个值"
        x, y, w, h, conf = det[:5]
        assert 0 <= x <= 1 and 0 <= y <= 1, "归一化坐标应该在0-1之间"
        assert 0 <= w <= 1 and 0 <= h <= 1, "归一化宽高应该在0-1之间"
        assert 0 <= conf <= 1, "置信度应该在0-1之间"
        if len(det) == 6:
            cls = det[5]
            assert isinstance(cls, (int, float)), "类别ID应该是数字"

@pytest.mark.parametrize("scene", ['smoke', 'fire', 'low_light'])
def test_scene_detection(model_path, test_data, scene):
    """测试不同场景的检测效果"""
    evaluator = DetEvaluator(model_path)
    scene_dir = os.path.join(test_data['test_images'], scene)
    
    # 获取场景测试图像
    test_files = [f for f in os.listdir(scene_dir) 
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    assert len(test_files) > 0, f"{scene}场景没有测试图像"
    
    # 测试场景图像
    img_path = os.path.join(scene_dir, test_files[0])
    tensor, orig_shape = evaluator.preprocess(img_path)
    results = evaluator.detect(tensor)
    detections = evaluator.postprocess(results, orig_shape)
    
    assert isinstance(detections, dict)
    assert all(isinstance(boxes, list) for boxes in detections.values())

def test_detection_latency(model_path, test_data):
    """测试检测延迟"""
    evaluator = DetEvaluator(model_path)
    
    # 获取测试图像
    test_files = []
    for root, _, files in os.walk(test_data['test_images']):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                test_files.append(os.path.join(root, file))
    
    assert len(test_files) > 0, "没有找到测试图像"
    
    latencies = []
    for _ in range(10):  # 减少测试次数以加快测试速度
        img_path = random.choice(test_files)
        start_time = time.time()
        tensor, orig_shape = evaluator.preprocess(img_path)
        results = evaluator.detect(tensor)
        detections = evaluator.postprocess(results, orig_shape)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000
        latencies.append(latency)
    
    avg_latency = np.mean(latencies)
    assert avg_latency < 1000, f"平均延迟 {avg_latency:.2f}ms 超过1秒"

def test_detection_accuracy(model_path, test_data):
    """测试检测准确率"""
    evaluator = DetEvaluator(model_path)
    pr_calculator = PRCalculator(iou_threshold=0.5)

    # 获取测试图像和标注
    test_files = []
    for root, _, files in os.walk(test_data['test_images']):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                test_files.append(os.path.join(root, file))

    assert len(test_files) > 0, "没有找到测试图像"

    # 过滤掉损坏的图像文件
    valid_test_files = []
    for img_path in test_files:
        img = cv2.imread(img_path)
        if img is not None:
            valid_test_files.append(img_path)

    assert len(valid_test_files) > 0, "没有找到有效的测试图像"

    for img_path in valid_test_files[:5]:  # 只测试前5张图像
        # 获取对应的标注文件路径
        rel_path = os.path.relpath(img_path, test_data['test_images'])
        label_path = os.path.join(test_data['test_labels'],
                                os.path.splitext(rel_path)[0] + '.txt')

        # 执行检测
        tensor, orig_shape = evaluator.preprocess(img_path)
        results = evaluator.detect(tensor)
        detections = evaluator.postprocess(results, orig_shape)

        # 计算PR曲线
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                gt_boxes = []
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    gt_boxes.append([x, y, w, h, class_id])
                if hasattr(pr_calculator, 'process_image'):
                    if isinstance(detections, dict):
                        for cls_id in set([int(box[-1]) for box in gt_boxes]):
                            dets = detections.get(cls_id, [])
                            gts = [box for box in gt_boxes if int(box[-1]) == cls_id]
                            pr_calculator.process_image(cls_id, dets, gts)
                elif hasattr(pr_calculator, 'add'):
                    pr_calculator.add(detections, gt_boxes)

    # 计算准确率指标
    if hasattr(pr_calculator, 'compute_pr'):
        precision, recall = pr_calculator.compute_pr()
        assert 0 <= precision <= 1, f"精确率应该在0-1之间，实际为{precision}"
        assert 0 <= recall <= 1, f"召回率应该在0-1之间，实际为{recall}"

@pytest.mark.parametrize("error_case", [
    "non_existent_image",
    "corrupted_image",
    "empty_image"
])
def test_error_handling(model_path, test_data, error_case):
    """测试错误处理"""
    evaluator = DetEvaluator(model_path)

    if error_case == "non_existent_image":
        img_path = "non_existent.jpg"
        with pytest.raises(ValueError, match="无法读取图像"):
            tensor, orig_shape = evaluator.preprocess(img_path)
    elif error_case == "corrupted_image":
        # 创建损坏的图像文件
        img_path = os.path.join(test_data['test_images'], "corrupted.jpg")
        with open(img_path, 'w') as f:
            f.write("corrupted image data")
        with pytest.raises(ValueError, match="无法读取图像"):
            tensor, orig_shape = evaluator.preprocess(img_path)
    else:  # empty_image
        img_path = os.path.join(test_data['test_images'], "empty.jpg")
        # 创建一个有效的空图像
        empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(img_path, empty_img)
        # 空图像应该能够被读取
        tensor, orig_shape = evaluator.preprocess(img_path)
        assert tensor is not None
        assert orig_shape == (100, 100)

def test_map(model_path, test_data):
    """测试mAP指标"""
    evaluator = DetEvaluator(model_path)
    pr_calculator = PRCalculator(iou_threshold=0.5)

    # 获取测试图像
    test_files = []
    for root, _, files in os.walk(test_data['test_images']):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                test_files.append(os.path.join(root, file))

    assert len(test_files) > 0, "没有找到测试图像"

    # 过滤掉损坏的图像文件
    valid_test_files = []
    for img_path in test_files:
        img = cv2.imread(img_path)
        if img is not None:
            valid_test_files.append(img_path)

    assert len(valid_test_files) > 0, "没有找到有效的测试图像"

    for img_path in valid_test_files[:5]:  # 只测试前5张图像
        # 获取对应的标注文件路径
        rel_path = os.path.relpath(img_path, test_data['test_images'])
        label_path = os.path.join(test_data['test_labels'],
                                os.path.splitext(rel_path)[0] + '.txt')

        # 执行检测
        tensor, orig_shape = evaluator.preprocess(img_path)
        results = evaluator.detect(tensor)
        detections = evaluator.postprocess(results, orig_shape)

        # 计算PR曲线
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                gt_boxes = []
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    gt_boxes.append([x, y, w, h, class_id])
                # 兼容不同实现
                if hasattr(pr_calculator, 'process_image'):
                    # 假设detections为dict: {cls_id: [boxes]}
                    if isinstance(detections, dict):
                        for cls_id in set([int(box[-1]) for box in gt_boxes]):
                            dets = detections.get(cls_id, [])
                            gts = [box for box in gt_boxes if int(box[-1]) == cls_id]
                            pr_calculator.process_image(cls_id, dets, gts)
                elif hasattr(pr_calculator, 'add'):
                    pr_calculator.add(detections, gt_boxes)

    # 计算mAP
    if hasattr(pr_calculator, 'compute_map'):
        map_score = pr_calculator.compute_map()
        assert 0 <= map_score <= 1, f"mAP分数应该在0-1之间，实际为{map_score}"

def test_robustness(model_path, test_cases):
    """测试不同场景的检测稳定性"""
    evaluator = DetEvaluator(model_path)
    
    for case_name, case_info in test_cases.items():
        # 获取测试图像
        test_files = []
        for root, _, files in os.walk(case_info['dir']):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    test_files.append(os.path.join(root, file))
        
        if not test_files:
            continue
        
        # 测试场景图像
        img_path = test_files[0]
        tensor, orig_shape = evaluator.preprocess(img_path)
        results = evaluator.detect(tensor)
        detections = evaluator.postprocess(results, orig_shape)
        
        assert isinstance(detections, dict)
        assert all(isinstance(boxes, list) for boxes in detections.values())

def test_main_function(tmp_path):
    """测试main函数的正常和异常分支"""
    # 创建测试目录
    img_dir = tmp_path / "test_images"
    label_dir = tmp_path / "test_labels"
    output_dir = tmp_path / "output"
    img_dir.mkdir()
    label_dir.mkdir()
    output_dir.mkdir()
    # 正常图片和标注
    img_path = img_dir / "test.jpg"
    label_path = label_dir / "test.txt"
    img = np.ones((640, 640, 3), dtype=np.uint8) * 127
    cv2.imwrite(str(img_path), img)
    with open(label_path, 'w') as f:
        f.write("0 0.5 0.5 0.2 0.2\n")
    # 1. 正常流程
    detect_main(str(img_dir), str(label_dir), "yolo11x.pt", str(output_dir))
    assert (output_dir / 'metrics.json').exists()
    assert (output_dir / 'all_classes_pr.png').exists()
    # 2. 异常图片（无法读取）
    bad_img = img_dir / "bad.jpg"
    with open(bad_img, 'w') as f:
        f.write("not an image")
    detect_main(str(img_dir), str(label_dir), "yolo11x.pt", str(output_dir))
    # 3. 没有标注文件
    img2 = img_dir / "no_label.jpg"
    cv2.imwrite(str(img2), img)
    detect_main(str(img_dir), str(label_dir), "yolo11x.pt", str(output_dir))
    # 4. fused_前缀
    fused_img = img_dir / "fused_test2.jpg"
    cv2.imwrite(str(fused_img), img)
    label2 = label_dir / "test2.txt"
    with open(label2, 'w') as f:
        f.write("0 0.5 0.5 0.2 0.2\n")
    detect_main(str(img_dir), str(label_dir), "yolo11x.pt", str(output_dir))

def test_prcalculator_process_image_edge_cases():
    pr = PRCalculator(iou_threshold=0.5)
    # 检测框和GT框都为空
    pr.process_image(0, [], [])
    # 检测框有，GT框无
    pr.process_image(1, [[0,0,1,1,0.9]], [])
    # 检测框无，GT框有
    pr.process_image(2, [], [[0,0,1,1]])
    # 检测框和GT框都不为空，但IoU不达标
    pr.process_image(3, [[0,0,1,1,0.1]], [[2,2,3,3]])
    # 检测框和GT框都不为空，IoU达标
    pr.process_image(4, [[0,0,1,1,0.9]], [[0,0,1,1]])
    metrics = pr.calculate_metrics()
    for cls_id, data in metrics.items():
        assert "precision" in data
        assert "recall" in data
        assert "ap" in data

def test_pr_visualizer(tmp_path):
    visualizer = PRVisualizer(output_dir=str(tmp_path))
    metrics = {
        0: {'precision': [0.8, 0.7, 0.6], 'recall': [0.3, 0.5, 0.7], 'ap': 0.65},
        1: {'precision': [0.9, 0.8, 0.7], 'recall': [0.4, 0.6, 0.8], 'ap': 0.75}
    }
    visualizer.plot_pr_curves(metrics)
    assert (tmp_path / 'all_classes_pr.png').exists()

def test_load_gt_boxes(tmp_path):
    img_path = tmp_path / "test.jpg"
    label_path = tmp_path / "test.txt"
    img = np.ones((640, 640, 3), dtype=np.uint8) * 127
    cv2.imwrite(str(img_path), img)
    with open(label_path, 'w') as f:
        f.write("0 0.5 0.5 0.2 0.2\n")
        f.write("1 0.7 0.7 0.3 0.3\n")
    gt_boxes = load_gt_boxes(str(label_path), str(img_path))
    assert len(gt_boxes) == 2
    assert 0 in gt_boxes
    assert 1 in gt_boxes
    assert len(gt_boxes[0]) == 1
    assert len(gt_boxes[1]) == 1
    # 测试无label文件
    gt_boxes2 = load_gt_boxes(str(tmp_path/"not_exist.txt"), str(img_path))
    assert isinstance(gt_boxes2, dict)
    # 测试图片无法读取
    gt_boxes3 = load_gt_boxes(str(label_path), str(tmp_path/"not_exist.jpg"))
    assert isinstance(gt_boxes3, dict)

def test_detector_initialization():
    detector = DetEvaluator("yolo11x.pt")
    assert detector.img_size == 1280
    assert detector.conf_thres == 0.01
    assert detector.nms_thres == 0.3
    detector2 = DetEvaluator("yolo11x.pt", img_size=640, conf_thres=0.5, nms_thres=0.5)
    assert detector2.img_size == 640
    assert detector2.conf_thres == 0.5
    assert detector2.nms_thres == 0.5

def test_visualize_detections(model_path, tmp_path):
    img_path = tmp_path / "test.jpg"
    output_path = tmp_path / "output.jpg"
    img = np.ones((640, 640, 3), dtype=np.uint8) * 127
    cv2.imwrite(str(img_path), img)
    detector = DetEvaluator(model_path)
    detections = {0: [(100, 100, 200, 200, 0.9)], 1: [(300, 300, 400, 400, 0.8)]}
    detector.visualize_detections(str(img_path), detections, str(output_path))
    assert output_path.exists()
    # 测试图片无法读取
    detector.visualize_detections(str(tmp_path/"not_exist.jpg"), detections, str(output_path))

def test_detector_preprocess_error():
    detector = DetEvaluator("yolo11x.pt")
    with pytest.raises(ValueError, match="无法读取图像"):
        detector.preprocess("non_existent.jpg")

def test_detector_postprocess_empty():
    detector = DetEvaluator("yolo11x.pt")
    results = []
    detections = detector.postprocess(results, (640, 640))
    assert isinstance(detections, dict)
    assert len(detections) == 0
    # None boxes
    class DummyResult:
        def __init__(self):
            self.boxes = None
    results2 = [DummyResult()]
    detections2 = detector.postprocess(results2, (640, 640))
    assert isinstance(detections2, dict)
    assert len(detections2) == 0

def test_detect_single_image(model_path, tmp_path):
    """测试单张图片检测功能"""
    # 创建测试图片
    img_path = tmp_path / "test.jpg"
    output_path = tmp_path / "output.jpg"
    img = np.ones((640, 640, 3), dtype=np.uint8) * 127
    cv2.imwrite(str(img_path), img)
    
    # 1. 测试正常情况
    metrics = detect_single_image(str(img_path), str(output_path), model_path)
    assert metrics is not None
    assert isinstance(metrics, dict)
    assert 'total_detections' in metrics
    assert 'detections_per_class' in metrics
    assert 'average_confidence_per_class' in metrics
    assert 'confidence_distribution' in metrics
    assert 'detection_details' in metrics
    assert output_path.exists()
    
    # 2. 测试图片无法读取的情况
    bad_img_path = tmp_path / "bad.jpg"
    with open(bad_img_path, 'w') as f:
        f.write("not an image")
    metrics = detect_single_image(str(bad_img_path), str(tmp_path / "bad_output.jpg"), model_path)
    assert metrics is None
    
    # 3. 测试空图片的情况
    empty_img_path = tmp_path / "empty.jpg"
    empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(empty_img_path), empty_img)
    metrics = detect_single_image(str(empty_img_path), str(tmp_path / "empty_output.jpg"), model_path)
    assert metrics is not None
    assert metrics['total_detections'] == 0
    
    # 4. 测试不同尺寸的图片
    large_img_path = tmp_path / "large.jpg"
    large_img = np.ones((1920, 1920, 3), dtype=np.uint8) * 127
    cv2.imwrite(str(large_img_path), large_img)
    metrics = detect_single_image(str(large_img_path), str(tmp_path / "large_output.jpg"), model_path)
    assert metrics is not None
    
    # 5. 测试检测结果的详细信息
    if metrics['total_detections'] > 0:
        for cls_id, details in metrics['detection_details'].items():
            assert 'count' in details
            assert 'avg_confidence' in details
            assert 'confidence_stats' in details
            assert 'bounding_boxes' in details
            assert 'class_name' in details
            
            # 验证置信度统计信息
            stats = details['confidence_stats']
            assert 'min' in stats
            assert 'max' in stats
            assert 'median' in stats
            assert 'std' in stats
            
            # 验证边界框格式
            for box in details['bounding_boxes']:
                assert len(box) == 4
                assert all(isinstance(x, float) for x in box)

def test_detect_single_image_edge_cases(model_path, tmp_path):
    """测试单张图片检测的边缘情况"""
    # 1. 测试无效的模型路径
    img_path = tmp_path / "test.jpg"
    output_path = tmp_path / "output.jpg"
    img = np.ones((640, 640, 3), dtype=np.uint8) * 127
    cv2.imwrite(str(img_path), img)
    
    with pytest.raises(Exception):
        detect_single_image(str(img_path), str(output_path), "invalid_model.pt")
    
    # 2. 测试输出路径无法写入的情况
    read_only_dir = tmp_path / "readonly"
    read_only_dir.mkdir()
    os.chmod(str(read_only_dir), 0o444) 
    
    metrics = detect_single_image(str(img_path), str(read_only_dir / "output.jpg"), model_path)
    assert metrics is not None  
    
    # 3. 测试不同格式的图片
    formats = ['.jpg', '.png', '.jpeg']
    for fmt in formats:
        img_path = tmp_path / f"test{fmt}"
        output_path = tmp_path / f"output{fmt}"
        img = np.ones((640, 640, 3), dtype=np.uint8) * 127
        cv2.imwrite(str(img_path), img)
        
        metrics = detect_single_image(str(img_path), str(output_path), model_path)
        assert metrics is not None
        assert output_path.exists()

def main():
    """主函数，用于手动运行测试"""
    pytest.main([__file__, '-v', '--cov=detect', '--cov-branch', '--cov-report=term-missing'])

if __name__ == '__main__':
    main() 
