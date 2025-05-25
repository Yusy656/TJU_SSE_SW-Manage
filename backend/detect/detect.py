import torch
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'yolo11x.pt')

class DetEvaluator:
    def __init__(self, model_path, img_size=1280, conf_thres=0.01, nms_thres=0.3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path).to(self.device)
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.scale = None
        self.pad = None
        self.class_names = self.model.names  # 新增类别名称属性

    def preprocess(self, img_path):
        """图像预处理（保持宽高比）"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像：{img_path}")
        
        h, w = img.shape[:2]
        self.scale = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * self.scale), int(h * self.scale)
        
        img = cv2.resize(img, (new_w, new_h))
        pad_top = (self.img_size - new_h) // 2
        pad_bottom = self.img_size - new_h - pad_top
        pad_left = (self.img_size - new_w) // 2
        pad_right = self.img_size - new_w - pad_left
        self.pad = (pad_top, pad_bottom, pad_left, pad_right)
        
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, 
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        tensor = torch.from_numpy(img).permute(2,0,1).float().div(255)
        return tensor.unsqueeze(0).to(self.device).half(), (h, w)

    def detect(self, tensor):
        """执行推理"""
        with torch.no_grad(), torch.cuda.amp.autocast():
            results = self.model.predict(
                tensor, 
                conf=self.conf_thres,
                iou=self.nms_thres,
                imgsz=self.img_size,
                augment=False
            )
        return results

    def postprocess(self, results, orig_shape):
        """后处理（保留所有类别）"""
        orig_h, orig_w = orig_shape
        detections = defaultdict(list)
        
        for r in results:
            if r.boxes is None:
                continue
                
            boxes = r.boxes.xyxyn.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            
            for cls, box, conf in zip(classes, boxes, confs):
                # 坐标转换
                x1 = (box[0] * self.img_size - self.pad[2]) / self.scale
                y1 = (box[1] * self.img_size - self.pad[0]) / self.scale
                x2 = (box[2] * self.img_size - self.pad[2]) / self.scale
                y2 = (box[3] * self.img_size - self.pad[0]) / self.scale
                
                # 边界保护
                x1 = np.clip(x1, 0, orig_w)
                y1 = np.clip(y1, 0, orig_h)
                x2 = np.clip(x2, 0, orig_w)
                y2 = np.clip(y2, 0, orig_h)
                
                detections[int(cls)].append((x1, y1, x2, y2, float(conf)))
                
        return detections
        
    def visualize_detections(self, img_path, detections, output_path):
        """可视化并保存检测结果"""
        # 创建输出目录
        # vis_dir = Path(output_dir)
        # vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取原始图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像：{img_path}")
            return
        
        # 绘制所有检测框
        for cls_id, boxes in detections.items():
            for box in boxes:
                x1, y1, x2, y2, conf = map(float, box)
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # 随机生成颜色种子
                color_seed = int(cls_id) * 10
                np.random.seed(color_seed)
                color = [int(c) for c in np.random.randint(0, 255, 3)]
                
                # 绘制边界框
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # 构造标签文本
                label = f"{self.class_names[int(cls_id)]} {conf:.2f}"
                
                # 计算文本尺寸
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                
                # 绘制文本背景
                cv2.rectangle(img, 
                            (x1, y1 - text_height - 10),
                            (x1 + text_width, y1 - 10),
                            color, -1)
                
                # 添加文本
                cv2.putText(img, label, 
                          (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, (255, 255, 255), 2)
        
        # 保存结果
        # output_path = vis_dir / "detected_image.jpg"
        cv2.imwrite(str(output_path), img)
        print(f"检测结果已保存至：{output_path}")
        
class PRCalculator:
    def __init__(self, iou_threshold=0.3):
        self.iou_thresh = iou_threshold
        self.class_data = defaultdict(lambda: {
            'detections': [],  # (conf, is_tp)
            'gt_count': 0
        })

    @staticmethod
    def bbox_iou(box1, box2):
        """计算IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
        return inter_area / (area1 + area2 - inter_area + 1e-6)

    def process_image(self, class_id, det_boxes, gt_boxes):
        """处理单张图片"""
        # 按置信度降序排序
        det_boxes = sorted(det_boxes, key=lambda x: x[4], reverse=True)
        matched = set()
        
        for det in det_boxes:
            max_iou = 0.0
            matched_idx = -1
            det_box = det[:4]
            
            # 寻找最佳匹配
            for idx, gt_box in enumerate(gt_boxes):
                if idx in matched:
                    continue
                iou = self.bbox_iou(det_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    matched_idx = idx
            
            # 记录检测结果
            is_tp = 1 if max_iou >= self.iou_thresh else 0
            self.class_data[class_id]['detections'].append((det[4], is_tp))
            
            if is_tp:
                matched.add(matched_idx)
        
        # 统计GT数量
        self.class_data[class_id]['gt_count'] += len(gt_boxes)

    def calculate_metrics(self):
        """计算所有类别的PR指标"""
        metrics = {}
        for cls_id, data in self.class_data.items():
            # 按置信度降序排序
            sorted_det = sorted(data['detections'], key=lambda x: x[0], reverse=True)
            
            # 初始化累积计数
            tp_acc = 0
            fp_acc = 0
            precisions = []
            recalls = []
            
            for conf, is_tp in sorted_det:
                tp_acc += is_tp
                fp_acc += (1 - is_tp)
                
                precision = tp_acc / (tp_acc + fp_acc + 1e-6)
                recall = tp_acc / data['gt_count'] if data['gt_count'] > 0 else 0.0
                
                precisions.append(precision)
                recalls.append(recall)
            
            # 计算AP（Pascal VOC 11点插值法）
            ap = 0.0
            for t in np.arange(0, 1.1, 0.1):
                prec_at_recall = [p for r, p in zip(recalls, precisions) if r >= t]
                ap += max(prec_at_recall) if prec_at_recall else 0.0
            ap /= 11.0
            
            metrics[cls_id] = {
                'detections': sorted_det,  # 新增此行
                'precision': precisions,
                'recall': recalls,
                'ap': ap,
                'gt_count': data['gt_count']
            }
        
        return metrics

class PRVisualizer:
    def __init__(self, output_dir="pr_curves"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_pr_curves(self, metrics):
        """绘制所有类别的PR曲线"""
        plt.figure(figsize=(12, 8))
        
        for cls_id, data in metrics.items():
            precisions = data['precision']
            recalls = data['recall']
            ap = data['ap']
            
            # 生成平滑曲线
            sorted_indices = np.argsort(recalls)
            recalls_sorted = np.array(recalls)[sorted_indices]
            precisions_sorted = np.array(precisions)[sorted_indices]
            
            plt.plot(recalls_sorted, precisions_sorted, lw=2,
                    label=f'Class {cls_id} (AP={ap:.2f})')

        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curves', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        
        plt.savefig(self.output_dir / 'all_classes_pr.png', bbox_inches='tight')
        plt.close()

def load_gt_boxes(label_path, img_path):
    """加载真实框"""
    img = cv2.imread(img_path)
    if img is None:
        return defaultdict(list)
    
    h, w = img.shape[:2]
    gt_boxes = defaultdict(list)
    
    if not os.path.exists(label_path):
        return gt_boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            try:
                cls_id, xc, yc, bw, bh = map(float, parts)
                cls_id = int(cls_id)
            except:
                continue
            
            # 转换为绝对坐标
            x1 = (xc - bw/2) * w
            y1 = (yc - bh/2) * h
            x2 = (xc + bw/2) * w
            y2 = (yc + bh/2) * h
            
            gt_boxes[cls_id].append((
                max(0.0, x1), max(0.0, y1),
                min(w, x2), min(h, y2)
            ))
    
    return gt_boxes

def main(img_dir, label_dir, model_path="./yolo11x.pt", output_dir="results"):
    # 初始化组件
    evaluator = DetEvaluator(model_path)
    pr_calculator = PRCalculator(iou_threshold=0.3)
    visualizer = PRVisualizer(output_dir)
    
    # 遍历图像
    img_dir = Path(img_dir)
    for img_path in img_dir.glob('*.*'):
        if img_path.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
            continue
        
        # 关键修改：保留原始文件名（含fused_前缀）
        base_name = img_path.stem
        if base_name.startswith("fused_"):
            base_name = base_name[6:]
        label_path = Path(label_dir) / f"{base_name}.txt"

        
        try:
            # 执行检测
            tensor, orig_shape = evaluator.preprocess(str(img_path))
            results = evaluator.detect(tensor)
            detections = evaluator.postprocess(results, orig_shape)
            
            # 加载真实框
            gt_boxes = load_gt_boxes(str(label_path), str(img_path))
            
            # 处理每个类别
            all_classes = set(detections.keys()).union(gt_boxes.keys())
            for cls_id in all_classes:
                pr_calculator.process_image(
                    cls_id,
                    detections.get(cls_id, []),
                    gt_boxes.get(cls_id, [])
                )
                
        except Exception as e:
            print(f"处理失败 {img_path.name}: {str(e)}")
            continue
    
    # 计算最终指标
    metrics = pr_calculator.calculate_metrics()
    
    # 保存结果
    (Path(output_dir)/'metrics.json').write_text(
        json.dumps(metrics, indent=2, default=float)
    )
    
    # 可视化
    visualizer.plot_pr_curves(metrics)
    
    # 打印汇总
    print("\n评估结果汇总（详细指标）：")
    for cls_id, data in sorted(metrics.items()):
        # 获取基础数据
        sorted_det = data['detections']  # 从metrics直接获取
        total_gt = data['gt_count']
        total_det = len(sorted_det)
        
        # 计算TP/FP/FN
        tp = sum(is_tp for _, is_tp in sorted_det)
        fp = total_det - tp
        fn = max(total_gt - tp, 0)
        
        # 计算精确率与召回率
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (total_gt + 1e-6) if total_gt > 0 else 0.0
        
        # 打印结果
        print(f"类别 {cls_id}:")
        print(f"  AP: {data['ap']:.4f}")
        print(f"  GT数量: {total_gt}")
        print(f"  检测数量: {total_det}")
        print(f"  TP: {tp} | FP: {fp} | FN: {fn}") 
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print("-"*60)

def detect_single_image(img_path, output_path, model_path=MODEL_PATH):
    """单张图片检测函数
    
    Args:
        img_path: 输入图片路径
        model_path: 模型路径
        output_dir: 输出目录
        
    Returns:
        dict: 包含检测结果的指标字典
    """
    # 初始化检测器，使用更优的参数设置
    evaluator = DetEvaluator(
        model_path=model_path,
        conf_thres=0.25,  # 降低置信度阈值以检测更多目标
        nms_thres=0.45,    # 使用更严格的NMS阈值
        img_size=1920     # 使用更大的输入尺寸
    )
    
    try:
        # 执行检测
        tensor, orig_shape = evaluator.preprocess(img_path)
        results = evaluator.detect(tensor)
        detections = evaluator.postprocess(results, orig_shape)
        
        # 计算检测统计信息
        detection_stats = {
            'total_detections': 0,
            'class_detections': defaultdict(int),
            'confidence_scores': defaultdict(list),
            'bounding_boxes': defaultdict(list),
            'class_names': evaluator.class_names  # 添加类别名称
        }
        
        # 统计每个类别的检测结果
        for cls_id, boxes in detections.items():
            detection_stats['total_detections'] += len(boxes)
            detection_stats['class_detections'][cls_id] = len(boxes)
            
            for box in boxes:
                x1, y1, x2, y2, conf = box
                detection_stats['confidence_scores'][cls_id].append(float(conf))
                detection_stats['bounding_boxes'][cls_id].append([float(x1), float(y1), float(x2), float(y2)])
        
        # 计算每个类别的平均置信度和置信度分布
        avg_confidence = {}
        confidence_distribution = {}
        for cls_id, scores in detection_stats['confidence_scores'].items():
            if scores:
                avg_confidence[cls_id] = sum(scores) / len(scores)
                # 计算置信度分布
                confidence_distribution[cls_id] = {
                    'min': min(scores),
                    'max': max(scores),
                    'median': sorted(scores)[len(scores)//2],
                    'std': np.std(scores) if len(scores) > 1 else 0
                }
            else:
                avg_confidence[cls_id] = 0
                confidence_distribution[cls_id] = {'min': 0, 'max': 0, 'median': 0, 'std': 0}
        
        # 可视化检测结果
        evaluator.visualize_detections(img_path, detections, output_path)
        
        # 准备返回的指标
        metrics = {
            'total_detections': detection_stats['total_detections'],
            'detections_per_class': dict(detection_stats['class_detections']),
            'average_confidence_per_class': avg_confidence,
            'confidence_distribution': confidence_distribution,
            'detection_details': {
                cls_id: {
                    'count': count,
                    'avg_confidence': avg_confidence[cls_id],
                    'confidence_stats': confidence_distribution[cls_id],
                    'bounding_boxes': detection_stats['bounding_boxes'][cls_id],
                    'class_name': detection_stats['class_names'].get(cls_id, f'Unknown_{cls_id}')
                }
                for cls_id, count in detection_stats['class_detections'].items()
            }
        }
        
        print(f"\n检测结果统计:")
        print(f"总检测数量: {metrics['total_detections']}")
        for cls_id, details in metrics['detection_details'].items():
            print(f"\n类别 {details['class_name']}:")
            print(f"  检测数量: {details['count']}")
            print(f"  平均置信度: {details['avg_confidence']:.4f}")
            print(f"  置信度分布: 最小={details['confidence_stats']['min']:.4f}, "
                  f"最大={details['confidence_stats']['max']:.4f}, "
                  f"中位数={details['confidence_stats']['median']:.4f}, "
                  f"标准差={details['confidence_stats']['std']:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"处理图片失败 {img_path}: {str(e)}")
        return None
    
if __name__ == "__main__":
    main(
        img_dir="fusion_results",
        label_dir="labels",
        model_path="yolo11x.pt",
        output_dir="evaluation_results"
    )