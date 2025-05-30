import sys
import argparse
import os

# https://www.codewithgpu.com/u/iscyy

sys.path.append('/root/ultralytics/') # Path 以Autodl为例

from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    model = YOLO(yaml) 

    model.info()

    model = YOLO('yolo11n.pt')

    model.predict('/root/YOLOv11/ultralytics/assets/smoke3.png', save=True, imgsz=640, conf=0.5)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r'yolov8n.pt', help='initial weights path')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)