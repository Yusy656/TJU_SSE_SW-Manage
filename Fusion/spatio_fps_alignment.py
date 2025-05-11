import cv2
import subprocess


def process_videos(rgb_input_path, ir_input_path, output_prefix):
    """独立处理双路视频，确保各自完整处理"""
    # 独立处理RGB视频
    process_rgb(rgb_input_path, f'{output_prefix}_rgb.mp4')
    # 独立处理红外视频
    process_ir(ir_input_path, f'{output_prefix}_ir_intermediate.mp4')


def process_rgb(input_path, output_path):
    """处理RGB视频流"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = 640, 419  # 目标尺寸

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 处理逻辑
        resized = cv2.resize(frame, (704, 419), interpolation=cv2.INTER_AREA)
        cropped = resized[:, 32:672]  # 裁剪宽度
        writer.write(cropped)

    cap.release()
    writer.release()


def process_ir(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 动态获取第一帧的裁剪后宽度
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return
    cropped = frame[66:485, :]
    target_width = cropped.shape[1]  # 裁剪后的实际宽度
    target_height = 419

    # 重置视频读取位置到开头
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_width, target_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[66:485, :]
        writer.write(cropped)

    cap.release()
    writer.release()


def video_speed_correction(input_path, output_path, speed_factor=1.2, target_fps=25):
    """红外视频倍速处理"""
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-i', input_path,
        '-vf', f'setpts={1 / speed_factor:.3f}*PTS, fps={target_fps}',
        '-r', str(target_fps),
        '-an',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    try:
        subprocess.run(
            ffmpeg_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"速度校正成功: {input_path} -> {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"错误信息:\n{e.stderr.decode()}")


def process_and_correct(rgb_input_path, ir_input_path, output_prefix):
    """完整处理流程入口函数"""
    # 第一步：空间变换处理
    process_videos(rgb_input_path, ir_input_path, output_prefix)

    # 第二步：红外视频倍速处理
    intermediate_ir = f'{output_prefix}_ir_intermediate.mp4'
    final_ir = f'{output_prefix}_ir_final.mp4'
    video_speed_correction(
        intermediate_ir,
        final_ir,
        speed_factor=1.2,
        target_fps=25
    )


# 使用示例（根据实际情况修改路径）
if __name__ == "__main__":
    # # 单个视频对处理示例
    # process_and_correct(
    #     'E:/BaiduNetdiskDownload/wuxi/output_rgb_smoked3.mp4',
    #     'E:/BaiduNetdiskDownload/wuxi/output_tr_smoked3.mp4',
    #     'processed_smoked3'
    # )

    # 批量处理示例（根据需要取消注释）
    video_pairs = [
        ('output_rgb.mp4', 'output_tr.mp4', 'processed'),
        # ('output_rgb_smoked.mp4', 'output_tr_smoked.mp4', 'processed_smoked'),
        # ('output_rgb_smoked1.mp4', 'output_tr_smoked1.mp4', 'processed_smoked1'),
        # ('output_rgb_smoked2.mp4', 'output_tr_smoked2.mp4', 'processed_smoked2'),
        # ('output_rgb_smoked3.mp4', 'output_tr_smoked3.mp4', 'processed_smoked3')
    ]

    for rgb, ir, prefix in video_pairs:
        process_and_correct(
            f'E:/BaiduNetdiskDownload/wuxi/{rgb}',
            f'E:/BaiduNetdiskDownload/wuxi/{ir}',
            f'{prefix}'
        )
