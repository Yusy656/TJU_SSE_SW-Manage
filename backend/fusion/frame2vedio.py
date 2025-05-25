import cv2
import os

def images_to_video(image_folder, video_name, fps=25):
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    image_folder = 'dehazed-results/processed_rgb_smoked3/'
    video_name = 'dehazed_rgb_smoked3.mp4'
    images_to_video(image_folder, video_name, fps=25)    