import os
import argparse
import cv2

def images_to_video(imgs_dir, fps=30, keep_imgs=False):
    fnames = sorted(os.listdir(imgs_dir))
    imgs = [cv2.imread(os.path.join(imgs_dir, fn), cv2.IMREAD_COLOR) for fn in fnames if fn.lower().endswith((".jpg", ".png"))]

    if len(imgs) == 0:
        print(f"[skip] no images found in {imgs_dir}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    size = imgs[0].shape[:2]  # (h, w)
    folder_name = os.path.basename(imgs_dir.rstrip("/\\"))  # 去掉末尾斜杠再取名字
    video_path = os.path.join(imgs_dir, f"{folder_name}.mp4")
    videoWriter = cv2.VideoWriter(video_path, fourcc, fps, (size[1], size[0]))

    for img in imgs:
        videoWriter.write(img)

    videoWriter.release()
    print(f"[done] video saved at {video_path}")

    if not keep_imgs:
        for fn in fnames:
            if fn.lower().endswith((".jpg", ".png")):
                os.remove(os.path.join(imgs_dir, fn))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images in subfolders to videos")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the root folder containing subfolders of images")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--keep_imgs", action="store_true", default=False)

    args = parser.parse_args()

    for subdir in os.listdir(args.root_dir):
        subdir_path = os.path.join(args.root_dir, subdir)
        if os.path.isdir(subdir_path):
            images_to_video(subdir_path, args.fps, args.keep_imgs)