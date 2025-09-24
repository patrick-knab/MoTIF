import os
import glob
import cv2

from tqdm import tqdm
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    InterpolationMode,
)
from PIL import Image
import torch
import numpy as np
import pandas as pd


def convert_avi_to_mp4(input_path, output_path):
    # Check if output_path already exists

    # Convert .avi to .mp4 using OpenCV
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use mp4 codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()


def create_images(video_path, output_dir):
    if os.path.isdir(output_dir):
        pass
    else:
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_idx += 1


def create_videos(video_path, output_dir):
    # Determine input and output paths for .avi and .webm
    base_path = video_path
    mp4_path = video_path.replace("/Data/", "/Video_data/")
    if mp4_path.endswith(".avi") or mp4_path.endswith(".webm"):
        mp4_path = mp4_path.rsplit(".", 1)[0] + ".mp4"
    else:
        mp4_path = mp4_path.replace(".mp4", ".mp4")  # fallback, should already be .mp4

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(mp4_path):
        # Try .avi first
        avi_path = video_path.replace(".mp4", ".avi")
        if os.path.exists(avi_path):
            convert_avi_to_mp4(avi_path, mp4_path)
        else:
            # Try .webm
            webm_path = video_path.replace(".mp4", ".webm")
            if os.path.exists(webm_path):
                convert_avi_to_mp4(webm_path, mp4_path)
            else:
                raise FileNotFoundError(
                    f"Neither .avi nor .webm found for {video_path}"
                )


# Code to convert one video to few images.
def video2image(video_path, frame_rate=1.0, size=224):
    def preprocess(size, n_px):
        return Compose(
            [
                Resize(size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(size),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )(n_px)

    cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps < 1:
        images = np.zeros([3, size, size], dtype=np.float32)
        print("ERROR: problem reading video file: ", video_path)
    else:
        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration
        interval = fps / frame_rate
        frames_idx = np.floor(np.arange(start_sec * fps, end_sec * fps, interval))
        ret = True
        images = np.zeros([len(frames_idx), 3, size, size], dtype=np.float32)

        for i, idx in enumerate(frames_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_frame = i
            images[i, :, :, :] = preprocess(size, Image.fromarray(frame).convert("RGB"))

        images = images[: last_frame + 1]
    cap.release()
    video_frames = torch.tensor(images)
    return video_frames


## Dataset specific


def create_images_breakfast(gobal_path, start_end=15):
    sub_dirs = sorted(glob.glob(gobal_path + "*/"))

    def get_folder_name(local_path):
        return local_path.split("/")[-2].replace("P", "")

    part_dirs = []

    for i in sub_dirs:
        if int(get_folder_name(i)) <= start_end:
            part_dirs.append(i)

    for i in tqdm(part_dirs, desc="Processing avi to images"):
        video_dirs = sorted(glob.glob(i + "*/"))
        for j in video_dirs:
            avi_files = sorted(glob.glob(j + "*.avi"))
            for k in avi_files:
                output_dir = k.replace("/Data/", "/Image_data/").replace(".avi", "")
                create_images(k, output_dir)


def create_videos_breakfast(gobal_path, start_end=15):
    sub_dirs = sorted(glob.glob(gobal_path + "*/"))

    def get_folder_name(local_path):
        return local_path.split("/")[-2].replace("P", "")

    part_dirs = []

    for i in sub_dirs:
        if int(get_folder_name(i)) <= start_end:
            part_dirs.append(i)

    for i in tqdm(part_dirs, desc="Processing avi to mp4"):
        video_dirs = sorted(glob.glob(i + "*/"))
        for j in video_dirs:
            avi_files = sorted(glob.glob(j + "*.avi"))
            for k in avi_files:
                output_dir = j.replace("/Data/", "/Video_data/")
                create_videos(k, output_dir)


def create_images_ucf(global_path, files):

    path_list = pd.read_csv(files, sep=" ", header=None)

    for i in tqdm(path_list.values):
        video_path = os.path.join(global_path, i[0])
        output_dir = video_path.replace("/Data/", "/Image_data/").replace(".avi", "")
        create_images(video_path, output_dir)


def create_videos_ucf(global_path, files):
    path_list = pd.read_csv(files, sep=" ", header=None)

    for i in tqdm(path_list.values):
        video_path = os.path.join(global_path, i[0])
        output_dir = os.path.dirname(
            video_path.replace("/Data/", "/Video_data/").replace(".avi", "")
        )
        create_videos(video_path, output_dir)


def create_images_hmdb(global_path, path_list):
    for i in tqdm(path_list):
        local_name = i.split("/")[1]
        video_path = global_path + i
        output_dir = (
            video_path.replace("/Data/", "/Image_data/")
            .replace(".avi", "")
            .replace("//", "/")
        )
        video_path = global_path + local_name + i
        create_images(video_path, output_dir)


def create_videos_hmdb(global_path, path_list):
    for i in tqdm(path_list):
        local_name = i.split("/")[1]
        video_path = global_path + local_name + i
        output_dir = (
            video_path.replace("/Data/", "/Video_data/")
            .replace(".avi", "")
            .replace("//", "/")
        )
        # remove last folder in outputdir
        output_dir = os.path.dirname(output_dir)
        video_path = global_path + local_name + i
        create_videos(video_path, output_dir)


def create_images_sth2(global_path, files):
    for i in tqdm(files):
        video_path = global_path + str(i) + ".webm"
        output_dir = video_path.replace("/Data/", "/Image_data/").replace(".webm", "")
        create_images(video_path, output_dir)


def create_videos_sth2(global_path, files):
    for i in tqdm(files):
        video_path = global_path + str(i) + ".webm"
        output_dir = os.path.dirname(
            video_path.replace("/Data/", "/Video_data/").replace(".webm", "")
        )
        create_videos(video_path, output_dir)
