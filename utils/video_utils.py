import os
import glob
import shutil
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
    """Write ``<output_dir>/<video stem>.mp4`` for one source video (.avi, .webm or .mp4).

    The output location is taken from ``output_dir`` (not re-derived from the input path),
    so nested or duplicated class folders in the raw data never leak into ``Video_data/``.
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    mp4_path = os.path.join(output_dir, stem + ".mp4")
    if os.path.exists(mp4_path):
        return mp4_path

    os.makedirs(output_dir, exist_ok=True)

    src = video_path
    if not os.path.exists(src):
        # allow callers to pass the .mp4 name of a video stored as .avi/.webm
        for ext in (".avi", ".webm"):
            candidate = os.path.splitext(video_path)[0] + ext
            if os.path.exists(candidate):
                src = candidate
                break
        else:
            raise FileNotFoundError(f"Neither .avi nor .webm found for {video_path}")

    if src.lower().endswith(".mp4"):
        shutil.copyfile(src, mp4_path)
    else:
        convert_avi_to_mp4(src, mp4_path)
    return mp4_path


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


def resolve_hmdb_video(global_path, rel_path):
    """Locate one HMDB-51 clip given ``rel_path = "<class>/<clip>.avi"``.

    Two raw layouts are accepted under ``Data/``:

    * ``Data/<class>/<clip>.avi`` (every per-class archive of ``hmdb51_org.rar``
      extracted directly into ``Data/``), and
    * ``Data/<class>/<class>/<clip>.avi`` (every per-class archive extracted into a
      folder of the same name), which earlier versions of this script assumed.
    """
    rel_path = rel_path.lstrip("/")
    class_name = rel_path.split("/")[0]
    candidates = [
        os.path.join(global_path, rel_path),
        os.path.join(global_path, class_name, rel_path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"HMDB clip {rel_path!r} not found; looked for " + " and ".join(candidates)
    )


def _hmdb_dataset_root(global_path):
    # ".../Datasets/HMDB/Data/" -> ".../Datasets/HMDB"
    return os.path.dirname(os.path.normpath(global_path))


def create_images_hmdb(global_path, path_list):
    """Extract frames to ``Image_data/<class>/<clip>/*.jpg`` for ``<class>/<clip>.avi`` entries."""
    root = _hmdb_dataset_root(global_path)
    for rel in tqdm(path_list, desc="HMDB avi to images"):
        rel = rel.lstrip("/")
        video_path = resolve_hmdb_video(global_path, rel)
        output_dir = os.path.join(root, "Image_data", os.path.splitext(rel)[0])
        create_images(video_path, output_dir)


def create_videos_hmdb(global_path, path_list):
    """Convert clips to ``Video_data/<class>/<clip>.mp4`` for ``<class>/<clip>.avi`` entries."""
    root = _hmdb_dataset_root(global_path)
    for rel in tqdm(path_list, desc="HMDB avi to mp4"):
        rel = rel.lstrip("/")
        video_path = resolve_hmdb_video(global_path, rel)
        output_dir = os.path.join(root, "Video_data", os.path.dirname(rel))
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
