"""Create the derived ``Video_data/`` (mp4 clips) and ``Image_data/`` (jpg frames) folders
from the raw dataset files under ``Datasets/<DatasetName>/Data/``.

Examples::

    python save_videos.py --dataset HMDB
    python save_videos.py --dataset UCF101
    python save_videos.py --dataset Something2
    python save_videos.py --dataset Breakfast            # all participants
    python save_videos.py --dataset Breakfast --p0 15    # participants P03..P15 only
"""

import argparse
import glob
import os

import pandas as pd

from utils.paths import DATASETS_DIR, dataset_folder
from utils.video_utils import (
    create_images_breakfast,
    create_images_hmdb,
    create_images_sth2,
    create_images_ucf,
    create_videos_breakfast,
    create_videos_hmdb,
    create_videos_sth2,
    create_videos_ucf,
)

# Breakfast participants are P03..P54; any bound >= 54 processes the whole dataset.
BREAKFAST_ALL_PARTICIPANTS = 55


def hmdb_split_entries(labels_path, split_index=1):
    """Return (train, test, ignore) lists of ``<class>/<clip>.avi`` for one HMDB split.

    ``labels_path`` is the ``testTrainMulti_7030_splits`` folder from the official
    ``test_train_splits.rar``; every ``<class>_test_split<k>.txt`` lists clips with a
    flag (1 = train, 2 = test, 0 = unused).
    """
    split_files = sorted(
        p for p in glob.glob(os.path.join(labels_path, "*.txt")) if f"split{split_index}" in p
    )
    if not split_files:
        raise FileNotFoundError(
            f"No *_test_split{split_index}.txt files in {labels_path}. Download "
            "test_train_splits.rar from the HMDB-51 website and extract it there."
        )

    train, test, ignore = [], [], []
    for path in split_files:
        class_name = os.path.basename(path).split("_test_split")[0]
        with open(path, "r") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue  # skip malformed lines
                clip, flag = parts[0], parts[1]
                rel = os.path.join(class_name, clip)  # joined exactly once
                if flag == "1":
                    train.append(rel)
                elif flag == "2":
                    test.append(rel)
                else:
                    ignore.append(rel)
    return train, test, ignore


def create_videos(dataset, p0=0, hmdb_split=1):
    folder = dataset_folder(dataset)
    dataset_root = DATASETS_DIR / folder
    video_path = str(dataset_root / "Data") + os.sep
    if not os.path.isdir(video_path):
        raise FileNotFoundError(
            f"Raw dataset folder {video_path} does not exist. Place the original files there first."
        )

    if folder == "Breakfast":
        if p0 <= 0:
            p0 = BREAKFAST_ALL_PARTICIPANTS
        create_images_breakfast(video_path, p0)
        create_videos_breakfast(video_path, p0)

    elif folder == "UCF101":
        for split_index in (1, 2, 3):
            ucf_test_list = str(dataset_root / "ucfTrainTestlist" / f"testlist0{split_index}.txt")
            create_images_ucf(video_path, ucf_test_list)
            create_videos_ucf(video_path, ucf_test_list)

    elif folder == "HMDB":
        labels_path = str(dataset_root / "testTrainMulti_7030_splits")
        train_dirs, test_dirs, ignore_dirs = hmdb_split_entries(labels_path, hmdb_split)
        for entries in (test_dirs, train_dirs, ignore_dirs):
            create_images_hmdb(video_path, entries)
            create_videos_hmdb(video_path, entries)

    elif folder == "Something2":
        labels_dir = dataset_root / "labels"
        for name in ("test", "train", "validation"):
            ids = pd.read_json(str(labels_dir / f"{name}.json")).values.tolist()
            ids = [i[0] for i in ids]
            create_images_sth2(video_path, ids)
            create_videos_sth2(video_path, ids)

    else:
        raise ValueError(f"save_videos.py has no preprocessing routine for dataset {dataset!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the derived Video_data/ and Image_data/ folders for a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Breakfast",
        help="Dataset name or key: Breakfast, UCF101, HMDB (hmdb51), Something2 (ssv2)",
    )
    parser.add_argument(
        "--p0",
        type=int,
        default=0,
        help="Breakfast only: highest participant number to process (P03..P<p0>); 0 = all participants",
    )
    parser.add_argument(
        "--hmdb-split",
        type=int,
        default=1,
        help="HMDB only: which *_test_split<k>.txt files to read (the union of its flags covers all clips)",
    )

    args = parser.parse_args()
    create_videos(dataset=args.dataset, p0=args.p0, hmdb_split=args.hmdb_split)
