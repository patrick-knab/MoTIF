import os
import glob
from collections import defaultdict
import cv2
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from utils.video_utils import *

def create_videos(dataset, p0):
    
    video_path = f"./Datasets/{dataset}/Data/"
    
    if dataset == "Breakfast":
        if p0 == 0:
            p0 == 55
        create_images_breakfast(video_path, p0) 
        create_videos_breakfast(video_path, p0)
        
    elif dataset == "UCF101":
        ucf_test_list = "./Datasets/UCF101/ucfTrainTestlist/testlist01.txt" # number of test list
        create_images_ucf(video_path, ucf_test_list)
        create_videos_ucf(video_path, ucf_test_list)
        ucf_test_list = "./Datasets/UCF101/ucfTrainTestlist/testlist02.txt" # number of test list
        create_images_ucf(video_path, ucf_test_list)
        create_videos_ucf(video_path, ucf_test_list)
        ucf_test_list = "./Datasets/UCF101/ucfTrainTestlist/testlist03.txt" # number of test list
        create_images_ucf(video_path, ucf_test_list)
        create_videos_ucf(video_path, ucf_test_list)
        
    elif dataset == "HMDB":
        labels_path = "./Datasets/HMDB/testTrainMulti_7030_splits/"
        path_text_dirs = glob.glob(os.path.join(labels_path, "*.txt"))
        
        idx_test_list = 1 # id of test set 
        path_text_dirs_idx = [i for i in path_text_dirs if f"split{idx_test_list}" in i]
        # sort the paths to ensure consistent order
        path_text_dirs_idx.sort()
        
        test_dirs = []
        train_dirs = []
        ignore_dirs = []
        labels = []
        
        for path in path_text_dirs_idx:
            folder_name = path.split("splits")[1]
            folder_name = folder_name.split("_test")[0]
            labels.append(folder_name.strip("/").replace("_", " "))
            with open(path, "r") as local_text:
                lines = local_text.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue  # skip malformed lines
                    filename, split = parts[0], parts[1]
                    filename = os.path.join(folder_name,filename)
                    filename = os.path.join(folder_name, filename)
                    if split == "1":
                        train_dirs.append(filename)
                    elif split == "2":
                        test_dirs.append(filename)
                    else:
                        ignore_dirs.append(filename)
        
        create_images_hmdb(video_path, test_dirs)
        create_videos_hmdb(video_path, test_dirs)
        create_images_hmdb(video_path, train_dirs)
        create_videos_hmdb(video_path, train_dirs)
        create_images_hmdb(video_path, ignore_dirs)
        create_videos_hmdb(video_path, ignore_dirs)
        
    if dataset == "Something2":
        test_path = "./Datasets/Something2/labels/test.json"
        test_ids = pd.read_json(test_path).values.tolist()
        test_ids = [i[0] for i in test_ids]
        create_images_sth2(video_path, test_ids)
        create_videos_sth2(video_path, test_ids)
        
        train_path = "./Datasets/Something2/labels/train.json"
        train_ids = pd.read_json(train_path).values.tolist()
        train_ids = [i[0] for i in train_ids]
        create_images_sth2(video_path, train_ids)
        create_videos_sth2(video_path, train_ids)
        
        val_path = "./Datasets/Something2/labels/validation.json"
        val_ids = pd.read_json(val_path).values.tolist()
        val_ids = [i[0] for i in val_ids]
        create_images_sth2(video_path, val_ids)
        create_videos_sth2(video_path, val_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create video embeddings for a dataset")
    parser.add_argument("--dataset", type=str, default="Breakfast", help="Dataset name")
    parser.add_argument("--p0", type=int, default=0, help="Number of parts to process")

    args = parser.parse_args()

    create_videos(
        dataset = args.dataset,
        p0 = args.p0 
    ) 
