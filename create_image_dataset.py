import numpy as np
import pandas as pd
import os
import cv2
from pandas.core.frame import DataFrame
from tqdm import tqdm
import math
from PIL import Image
import random

def create_images_and_pandas_file(video_folder, sponsor_data, save_path):
    video_list_full_name = os.listdir(video_folder)
    video_list = [name.split(".")[0] for name in tqdm(video_list_full_name)]
    #video_list = [name for name in tqdm(video_list) if "sponsor" in sponsor_data[sponsor_data['videoID'] == name]["category"].tolist()]
    save_dic = {"video_id": [], "frame_name": [], "label": []}
    for id in tqdm(video_list):
        convert_video_to_frames(os.path.join(video_folder, id + ".mp4"))
        if os.listdir("data/frames"):
            video_info = sponsor_data[sponsor_data["videoID"] == id]
            start_times = video_info["startTime"].tolist()
            end_times = video_info["endTime"].tolist()
            cats = video_info["category"].tolist()
            num_images = 0
            del_list = []
            for start_time, end_time, cat in zip(start_times, end_times, cats):
                if cat == "sponsor":
                    for i in range(int(start_time)+1, int(end_time)+1):
                        try:
                            image = Image.open("data/frames/frame%d.jpg" % i)
                            save_name = f"sponsor_{id}_{i}.jpg"
                            image.save(os.path.join(save_path, save_name))
                            del_list.append("data/frames/frame%d.jpg" % i)
                            save_dic["video_id"].append(id)
                            save_dic["frame_name"].append(save_name)
                            save_dic["label"].append("sponsor")
                            num_images += 1
                        except:
                            print(f"WARNING: Frame not found {id}")
            for del_elem in set(del_list):
                os.remove(del_elem)
            if num_images > len(os.listdir("data/frames")):
                num_images = len(os.listdir("data/frames"))
            frame_files = random.sample(os.listdir("data/frames"), num_images)
            for file, i in zip(frame_files, range(num_images)):
                image = Image.open(os.path.join("data/frames", file))
                save_name = f"video_{id}_{i}.jpg"
                image.save(os.path.join(save_path, save_name))
                save_dic["video_id"].append(id)
                save_dic["frame_name"].append(save_name)
                save_dic["label"].append("video")
            for file in os.listdir("data/frames"):
                os.remove(os.path.join("data/frames",file))
    df = pd.DataFrame(save_dic)
    df.to_csv(os.path.join(save_path, "dataset.csv"))


def train_val_test_split(dataset_path, val=0.15, test=0.1):
    dataset = pd.read_csv(os.path.join(dataset_path, "dataset.csv"))
    video_ids = list(set(dataset["video_id"].tolist()))
    zip_data = list(zip(dataset["video_id"].tolist(), dataset["frame_name"].tolist(), dataset["label"].tolist()))
    total_len = len(zip_data)
    train_num = int(total_len * (1 - val - val - test))
    val_num = int(total_len * val) 
    test_num = int(total_len * test)
    random.shuffle(video_ids)
    hard_val = []
    for id in video_ids:
        part = [d for d in zip_data if d[0] == id]
        if len(hard_val) + len(part) <= val_num:
            hard_val.extend(part)
            zip_data = [d for d in zip_data if d[0] != id]
    hard_id, hard_name, hard_label = zip(*hard_val)
    hard_dic = {"video_id":hard_id, "frame_name":hard_name, "label":hard_label}
    val_hard = pd.DataFrame(hard_dic)
    val_hard.to_csv(os.path.join(dataset_path, "val_hard.csv"))
    data  = random.sample(zip_data, train_num+val_num+test_num)
    train_data = data[:train_num]
    val_data = data[train_num:train_num+val_num]
    test_data = data[train_num+val_num:]
    for d, p in zip([train_data, val_data, test_data], ["train.csv", "val.csv", "test.csv"]):
        id, name, label = zip(*d)
        dic = {"video_id":id, "frame_name":name, "label":label}
        df = pd.DataFrame(dic)
        df.to_csv(os.path.join(dataset_path, p))


def convert_video_to_frames(video_file):
    count = 0
    caption = cv2.VideoCapture(video_file)
    frame_rate = caption.get(5)
    while caption.isOpened():
        frame_id = caption.get(1) #current frame number
        ret, frame = caption.read()
        if (ret != True):
            break
        if (frame_id % math.floor(frame_rate) == 0):
            filename ="data/frames/frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, frame)
    caption.release()


if __name__=="__main__":
    sponsor_data = pd.read_csv("data/sponsor_timestamps.csv")
    #create_images_and_pandas_file("data/videos", sponsor_data, "data/image_dataset")
    train_val_test_split("data/image_dataset")