import pandas as pd
import os,sys
import subprocess
import random
from pandas.core import frame
from tqdm import tqdm
import cv2
import math
import torch
from PIL import Image
from torchvision import models, transforms, datasets

def embedding_to_annotate(save_dir, embeddings, vid_csv):
    start_times = vid_csv["startTime"].tolist()
    end_times = vid_csv["endTime"].tolist()
    cats = vid_csv["category"].tolist()
    category = ["video" for _ in range(len(embeddings))]
    for i in range(len(category)):
        for start_time, end_time, cat in zip(start_times, end_times, cats):
            if i >= start_time and i <= end_time:
                category[i] = cat
    dic = {'embedding':embeddings, 'category':category}
    df = pd.DataFrame(data=dic)
    df.to_csv(save_dir)


def do_frame_embedding(model, frame_dir):
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    embeddings = []
    for image_file in tqdm(os.listdir(frame_dir)):
        image = Image.open(os.path.join(frame_dir,image_file))
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')
        with torch.no_grad():
            output = model(input_batch)
        output = torch.flatten(output)
        output = output.cpu()
        embeddings.append(output.numpy())
    return embeddings


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

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
sponsor_data = pd.read_csv("data/sponsor_timestamps.csv")
vid_ids = list(sponsor_data["videoID"].unique())
random.shuffle(vid_ids)
for id in tqdm(vid_ids):
    p = f"data/videos/{id}.mp4"
    if not os.path.isfile(p):
        try:
            subprocess.call(["youtube-dl", f"https://www.youtube.com/watch?v={id}", "-f",'bestvideo[height<=240]', "-o", p])
            # Threshold need more attention
            subprocess.call(["scenedetect","-q", "-i", p, "detect-content", "--threshold", "20", "list-scenes", "-o", "data/scene", "-f", f"{id}.csv"])
            convert_video_to_frames(p)
            embedding = do_frame_embedding(model, "data/frames")
            for file in os.listdir("data/frames"):
                os.remove(os.path.join("data/frames",file))
        except:
            print(f"WARNING: video failed {id}")
            try:
                os.remove(p)
                os.remove(f"data/scene/{id}.csv")
            except:
                pass
        vid_csv = sponsor_data[sponsor_data["videoID"]==id]
        drop = []
        for i,row in vid_csv.iterrows():  # Drop all bad sponsor timestamps
            if row["votes"] < 0:
                drop.append(i)
                continue
            for _,row2 in vid_csv.iterrows():
                if row2["startTime"]<row["endTime"] and row2["endTime"]>row["startTime"]:
                    if row2["votes"] > row["votes"]:
                        drop.append(i)
                        break
        vid_csv = vid_csv.drop(index=drop)
        if embedding:
            embedding_to_annotate(f"data/embeddings/{id}.csv", embedding, vid_csv)