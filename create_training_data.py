""" Requires youtube-dl """
import pandas as pd
import os,sys
import subprocess
import time
import random
import re
from datetime import datetime,timedelta
from tqdm import tqdm

MAX_NUMBER_WORDS = 5000

def parse_time_string(tstr):
    t = datetime.strptime(tstr,"%H:%M:%S.%f")
    delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)
    return delta.total_seconds()

sponsor_data = pd.read_csv("data/sponsor_timestamps.csv")
vid_ids = list(sponsor_data["videoID"].unique())
random.shuffle(vid_ids)

vid_ids = [x.split(".")[0] for x in os.listdir("data/transcripts")]  #

print(len(vid_ids))
no_sponsors = 0

for id in tqdm(vid_ids):
    p = f"data/transcripts/{id}"
    proc_path = os.path.join("data/processed",id+".csv")
    if not os.path.isfile(p+".en.vtt"):
        try:
            pass
            #subprocess.call(["torsocks","-i", "youtube-dl", "--skip-download", "--write-auto-sub", f"https://www.youtube.com/watch?v={id}","-o", p])
        except:
            print(f"WARNING: youtube-dl failed {id}")
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
    vid_csv = vid_csv.drop(index=drop).sort_values("startTime")
    if len(vid_csv) == 0:
        continue

    no_sponsor = True  #

    if os.path.isfile(p+".en.vtt") and not os.path.isfile(proc_path):
        training_data = []
        reg_time = r'[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{3} --> [0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{3}'
        curr_i = 0
        with open(p+".en.vtt") as f:
            content = f.read()
            c_count = content.count("<c>")
            if c_count == 0:
                continue
            slash_c_count = content.count("</c>\n")
            if c_count + slash_c_count > MAX_NUMBER_WORDS:
                continue

            lines = content.splitlines()
            for line in lines:
                if "<c>" in line:
                    parts = [x.replace(">","").split("<") for x in line.replace("</c>","").split("<c>")]
                    parts = [(x[0],parse_time_string(x[1])) if len(x)==2 else (x[0],end_time) for x in parts]
                    for wt in parts:
                        entry = {}
                        entry["word"] = wt[0].strip()
                        entry["start"] = start_time
                        entry["end"] = wt[1]
                        entry["category"] = "video"
                        while len(vid_csv)>curr_i and entry["start"]>vid_csv.iloc[curr_i]["endTime"]:
                            curr_i += 1
                        if len(vid_csv)>curr_i:
                            row = vid_csv.iloc[curr_i]
                            if entry["end"]>row["startTime"]:
                                entry["category"] = row["category"]
                                if entry["category"] == "sponsor": #
                                    no_sponsor = False #
                        start_time = wt[1]
                        training_data.append(entry)
                else:
                    searched = re.search(reg_time,line)
                    if searched is not None:
                        s,e = searched.group(0).split(" --> ")
                        start_time = parse_time_string(s)
                        end_time = parse_time_string(e)
        df = pd.DataFrame(training_data)
        df.to_csv(proc_path)
    else:
        print(f"WARNING: Transcript download failed {id}")
    no_sponsors += int(no_sponsor)

print(f"{no_sponsors} sponsor free videos. This is {(no_sponsor/len(vid_ids))*100} percent")