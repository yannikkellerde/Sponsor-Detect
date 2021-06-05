import pandas as pd
import os
from tqdm import tqdm

MAX_NUMBER_WORDS = 5000
UNKNOWN_TOKEN = "unknownunknown"
CATEGORY_MAP = {
    "video":"video",
    "sponsor":"sponsor",
    "intro":"video",
    "outro":"video",
    "interaction":"video",
    "selfpromo":"video",
    "music_offtopic":"video",
    "offtopic":"video"
}

dfs = []
for f in tqdm(os.listdir("processed")):
    df = pd.read_csv(os.path.join("processed",f),index_col=0)
    if len(df) > MAX_NUMBER_WORDS or len(df)<10:
        continue
    df["word"] = df["word"].fillna(UNKNOWN_TOKEN)
    df["video"] = ".".join(f.split(".")[:-1])
    dfs.append(df)

big_df = pd.concat(dfs)
big_df["category"] = [CATEGORY_MAP[x] for x in tqdm(big_df["category"])]
big_df.to_csv("all_processed.csv")