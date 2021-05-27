import pandas as pd
import os
import random
import shutil
import numpy as np
from tqdm import tqdm

VAL_PERCENT = 20
TEST_PERCENT = 10
SOURCE = "all_processed.csv"
TARG_FOLDER = "sponsor_nlp_data"
GROUP_COL = "video"
KEEP_COLS = ["word","category"]

shutil.rmtree(TARG_FOLDER)
os.makedirs(TARG_FOLDER)
df = pd.read_csv(SOURCE,index_col=0)
vids = list(df[GROUP_COL].unique())

random.shuffle(vids)
train_val_index = int(len(vids)*(1-(VAL_PERCENT+TEST_PERCENT)/100))
val_test_index = int(train_val_index+len(vids)*(VAL_PERCENT/100))

train = df[df[GROUP_COL].isin(vids[:train_val_index])]
val = df[df[GROUP_COL].isin(vids[train_val_index:val_test_index])]
test = df[df[GROUP_COL].isin(vids[val_test_index:])]

combos = ((train,"train.tsv"),(val,"val.tsv"),(test,"test.tsv"))

for data_df,fname in tqdm(combos,desc="sets"):
    vids = data_df[GROUP_COL].unique()
    for vid in tqdm(vids,desc="videos"):
        vid_df = data_df[data_df[GROUP_COL]==vid][KEEP_COLS]
        vid_df.to_csv(os.path.join(TARG_FOLDER,fname),mode="a",header=False,index=False,sep="\t")
        with open(os.path.join(TARG_FOLDER,fname),"a") as f:
            f.write("\n")