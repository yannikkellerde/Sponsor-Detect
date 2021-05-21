import pandas as pd
import os
from tqdm import tqdm

dfs = []
for f in tqdm(os.listdir("processed")):
    df = pd.read_csv(os.path.join("processed",f),index_col=0)
    df["video"] = ".".join(f.split(".")[:-1])
    dfs.append(df)

big_df = pd.concat(dfs)
big_df.to_csv("all_processed.csv")