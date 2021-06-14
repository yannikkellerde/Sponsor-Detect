import pandas as pd
import os
from tqdm import tqdm
import random
import util

if __name__=="__main__":
    VAL_PERCENT = 20
    TEST_PERCENT = 10
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
    DATA_PATH = "data/embeddings"
    SAVE_PATH = "data/processed_videos"

    all_videos = os.listdir(DATA_PATH)
    random.shuffle(all_videos)

    train_val_index = int(len(all_videos)*(1-(VAL_PERCENT+TEST_PERCENT)/100))
    val_test_index = int(train_val_index+len(all_videos)*(VAL_PERCENT/100))
    train = all_videos[:train_val_index]
    val = all_videos[train_val_index:val_test_index]
    test = all_videos[val_test_index:]

    combos = ((train,"train.pkl"),(val,"val.pkl"),(test,"test.pkl"))
    for pkl_list,fname in tqdm(combos,desc="sets"):
        all_embeddings = []
        all_categorys = []
        for pkl_item in tqdm(pkl_list,desc="videos"):
            video_dic = util.load_obj(os.path.join(DATA_PATH, pkl_item))
            all_embeddings.append(video_dic["embedding"])
            clean_category = [CATEGORY_MAP[x] for x in video_dic["category"]]
            all_categorys.append(clean_category)
        big_dic = {'embeddings':all_embeddings, 'categorys':all_categorys}
        util.save_obj(big_dic, os.path.join(SAVE_PATH, fname))



