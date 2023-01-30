import os
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from preprocess_demo_esta import preprocess_demo


def main():
    path = r"C:\Users\Piotrek\Documents\Inzynierka\esta-v1.0\pnxenopoulos-esta-f684ccf\data"
    online_files = os.listdir(os.path.join(path, "online"))
    online_files = [os.path.join(path, "online", f) for f in online_files]
    lan_files = os.listdir(os.path.join(path, "lan"))
    lan_files = [os.path.join(path, "lan", f) for f in lan_files]
    demo_files = online_files + lan_files
    with Pool() as p:
        data = list(tqdm(p.imap(preprocess_demo, demo_files), total=len(demo_files)))

    frames = [el["frame"] for el in data]
    all_frames = pd.concat(frames)
    all_frames.to_parquet("data/ESTA_frames.parquet")
    del all_frames
    kills = [el["kills"] for el in data]
    all_kills = pd.concat(kills)
    all_kills.to_parquet("data/ESTA_kills.parquet")
    del all_kills
    rounds = [el["roundScore"] for el in data]
    all_rounds = pd.concat(rounds)
    all_rounds.to_parquet("data/ESTA_rounds.parquet")
    del all_rounds


if __name__ == "__main__":
    main()
