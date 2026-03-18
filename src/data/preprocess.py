# import pandas as pd

# path = cfg["paths"]["data_raw"]
# df = pd.read_csv(path+"/train_essays.csv")

from dotenv import load_dotenv
from src.utils.config_loader import cfg
from datasets import load_dataset
import os.path
import pandas as pd
import csv

external_path = cfg["paths"]["data_external"]
raw_path = cfg["paths"]["data_raw"]
hf_ds_name = "artem9k/ai-text-detection-pile"
hf_ds_path = external_path+ hf_ds_name.split("/")[1] + ".csv"
drcat_ds_path = external_path + "train_v2_drcat_02.csv"
train_essays_path = raw_path + "train_essays.csv"
merged_output_path = "data/processed.csv"

def load_hf_dataset():
    if not os.path.exists(hf_ds_path):
        print("Loading Hugging Face Dataset...")

        load_dotenv()
        
        shards = [
            "data/train-00000-of-00007-bc5952582e004d67.parquet",
            "data/train-00006-of-00007-3d8a471ba0cf1c8d.parquet" # L'ultimo
        ]


        print(f"\n--- Loading shards: {shards} ---")

        ds_human = load_dataset(hf_ds_name, data_files=shards[0], split='train', streaming=True) 
        df_human = pd.DataFrame(list(ds_human.take(50000)))

        ds_ai = load_dataset(hf_ds_name, data_files=shards[1], split='train', streaming=True)
        df_ai = pd.DataFrame(list(ds_ai.take(10000)))

        hf_df_10 = pd.concat([df_human, df_ai], axis=0, ignore_index=True)
        hf_df_10 = hf_df_10.sample(frac=1, random_state=42).reset_index(drop=True)    
        hf_df_10.to_csv(hf_ds_path, index=False)

        print(f"Dataset Loaded! Check {hf_ds_path}")
    else:
        print("Hugging Face Dataset Found!")
    return

def merge_ds():
    if not os.path.exists(merged_output_path):
        
        print("Loading HuggingFace dataset... ")
        hf_ds_df = pd.read_csv(hf_ds_path, usecols=["text", "source"]).reindex(columns=(["text", "source"]))
        hf_ds_df["source"] = hf_ds_df["source"].map(lambda x: 0 if x == 'human' else 1)
        hf_ds_df = hf_ds_df.rename(columns={"source": "generated"})
        print(hf_ds_df.head(1000))
        print(f"Human generated count: {len(hf_ds_df[hf_ds_df["generated"] == 0])}, AI generated count: {len(hf_ds_df[hf_ds_df["generated"] == 1])}")
        print("HuggingFace Dataset loaded!")

        print("Loading drcat v2 dataset...")
        drcat_ds_df = pd.read_csv(drcat_ds_path, usecols=["text", "label"])
        drcat_ds_df = drcat_ds_df.rename(columns={"label": "generated"})
        print(drcat_ds_df.head(1000))
        print(f"Human generated count: {len(drcat_ds_df[drcat_ds_df["generated"] == 0])},  AI generated count: {len(drcat_ds_df[drcat_ds_df["generated"] == 1])}")
        print("drcat v2 dataset loaded!")

        print("Loading base dataset...")
        train_essays_df = pd.read_csv(train_essays_path, usecols=["text", "generated"])
        print(train_essays_df.head(1000))
        print(f"Human generated count: {len(train_essays_df[train_essays_df["generated"] == 0])},  AI generated count: {len(train_essays_df[train_essays_df["generated"] == 1])}")
        print("Base train dataset loaded")

        print("Merging datasets...")
        final_df = pd.concat([hf_ds_df, drcat_ds_df, train_essays_df], axis=0)
        final_df = final_df.sample(frac = 1, random_state=42).reset_index(drop=True)
        print("Final dataset:")
        print(final_df.head(1000))
        print(f"Human generated count: {len(final_df[final_df["generated"] == 0])},  AI generated count: {len(final_df[final_df["generated"] == 1])}")
        final_df.to_csv(merged_output_path)
        print(f"Final dataset saved! check {merged_output_path}")
    
    else:
        print(f"Dataset already processed! check {merged_output_path}")

if __name__ == "__main__":
    load_hf_dataset()
    merge_ds()