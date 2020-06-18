from os.path import join as jpath

import numpy as np
import pandas as pd

FILES = [
    "jigsaw-toxic-comment-train.csv",
    "jigsaw-unintended-bias-train.csv",
    "test.csv",
    "validation.csv"
]
INPUT_DIR = "data"
OUTPUT_DIR = "data/mlm_text/"


def save_df_to_txt(input_file):
    df = pd.read_csv(input_file)
    if "content" in df.columns:
        df = df.rename(columns={"content": "comment_text"})
    df["comment_text"] = df["comment_text"].str.strip('"\n')
    out_file = OUTPUT_DIR + input_file.replace(".csv", "-lm.txt").split("/")[-1]
    np.savetxt(out_file, df["comment_text"].values, fmt='%s')


def main():
    for file in FILES:
        print(f"Processing file: {file}")
        save_df_to_txt(jpath(INPUT_DIR, file))


if __name__ == "__main__":
    main()
