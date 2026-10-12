import os
import glob
import pandas as pd



for path in glob.glob("result/*/*.csv"):


    df = pd.read_csv(path)

    # Make sure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    # Split point: first 93%, last 7
    split_idx = int(len(df) * 0.93)

    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    df_train.to_csv(path.replace('result' , 'train_result'),
                    index=False)
    df_test.to_csv(path.replace('result' , 'test_result'),
                   index=False)

