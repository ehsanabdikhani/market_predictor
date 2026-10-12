import os
import pandas as pd
import numpy as np

input_root_dir = 'result'
output_root_dir = 'Augmented_Result'

ignore_columns = ['Date']
label_columns = ['Potential_type', 'Movement_Signal']
date_column = 'Date'

target_unique_ratio = 0.1

label_mapping = {
    'oversold': 'overbought',
    'overbought': 'oversold',
    'Long': 'Short',
    'Short': 'Long',
}


def augment_rows(df_part):
    aug = df_part.copy()

    numeric_cols = aug.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ignore_columns:
            aug[col] = aug[col] * -1

    for col in label_columns:
        if col in aug.columns:
            aug[col] = aug[col].replace(label_mapping)

    return aug


def process_single_file(file_path, output_path):
    try:
        df = pd.read_csv(file_path)

        if date_column not in df.columns:
            raise ValueError(f"ستون '{date_column}' در فایل وجود ندارد")

        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        df = df.sort_values(by=date_column, ascending=False).reset_index(drop=True)

        N = len(df)

        x = int(round((0.9 / 1.1) * N))
        x = min(x, N)  # محافظت

        n_unique = N - x  # جدیدترها، فقط اصلی، بدون داپلیکیت

        # جدیدترها (بدون داپلیکیت)
        unique_part = df.iloc[:n_unique].copy()

        # قدیمی‌ترها (اصلی + augmented)
        to_dup_part = df.iloc[n_unique:].copy()
        to_dup_aug = augment_rows(to_dup_part)

        final_df = pd.concat(
            [unique_part, to_dup_part, to_dup_aug],
            ignore_index=True
        )

        final_df = final_df.sort_values(by=date_column, ascending=False).reset_index(drop=True)

        final_df.to_csv(output_path, index=False)

        total_out = len(final_df)
        print(f"OK: {os.path.basename(file_path)} | N={N} | "
              f"unique(no-dup)={n_unique} | duplicated_orig={len(to_dup_part)} | "
              f"final={total_out} | ratio={n_unique/total_out:.4f}")

    except Exception as e:
        print(f"ERROR on {file_path}: {e}")


def main():
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    for root, dirs, files in os.walk(input_root_dir):
        relative_path = os.path.relpath(root, input_root_dir)

        if relative_path == '.':
            current_output_dir = output_root_dir
        else:
            current_output_dir = os.path.join(output_root_dir, relative_path)

        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)

        for file in files:
            if file.endswith('.csv'):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(current_output_dir, file)
                process_single_file(input_file_path, output_file_path)


if __name__ == "__main__":
    main()