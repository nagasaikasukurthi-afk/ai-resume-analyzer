from huggingface_hub import hf_hub_download
import pandas as pd


REPO_ID = "yashpwr/resume-ner-training-data"

print("Downloading raw dataset file information...")

file_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="data/train-00000-of-00001.parquet",
    repo_type="dataset"
)

print("\nRaw file:")
print(file_path)

print("\nReading Parquet file...")

df = pd.read_parquet(file_path)

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== FIRST RECORD =====")
print(df.iloc[0].to_dict())