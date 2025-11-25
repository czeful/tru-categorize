import pandas as pd

df = pd.read_parquet("data/processed/pseudo_labels_v1.parquet")
print(df.columns)
print(df.head())
