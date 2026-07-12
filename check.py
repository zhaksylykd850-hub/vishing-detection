import pandas as pd

df = pd.read_csv("text_label_preprocessed.csv")
df = df[["text", "label"]].dropna()
df["text"] = df["text"].astype(str).str.strip()
print(f"   Жалпы жол саны : {len(df)}")
print(f"   Лейбл үлестірімі:\n{df['label'].value_counts()}")