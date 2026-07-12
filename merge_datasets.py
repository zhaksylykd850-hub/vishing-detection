import pandas as pd

# ─────────────────────────────────────────
# Файл жолдарын өзгерт
# ─────────────────────────────────────────
FILE_1 = "C:/Users/LOQ/PycharmProjects/MyNNProjectForMaster/new_datasets/merged_datasets.csv"
FILE_2 = "fraud_new_100.csv"

OUTPUT_PATH = "merged_dataset.csv"

# ─────────────────────────────────────────
# Оқу
# ─────────────────────────────────────────
df1 = pd.read_csv(FILE_1)
df2 = pd.read_csv(FILE_2)

print(f"{FILE_1}: {len(df1)} жол | колонкалар: {df1.columns.tolist()}")
print(f"{FILE_2}: {len(df2)} жол | колонкалар: {df2.columns.tolist()}")

# ─────────────────────────────────────────
# Біріктіру
# ─────────────────────────────────────────
merged = pd.concat([df1, df2], ignore_index=True)
merged = merged.drop_duplicates(subset="text").reset_index(drop=True)

print(f"\nБіріктірілген: {len(merged)} жол")
print(f"Лейбл:\n{merged['label'].value_counts()}")

merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"\n✅ Сақталды: {OUTPUT_PATH}")