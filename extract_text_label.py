import pandas as pd

# Укажи пути к двум файлам
FILE_1 = "new_datasets/text_label_only.csv"
FILE_2 = "new_datasets/text_label_mapped.csv"

OUTPUT_PATH = "new_datasets/merged_datasets.csv"

df1 = pd.read_csv(FILE_1)
df2 = pd.read_csv(FILE_2)

print(f"{FILE_1}: {len(df1)} строк")
print(f"{FILE_2}: {len(df2)} строк")

merged = pd.concat([df1, df2], ignore_index=True)

merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"\nИтого строк: {len(merged)}")
print(f"Сохранено в: {OUTPUT_PATH}")