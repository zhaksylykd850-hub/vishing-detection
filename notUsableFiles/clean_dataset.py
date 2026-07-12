import pandas as pd
import re

# 1. Загрузите ваш CSV (укажите правильный путь)
# Если файл в папке content:
df = pd.read_csv("final_dataset.csv")

print(f"До очистки: {len(df)} записей")
print("Первые 3 строки:")
print(df.head(3))

# 2. Функция очистки
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Убираем лишние кавычки
    text = text.replace('"""', '').replace('""', '"')
    # Убираем ;;;;;
    text = re.sub(r';{2,}', '', text)
    # Убираем &amp; и другой HTML-мусор
    text = text.replace('&amp;', '&').replace('&quot;', '"')
    # Убираем лишние пробелы
    text = ' '.join(text.split())
    return text.strip().strip('"').strip("'")

# 3. Применяем очистку
df['text'] = df['text'].apply(clean_text)
if 'original_text' in df.columns:
    df['original_text'] = df['original_text'].apply(clean_text)

# 4. Удаляем дубликаты
df_clean = df.drop_duplicates(subset=['text'], keep='first')
print(f"\nПосле удаления дубликатов: {len(df_clean)} записей")
print(f"Удалено: {len(df) - len(df_clean)}")

# 5. Проверяем результат
print("\nПример очищенного текста:")
print(df_clean['text'].iloc[0][:200])

# 6. Сохраняем
df_clean.to_csv("final_dataset.csv", index=False, encoding='utf-8')
print("\n✅ Сохранено: fraud_dataset_clean.csv")