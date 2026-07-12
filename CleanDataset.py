import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


def clean_text(text):
    # 1. Lower case
    text = text.lower()

    # 2. Remove links
    text = re.sub(r'http\S+', '', text)

    # 3. Remove special characters and digits
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s]', '', text)

    # 4. Tokenize
    words = word_tokenize(text, language='russian')

    # 5. Remove stopwords
    stop_words = set(stopwords.words('russian'))
    filtered_words = [word for word in words if word not in stop_words]

    # 6. Join back
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text


INPUT_PATH = "new_datasets/text_label_mapped.csv"
OUTPUT_PATH = "text_label_cleaned.csv"

df = pd.read_csv(INPUT_PATH)
df['text'] = df['text'].astype(str).apply(clean_text)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"Готово! Обработано {len(df)} строк.")
print(f"Сохранено в: {OUTPUT_PATH}")
print(f"\nПример:")
print(df.head())