import re
import pandas as pd

INPUT_PATH  = "merged_dataset.csv"
OUTPUT_PATH = "text_label_preprocessed.csv"

# ─────────────────────────────────────────
# 1. ДЕРЕКТЕРДІ ОҚУ
# ─────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)
print(f"Бастапқы: {len(df)} жол")
print(f"Лейбл: {df['label'].value_counts().to_dict()}")

# ─────────────────────────────────────────
# 2. БОС ЖОЛДАРДЫ АЛУ
# ─────────────────────────────────────────
df = df[["text", "label"]].dropna()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
df = df[df["text"] != "nan"]
print(f"\nБос жолдар алынды: {len(df)} жол қалды")

# ─────────────────────────────────────────
# 3. ДУБЛИКАТТАРДЫ АЛУ
# ─────────────────────────────────────────
before = len(df)
df = df.drop_duplicates(subset="text").reset_index(drop=True)
print(f"Дублікаттар алынды: {before - len(df)} жол өшірілді → {len(df)} жол қалды")

# ─────────────────────────────────────────
# 4. МӘТІН ҰЗЫНДЫҒЫ БЕЛГІСІ
#    (ruBERT + Ensemble үшін қосымша белгі)
# ─────────────────────────────────────────
df["text_length"] = df["text"].apply(lambda t: len(t.split()))

# Өте қысқа мәтіндерді алып тастаймыз (1-2 сөз — мағынасыз)
before = len(df)
df = df[df["text_length"] >= 3].reset_index(drop=True)
print(f"Қысқа мәтіндер алынды (<3 сөз): {before - len(df)} жол өшірілді")

# ─────────────────────────────────────────
# 5. FRAUD МАРКЕР БЕЛГІСІ
#    Алаяқтыққа тән сөздер саны
# ─────────────────────────────────────────
FRAUD_MARKERS = [
    "заблокирован", "блокировка", "верификация", "подтвердите",
    "следователь", "прокурор", "полиция", "фсб", "мвд",
    "срочно", "немедленно", "переведите", "перевод",
    "код", "пароль", "cvv", "пин",
    "вирус", "anydesk", "teamviewer", "удалённый",
    "картаңыз", "блокталды", "аударым",
    "выиграли", "приз", "лотерея", "бесплатно",
    "кредит", "долг", "штраф", "арест"
]

def count_fraud_markers(text):
    text_lower = text.lower()
    return sum(1 for marker in FRAUD_MARKERS if marker in text_lower)

df["fraud_markers"] = df["text"].apply(count_fraud_markers)

# ─────────────────────────────────────────
# 6. СТОП-СӨЗДЕР АЛУ + МӘТІН ТАЗАЛАУ
# ─────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# ─────────────────────────────────────────
# 7. НӘТИЖЕНІ САҚТАУ
# ─────────────────────────────────────────
# Модель үшін тек text + label сақтаймыз
# (text_length және fraud_markers — анализ үшін ғана)
df_final = df[["text", "label"]].copy()
df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

# ─────────────────────────────────────────
# 8. СТАТИСТИКА
# ─────────────────────────────────────────
print(f"\n{'='*50}")
print(f"✅ Preprocessing аяқталды!")
print(f"{'='*50}")
print(f"Жалпы жол саны : {len(df_final)}")
print(f"Лейбл үлестірімі:\n{df_final['label'].value_counts()}")
print(f"\nМәтін ұзындығы (сөз саны):")
print(df.groupby("label")["text_length"].describe().round(1))
print(f"\nFraud маркерлері бар жолдар:")
print(f"  fraud  класы: {(df[df['label']=='fraud']['fraud_markers'] > 0).sum()} / {(df['label']=='fraud').sum()}")
print(f"  normal класы: {(df[df['label']=='normal']['fraud_markers'] > 0).sum()} / {(df['label']=='normal').sum()}")
print(f"\n✅ Сақталды: {OUTPUT_PATH}")
