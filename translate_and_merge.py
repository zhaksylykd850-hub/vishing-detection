"""
translate_and_merge.py
======================
Переводит SMS_fraud_Dataset.csv через Google Translate (бесплатно),
затем объединяет с твоим датасетом.

Установка:
    pip install deep-translator pandas tqdm

Запуск:
    # Только перевод
    python translate_and_merge.py --sms SMS_fraud_Dataset.csv

    # Перевод + объединение с твоим датасетом
    python translate_and_merge.py --sms SMS_fraud_Dataset.csv --mine my_dataset.csv

    # Тест на 50 строках
    python translate_and_merge.py --sms SMS_fraud_Dataset.csv --limit 50
"""

import argparse
import os
import time
import pandas as pd
from deep_translator import GoogleTranslator

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False

# =============================================================================
# АРГУМЕНТЫ
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--sms",    default="SMS_fraud_Dataset.csv")
parser.add_argument("--mine",   default=None,
                    help="Путь к своему датасету (опционально)")
parser.add_argument("--output", default="merged_dataset.csv")
parser.add_argument("--limit",  type=int, default=None,
                    help="Только первые N строк (для теста)")
parser.add_argument("--batch",  type=int, default=10,
                    help="Размер батча (по умолчанию 10)")
parser.add_argument("--delay",  type=float, default=0.5,
                    help="Пауза между батчами в секундах")
args = parser.parse_args()

# =============================================================================
# 1. ЗАГРУЗКА
# =============================================================================

print(f"\n[1/4] Загружаем {args.sms}...")
df = pd.read_csv(args.sms)
df.columns = df.columns.str.lower().str.strip()

col_map = {"message": "text", "sms": "text", "content": "text"}
for old, new in col_map.items():
    if old in df.columns:
        df = df.rename(columns={old: new})

df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].dropna(subset=["text"]).reset_index(drop=True)

if args.limit:
    df = df.head(args.limit)
    print(f"  Режим теста: первые {args.limit} строк")

print(f"  Строк:      {len(df)}")
print(f"  Вишинг (1): {df['label'].sum()}")
print(f"  Норма  (0): {(df['label']==0).sum()}")

# =============================================================================
# 2. ПЕРЕВОД
# =============================================================================

translator = GoogleTranslator(source="en", target="ru")

def translate_batch(texts: list) -> list:
    """
    Переводит батч строк через Google Translate.
    Объединяет через ||| , переводит одним запросом, разбивает обратно.
    Если не получается — переводит по одному.
    """
    SEP = " ||| "
    joined = SEP.join(t.replace(SEP, " ") for t in texts)

    if len(joined) > 4800:
        # Слишком длинно — по одному
        results = []
        for t in texts:
            try:
                results.append(translator.translate(t[:4800]) or t)
            except Exception:
                results.append(t)
            time.sleep(0.15)
        return results

    try:
        translated_joined = translator.translate(joined) or joined
    except Exception:
        # Fallback — по одному
        results = []
        for t in texts:
            try:
                results.append(translator.translate(t[:4800]) or t)
            except Exception:
                results.append(t)
            time.sleep(0.15)
        return results

    parts = translated_joined.split(SEP)
    if len(parts) != len(texts):
        # Разделитель не сохранился — по одному
        results = []
        for t in texts:
            try:
                results.append(translator.translate(t[:4800]) or t)
            except Exception:
                results.append(t)
            time.sleep(0.15)
        return results

    return parts


texts = df["text"].tolist()
total = len(texts)
est_min = round(total / args.batch * args.delay / 60, 1)

print(f"\n[2/4] Переводим {total} строк")
print(f"  батч={args.batch}, пауза={args.delay}s, ~{est_min} мин\n")

translated = []
errors = 0
indices = list(range(0, total, args.batch))

if TQDM:
    indices = tqdm(indices, desc="Перевод", unit="батч")

for i in indices:
    batch = texts[i: i + args.batch]
    try:
        result = translate_batch(batch)
        translated.extend(result)
    except Exception as e:
        if not TQDM:
            print(f"  [{i}/{total}] Ошибка: {e}")
        translated.extend(batch)
        errors += 1
        time.sleep(3)
        continue

    time.sleep(args.delay)

    if not TQDM and i % 200 == 0 and i > 0:
        print(f"  [{i}/{total}] {i/total*100:.0f}%")

print(f"\n  Готово: {len(translated)} строк, ошибок: {errors}")

# =============================================================================
# 3. СБОРКА
# =============================================================================

df["text_orig"] = df["text"]
df["text"]      = translated
df["source"]    = "sms_fraud_en_translated"
df["language"]  = "ru"

# Показываем примеры
print("\n  Примеры перевода:")
for _, row in df.sample(min(3, len(df)), random_state=1).iterrows():
    tag = "ВИШИНГ" if row["label"] == 1 else "норма "
    print(f"  [{tag}] EN: {row['text_orig'][:70]}")
    print(f"          RU: {row['text'][:70]}")
    print()

# =============================================================================
# 4. СВОЙ ДАТАСЕТ
# =============================================================================

df_mine = None
if args.mine and os.path.exists(args.mine):
    print(f"[3/4] Загружаем твой датасет: {args.mine}...")
    df_mine = pd.read_csv(args.mine)
    df_mine.columns = df_mine.columns.str.lower().str.strip()

    for old, new in col_map.items():
        if old in df_mine.columns:
            df_mine = df_mine.rename(columns={old: new})

    if "source"   not in df_mine.columns: df_mine["source"]   = "my_dataset"
    if "language" not in df_mine.columns: df_mine["language"] = "ru"

    df_mine = df_mine[["text", "label", "source", "language"]].copy()
    print(f"  Строк: {len(df_mine)}, вишинг: {df_mine['label'].sum()}")
else:
    print("[3/4] Свой датасет не указан.")

# =============================================================================
# 5. ОБЪЕДИНЕНИЕ И СОХРАНЕНИЕ
# =============================================================================

print("\n[4/4] Сохраняем...")

parts = [df[["text", "label", "source", "language"]]]
if df_mine is not None:
    parts.append(df_mine)

df_final = (
    pd.concat(parts, ignore_index=True)
    .drop_duplicates(subset=["text"])
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

print(f"\n{'='*50}")
print(f"ИТОГ")
print(f"{'='*50}")
print(f"Всего строк:  {len(df_final)}")
print(f"Вишинг (1):   {df_final['label'].sum()}")
print(f"Норма  (0):   {(df_final['label']==0).sum()}")
print(f"Доля вишинга: {df_final['label'].mean():.1%}")
if df_mine is not None:
    print(f"\nПо источникам:")
    print(df_final["source"].value_counts().to_string())

df_final.to_csv(args.output, index=False, encoding="utf-8-sig")
print(f"\n✓  Объединённый датасет: {args.output}")

sms_check = args.output.replace(".csv", "_with_originals.csv")
df[["text", "text_orig", "label", "source", "language"]].to_csv(
    sms_check, index=False, encoding="utf-8-sig"
)
print(f"✓  SMS с оригиналами:    {sms_check}")
