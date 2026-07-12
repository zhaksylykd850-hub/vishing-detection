"""
train.py — Обучение классификаторов мошеннических сообщений
=============================================================
Модели:
  1. TF-IDF + Logistic Regression   (baseline, интерпретируемый)
  2. TF-IDF + SVM (LinearSVC)       (сильный baseline для текста)
  3. TF-IDF + XGBoost               (градиентный бустинг)
  4. TF-IDF + Random Forest         (ансамблевый метод)

Запуск:
  python train.py

Требования:
  pip install scikit-learn xgboost pandas numpy
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score,
    classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline

import xgboost as xgb

warnings.filterwarnings("ignore")

# ─── Пути к файлам ────────────────────────────────────────────────────────────
TRAIN_PATH = "train.csv"   # скачай с Claude
TEST_PATH  = "test.csv"    # скачай с Claude, не трогай до финала

# ─── Параметры ────────────────────────────────────────────────────────────────
N_FOLDS   = 5       # количество фолдов кросс-валидации
RANDOM_STATE = 42

# ─── TF-IDF конфигурация (общая для всех моделей) ────────────────────────────
TFIDF_PARAMS = dict(
    analyzer="word",
    ngram_range=(1, 2),      # унграммы + биграммы
    max_features=8000,
    sublinear_tf=True,       # log(TF) вместо TF — стандарт для текста
    min_df=2,                # игнорируем слова встречающиеся < 2 раз
    strip_accents="unicode",
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Загрузка данных
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path: str):
    df = pd.read_csv(path)
    X = df["text"].fillna("").values
    y = (df["label"] == "fraud").astype(int).values
    return X, y, df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Определение моделей
# ══════════════════════════════════════════════════════════════════════════════

def build_models():
    """
    Возвращает словарь {имя: Pipeline}.

    Обоснование выбора моделей для диссертации:
    ─────────────────────────────────────────────
    TF-IDF + Logistic Regression:
        Классический baseline для задач классификации текста.
        Wang & Manning (2012) показали, что LR с TF-IDF конкурирует
        с нейросетевыми моделями на коротких текстах. Высокая
        интерпретируемость — можно объяснить решение через веса признаков.

    TF-IDF + SVM (LinearSVC):
        Метод опорных векторов исторически показывает лучшие результаты
        на текстовых задачах в сравнении с другими ML-алгоритмами
        (Joachims, 1998; Zhang et al., 2008). Устойчив к
        высокоразмерным пространствам признаков (что типично для TF-IDF).

    TF-IDF + XGBoost:
        Градиентный бустинг позволяет улавливать нелинейные зависимости
        между TF-IDF признаками. Chen & Guestrin (2016) демонстрируют
        превосходство XGBoost над классическим ML на структурированных данных.
        Используется как мост между классическим ML и deep learning.

    TF-IDF + Random Forest:
        Ансамблевый метод (Breiman, 2001), устойчивый к переобучению.
        Важность признаков (feature importances) позволяет интерпретировать
        какие N-граммы наиболее сигнальны для детекции фрода.
    """
    models = {
        "LogReg": Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            )),
        ]),

        "SVM": Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            # LinearSVC не даёт predict_proba напрямую → оборачиваем
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
                cv=3,
            )),
        ]),

        "XGBoost": Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf", xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                verbosity=0,
            )),
        ]),

        "RandomForest": Pipeline([
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }
    return models


# ══════════════════════════════════════════════════════════════════════════════
# 3. Кросс-валидация
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate_model(name: str, pipeline, X, y, n_folds=5):
    """
    Стратифицированная k-fold кросс-валидация.

    Почему стратифицированная:
        При дисбалансе классов обычный KFold может создать фолды
        где один класс почти отсутствует. StratifiedKFold сохраняет
        пропорции классов в каждом фолде (Kohavi, 1995).

    Почему k=5:
        Стандартный выбор в литературе (Arlot & Celisse, 2010).
        Компромисс между смещением оценки (bias) и дисперсией (variance).
        k=10 даёт чуть меньше смещение, но дороже вычислительно.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred  = pipeline.predict(X_vl)
        y_proba = pipeline.predict_proba(X_vl)[:, 1]

        fold_results.append({
            "fold":      fold,
            "f1":        f1_score(y_vl, y_pred),
            "precision": precision_score(y_vl, y_pred),
            "recall":    recall_score(y_vl, y_pred),
            "roc_auc":   roc_auc_score(y_vl, y_proba),
            "accuracy":  accuracy_score(y_vl, y_pred),
        })

    fdf = pd.DataFrame(fold_results)

    summary = {}
    for col in ["f1", "precision", "recall", "roc_auc", "accuracy"]:
        summary[f"{col}_mean"] = fdf[col].mean()
        summary[f"{col}_std"]  = fdf[col].std()

    print(f"\n{'─'*55}")
    print(f"  {name}  —  {n_folds}-Fold CV")
    print(f"{'─'*55}")
    print(f"  {'Метрика':<12} {'Среднее':>8}  {'±std':>7}  Фолды")
    for col in ["f1", "precision", "recall", "roc_auc", "accuracy"]:
        vals = fdf[col].values
        folds_str = "  ".join(f"{v:.3f}" for v in vals)
        print(f"  {col:<12} {vals.mean():>8.4f}  {vals.std():>7.4f}  [{folds_str}]")

    return summary, fdf


# ══════════════════════════════════════════════════════════════════════════════
# 4. Финальная оценка на тест-сете
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_on_test(name: str, pipeline, X_train, y_train, X_test, y_test):
    """Обучаем на всём train, оцениваем на test — только один раз в конце."""
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"\n{'═'*55}")
    print(f"  {name}  —  TEST SET")
    print(f"{'═'*55}")
    print(classification_report(y_test, y_pred, target_names=["normal", "fraud"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion matrix:")
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    return {
        "f1":        f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
        "accuracy":  accuracy_score(y_test, y_pred),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Топ-признаки (интерпретируемость)
# ══════════════════════════════════════════════════════════════════════════════

def print_top_features(name: str, pipeline, top_n=15):
    """Печатает топ N-грамм для каждого класса (только для LogReg и RF)."""
    if name not in ("LogReg", "RandomForest"):
        return

    tfidf = pipeline.named_steps["tfidf"]
    clf   = pipeline.named_steps["clf"]
    vocab = np.array(tfidf.get_feature_names_out())

    if name == "LogReg":
        coef = clf.coef_[0]
        top_fraud  = vocab[np.argsort(coef)[-top_n:][::-1]]
        top_normal = vocab[np.argsort(coef)[:top_n]]
        print(f"\n  Топ признаки → FRAUD:  {', '.join(top_fraud)}")
        print(f"  Топ признаки → NORMAL: {', '.join(top_normal)}")

    elif name == "RandomForest":
        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[-top_n:][::-1]
        print(f"\n  Топ признаки (importance): {', '.join(vocab[top_idx])}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Главная функция
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  ДЕТЕКЦИЯ МОШЕННИЧЕСКИХ СООБЩЕНИЙ")
    print("  Сравнение классификаторов с кросс-валидацией")
    print("=" * 55)

    # Загрузка
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Файл не найден: {TRAIN_PATH}")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Файл не найден: {TEST_PATH}")

    X_train, y_train, df_train = load_data(TRAIN_PATH)
    X_test,  y_test,  df_test  = load_data(TEST_PATH)

    print(f"\n  Train: {len(X_train)} строк  "
          f"(fraud={y_train.sum()}, normal={(y_train==0).sum()})")
    print(f"  Test:  {len(X_test)} строк   "
          f"(fraud={y_test.sum()},  normal={(y_test==0).sum()})")

    models = build_models()

    cv_summary_all  = {}
    test_results_all = {}

    # ── Кросс-валидация на train ──────────────────────────────────────────────
    print("\n\n" + "═"*55)
    print("  КРОСС-ВАЛИДАЦИЯ (5-Fold Stratified KFold)")
    print("═"*55)

    for name, pipeline in models.items():
        summary, fold_df = cross_validate_model(name, pipeline, X_train, y_train, N_FOLDS)
        cv_summary_all[name] = summary
        fold_df["model"] = name
        fold_df.to_csv(f"cv_{name.lower()}_folds.csv", index=False)

    # ── Сводная таблица CV ────────────────────────────────────────────────────
    print("\n\n" + "═"*55)
    print("  СВОДНАЯ ТАБЛИЦА КРОСС-ВАЛИДАЦИИ")
    print("═"*55)
    print(f"  {'Модель':<15} {'F1':>8} {'±':>6} {'AUC':>8} {'±':>6} {'Prec':>8} {'Rec':>8}")
    for name, s in cv_summary_all.items():
        print(f"  {name:<15} "
              f"{s['f1_mean']:>8.4f} {s['f1_std']:>6.4f} "
              f"{s['roc_auc_mean']:>8.4f} {s['roc_auc_std']:>6.4f} "
              f"{s['precision_mean']:>8.4f} {s['recall_mean']:>8.4f}")

    # ── Финальная оценка на тест-сете ─────────────────────────────────────────
    print("\n\n" + "═"*55)
    print("  ФИНАЛЬНАЯ ОЦЕНКА НА TEST SET")
    print("  (обучение на полном train, однократная оценка)")
    print("═"*55)

    for name, pipeline in models.items():
        test_res = evaluate_on_test(
            name, pipeline, X_train, y_train, X_test, y_test
        )
        test_results_all[name] = test_res
        print_top_features(name, pipeline)

    # ── Финальная сводка ──────────────────────────────────────────────────────
    print("\n\n" + "═"*55)
    print("  ИТОГОВОЕ СРАВНЕНИЕ (TEST SET)")
    print("═"*55)
    print(f"  {'Модель':<15} {'F1':>8} {'AUC':>8} {'Prec':>8} {'Rec':>8} {'Acc':>8}")
    for name, r in test_results_all.items():
        print(f"  {name:<15} "
              f"{r['f1']:>8.4f} {r['roc_auc']:>8.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} {r['accuracy']:>8.4f}")

    # ── Сохранение результатов ────────────────────────────────────────────────
    all_results = {
        "cross_validation": cv_summary_all,
        "test_set": test_results_all,
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Сводный CSV для диссертации
    rows = []
    for name in models:
        cv = cv_summary_all[name]
        ts = test_results_all[name]
        rows.append({
            "model":          name,
            "cv_f1_mean":     cv["f1_mean"],
            "cv_f1_std":      cv["f1_std"],
            "cv_auc_mean":    cv["roc_auc_mean"],
            "cv_auc_std":     cv["roc_auc_std"],
            "cv_prec_mean":   cv["precision_mean"],
            "cv_rec_mean":    cv["recall_mean"],
            "test_f1":        ts["f1"],
            "test_auc":       ts["roc_auc"],
            "test_precision": ts["precision"],
            "test_recall":    ts["recall"],
            "test_accuracy":  ts["accuracy"],
        })
    pd.DataFrame(rows).to_csv("results_summary.csv", index=False)

    print("\n\n  Файлы сохранены:")
    print("  ├── results.json           — все результаты")
    print("  ├── results_summary.csv    — сводная таблица для диссертации")
    print("  ├── cv_logreg_folds.csv    — детали по фолдам LogReg")
    print("  ├── cv_svm_folds.csv       — детали по фолдам SVM")
    print("  ├── cv_xgboost_folds.csv   — детали по фолдам XGBoost")
    print("  └── cv_randomforest_folds.csv")


if __name__ == "__main__":
    main()
