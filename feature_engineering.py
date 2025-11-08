"""
feature_engineering.py

Формирование признаков и таргета для обучения моделей.

Создает два набора данных:
1. Для ML модели (LogReg + TF-IDF)
2. Для DL модели (ruBERT)
"""

import pandas as pd
import numpy as np
import pickle
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("FEATURE ENGINEERING: ФОРМИРОВАНИЕ ПРИЗНАКОВ И ТАРГЕТА")
print("=" * 80)


# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 1: ЗАГРУЗКА ПРЕДОБРАБОТАННЫХ ДАННЫХ")
print("=" * 80)

df = pd.read_csv("data_preprocessed.csv")

print(f"\nДанные загружены успешно!")
print(f"Размер: {len(df):,} записей")

# Преобразуем rubrics_list обратно в список
import ast

df["rubrics_list"] = df["rubrics_list"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)


# ============================================================================
# 2. АНАЛИЗ И ВЫБОР СТРАТЕГИИ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 2: АНАЛИЗ БАЛАНСА КЛАССОВ")
print("=" * 80)

# Подсчет всех рубрик
all_rubrics = []
for rubrics in df["rubrics_list"]:
    all_rubrics.extend(rubrics)

rubric_counts = Counter(all_rubrics)

print(f"\nВсего уникальных рубрик: {len(rubric_counts):,}")
print(f"\nРаспределение по частоте:")

frequency_stats = {
    "Очень редкие (< 100)": len([c for c in rubric_counts.values() if c < 100]),
    "Редкие (100-500)": len([c for c in rubric_counts.values() if 100 <= c < 500]),
    "Средние (500-2000)": len([c for c in rubric_counts.values() if 500 <= c < 2000]),
    "Частые (>= 2000)": len([c for c in rubric_counts.values() if c >= 2000]),
}

for category, count in frequency_stats.items():
    percent = (count / len(rubric_counts)) * 100
    print(f"  {category:<30} {count:>5} ({percent:>5.1f}%)")

print(f"\n{'=' * 80}")
print("ВЫБРАННАЯ СТРАТЕГИЯ: SINGLE-LABEL КЛАССИФИКАЦИЯ")
print("=" * 80)
print("""
Обоснование:
  • 51.8% записей имеют множественные рубрики с шумом
  • Multi-label разметка содержит ошибки и непоследовательности
  • Single-label дает более чистые и надежные данные
  • Достаточно примеров для обучения (239,840 записей)
  • Проще интерпретировать и оценивать результаты
""")


# ============================================================================
# 3. ФИЛЬТРАЦИЯ: SINGLE-LABEL ЗАПИСИ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 3: ФИЛЬТРАЦИЯ SINGLE-LABEL ЗАПИСЕЙ")
print("=" * 80)

# Оставляем только записи с одной рубрикой
df_single = df[df["rubric_count"] == 1].copy()

print(f"Записей с одной рубрикой: {len(df_single):,}")

# Извлекаем единственную рубрику
df_single["rubric"] = df_single["rubrics_list"].apply(
    lambda x: x[0] if len(x) > 0 else None
)

# Удаляем возможные пропуски
df_single = df_single[df_single["rubric"].notna()].copy()

# Статистика
print(f"Уникальных рубрик: {df_single['rubric'].nunique()}")

print(f"\nТоп-20 категорий:")
top_rubrics = df_single["rubric"].value_counts().head(20)
for rubric, count in top_rubrics.items():
    print(f"  {rubric:<40} {count:>7,}")


# ============================================================================
# 4. СОЗДАНИЕ ДВУХ НАБОРОВ ДАННЫХ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 4: СОЗДАНИЕ НАБОРОВ ДАННЫХ ДЛЯ ML И DL")
print("=" * 80)


# ============================================================================
# 4.1. НАБОР ДЛЯ ML (LogReg + TF-IDF)
# ============================================================================

print("\n--- НАБОР 1: ДЛЯ ML МОДЕЛИ (LogReg + TF-IDF) ---")

# Конфигурация для ML
config_ml = {
    "min_text_length": 20,
    "max_text_length": 2000,
    "min_samples_per_class": 500,  # строгая фильтрация для стабильности
    "model_type": "ML",
    "test_size": 0.2,
    "random_state": 42,
}

print(f"\nКонфигурация ML:")
for key, value in config_ml.items():
    print(f"  {key}: {value}")

# Фильтрация редких классов
min_samples_ml = config_ml["min_samples_per_class"]
class_counts_ml = df_single["rubric"].value_counts()
valid_classes_ml = class_counts_ml[class_counts_ml >= min_samples_ml].index

df_ml = df_single[df_single["rubric"].isin(valid_classes_ml)].copy()

print(f"\nПосле фильтрации (>={min_samples_ml} примеров на класс):")
print(f"  Записей: {len(df_ml):,}")
print(f"  Классов: {df_ml['rubric'].nunique()}")

# Распределение классов
print(f"\nРаспределение классов (топ-10):")
for rubric, count in df_ml["rubric"].value_counts().head(10).items():
    percent = (count / len(df_ml)) * 100
    print(f"  {rubric:<40} {count:>6,} ({percent:>5.2f}%)")

# Label encoding для ML
le_ml = LabelEncoder()
df_ml["label"] = le_ml.fit_transform(df_ml["rubric"])

# Train/test split для ML
X_ml_texts = df_ml["text_clean"].values
y_ml_labels = df_ml["label"].values

# Стратифицированное разбиение
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
    X_ml_texts,
    y_ml_labels,
    test_size=config_ml["test_size"],
    random_state=config_ml["random_state"],
    stratify=y_ml_labels,
)

print(f"\nРазбиение на train/test:")
print(f"  Train: {len(X_train_ml):,} записей")
print(f"  Test:  {len(X_test_ml):,} записей")
print(f"  Соотношение: {config_ml['test_size'] * 100:.0f}% test")

# Сохранение данных для ML
data_ml = {
    "X_train": X_train_ml,
    "X_test": X_test_ml,
    "y_train": y_train_ml,
    "y_test": y_test_ml,
    "label_encoder": le_ml,
    "classes": le_ml.classes_,
    "n_classes": len(le_ml.classes_),
    "config": config_ml,
    "statistics": {
        "train_size": len(X_train_ml),
        "test_size": len(X_test_ml),
        "n_classes": len(le_ml.classes_),
        "avg_text_length_chars": float(df_ml["text_clean"].str.len().mean()),
        "avg_text_length_words": float(df_ml["word_count"].mean()),
        "class_distribution": class_counts_ml[valid_classes_ml].to_dict(),
    },
    "created_at": datetime.now().isoformat(),
}

with open("data_ml_preprocessed.pkl", "wb") as f:
    pickle.dump(data_ml, f)

print(f"\n✓ Данные для ML сохранены: data_ml_preprocessed.pkl")


# ============================================================================
# 4.2. НАБОР ДЛЯ DL (ruBERT)
# ============================================================================

print("\n--- НАБОР 2: ДЛЯ DL МОДЕЛИ (ruBERT) ---")

# Конфигурация для DL
config_dl = {
    "min_text_length": 20,
    "max_text_length": 2000,
    "min_samples_per_class": 100,  # менее строгая фильтрация (BERT нужно меньше данных)
    "model_type": "DL",
    "test_size": 0.2,
    "sample_size": None,  # None = использовать все данные, или 0.2 для быстрых экспериментов
    "random_state": 42,
}

print(f"\nКонфигурация DL:")
for key, value in config_dl.items():
    print(f"  {key}: {value}")

# Фильтрация редких классов
min_samples_dl = config_dl["min_samples_per_class"]
class_counts_dl = df_single["rubric"].value_counts()
valid_classes_dl = class_counts_dl[class_counts_dl >= min_samples_dl].index

df_dl = df_single[df_single["rubric"].isin(valid_classes_dl)].copy()

print(f"\nПосле фильтрации (>={min_samples_dl} примеров на класс):")
print(f"  Записей: {len(df_dl):,}")
print(f"  Классов: {df_dl['rubric'].nunique()}")

# Распределение классов
print(f"\nРаспределение классов (топ-10):")
for rubric, count in df_dl["rubric"].value_counts().head(10).items():
    percent = (count / len(df_dl)) * 100
    print(f"  {rubric:<40} {count:>6,} ({percent:>5.2f}%)")

# Label encoding для DL
le_dl = LabelEncoder()
df_dl["label"] = le_dl.fit_transform(df_dl["rubric"])

# Train/test split для DL
X_dl_texts = df_dl["text_clean"].values
y_dl_labels = df_dl["label"].values

# Стратифицированное разбиение
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
    X_dl_texts,
    y_dl_labels,
    test_size=config_dl["test_size"],
    random_state=config_dl["random_state"],
    stratify=y_dl_labels,
)

# Опциональное сэмплирование (для быстрых экспериментов)
sample_size = config_dl.get("sample_size")
if sample_size is not None and 0 < sample_size < 1.0:
    print(f"\nСоздание выборки ({sample_size * 100:.0f}%)...")
    X_train_dl, _, y_train_dl, _ = train_test_split(
        X_train_dl,
        y_train_dl,
        train_size=sample_size,
        random_state=config_dl["random_state"],
        stratify=y_train_dl,
    )

    X_test_dl, _, y_test_dl, _ = train_test_split(
        X_test_dl,
        y_test_dl,
        train_size=sample_size,
        random_state=config_dl["random_state"],
        stratify=y_test_dl,
    )

    print(f"  Train после сэмплирования: {len(X_train_dl):,}")
    print(f"  Test после сэмплирования: {len(X_test_dl):,}")

print(f"\nРазбиение на train/test:")
print(f"  Train: {len(X_train_dl):,} записей")
print(f"  Test:  {len(X_test_dl):,} записей")
print(f"  Соотношение: {config_dl['test_size'] * 100:.0f}% test")

# Сохранение данных для DL
data_dl = {
    "X_train": X_train_dl,
    "X_test": X_test_dl,
    "y_train": y_train_dl,
    "y_test": y_test_dl,
    "label_encoder": le_dl,
    "classes": le_dl.classes_,
    "n_classes": len(le_dl.classes_),
    "config": config_dl,
    "statistics": {
        "train_size": len(X_train_dl),
        "test_size": len(X_test_dl),
        "n_classes": len(le_dl.classes_),
        "avg_text_length_chars": float(df_dl["text_clean"].str.len().mean()),
        "avg_text_length_words": float(df_dl["word_count"].mean()),
        "class_distribution": class_counts_dl[valid_classes_dl].to_dict(),
    },
    "created_at": datetime.now().isoformat(),
}

with open("data_bert_preprocessed.pkl", "wb") as f:
    pickle.dump(data_dl, f)

print(f"\n✓ Данные для DL сохранены: data_bert_preprocessed.pkl")


# ============================================================================
# 5. ИТОГОВАЯ СТАТИСТИКА
# ============================================================================

print("\n" + "=" * 80)
print("ИТОГОВАЯ СТАТИСТИКА FEATURE ENGINEERING")
print("=" * 80)

summary = f"""
НАБОР 1: ML (LogReg + TF-IDF)
  Файл:                     data_ml_preprocessed.pkl
  Train размер:             {len(X_train_ml):,} записей
  Test размер:              {len(X_test_ml):,} записей
  Классов:                  {len(le_ml.classes_)}
  Min примеров на класс:    {min_samples_ml}
  Средняя длина текста:     {data_ml["statistics"]["avg_text_length_words"]:.1f} слов
  
  Топ-5 классов (по частоте в полном датасете):
"""

# Топ-5 для ML
for rubric, count in class_counts_ml[valid_classes_ml].head(5).items():
    summary += f"    {rubric:<40} {count:>7,}\n"

summary += f"""
НАБОР 2: DL (ruBERT)
  Файл:                     data_bert_preprocessed.pkl
  Train размер:             {len(X_train_dl):,} записей
  Test размер:              {len(X_test_dl):,} записей
  Классов:                  {len(le_dl.classes_)}
  Min примеров на класс:    {min_samples_dl}
  Sample size:              {"100%" if sample_size is None else f"{sample_size * 100:.0f}%"}
  Средняя длина текста:     {data_dl["statistics"]["avg_text_length_words"]:.1f} слов
  
  Топ-5 классов (по частоте в полном датасете):
"""

# Топ-5 для DL
for rubric, count in class_counts_dl[valid_classes_dl].head(5).items():
    summary += f"    {rubric:<40} {count:>7,}\n"

summary += f"""
РАЗЛИЧИЯ МЕЖДУ НАБОРАМИ:
  • ML использует более строгую фильтрацию (min_samples={min_samples_ml})
  • DL может работать с меньшим количеством примеров (min_samples={min_samples_dl})
  • ML: {len(le_ml.classes_)} классов, DL: {len(le_dl.classes_)} классов
  • Разница в классах: {len(le_dl.classes_) - len(le_ml.classes_)} дополнительных в DL

СОЗДАННЫЕ ФАЙЛЫ:
  ✓ data_ml_preprocessed.pkl         - данные для ML модели
  ✓ data_bert_preprocessed.pkl       - данные для DL модели
  ✓ feature_engineering_summary.txt  - текстовый отчет

СЛЕДУЮЩИЙ ШАГ:
  Обучение моделей:
  1. ml_logreg.py    - Logistic Regression + TF-IDF
  2. dl_rubert.py    - ruBERT fine-tuning
"""

print(summary)

# Сохранение отчета
with open("feature_engineering_summary.txt", "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("ОТЧЕТ ПО FEATURE ENGINEERING\n")
    f.write("=" * 80 + "\n\n")
    f.write(summary)

print("✓ Текстовый отчет сохранен: feature_engineering_summary.txt")

# Сохранение метаданных для последующего анализа
metadata = {
    "ml": {
        "n_classes": len(le_ml.classes_),
        "train_size": len(X_train_ml),
        "test_size": len(X_test_ml),
        "min_samples": min_samples_ml,
        "classes": le_ml.classes_.tolist(),
    },
    "dl": {
        "n_classes": len(le_dl.classes_),
        "train_size": len(X_train_dl),
        "test_size": len(X_test_dl),
        "min_samples": min_samples_dl,
        "classes": le_dl.classes_.tolist(),
    },
}

with open("feature_engineering_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✓ Метаданные сохранены: feature_engineering_metadata.pkl")

print("\n" + "=" * 80)
print("✅ FEATURE ENGINEERING ЗАВЕРШЕН УСПЕШНО!")
print("=" * 80)
