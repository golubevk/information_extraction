"""
data_preprocessing.py

Предобработка данных для задачи классификации рубрик по текстам отзывов.

Этапы:
1. Загрузка данных
2. Первичная очистка текста
3. Нормализация рубрик
4. Фильтрация данных
5. Сохранение предобработанных данных
"""

import pandas as pd
import numpy as np
import re
import pickle
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("ПРЕДОБРАБОТКА ДАННЫХ ДЛЯ КЛАССИФИКАЦИИ РУБРИК")
print("=" * 80)


# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 1: ЗАГРУЗКА ДАННЫХ")
print("=" * 80)

# Загрузка датасета
df = pd.read_csv("geo-reviews-dataset-2023.csv")

print(f"\nДатасет загружен успешно!")
print(f"Размер: {df.shape[0]:,} строк × {df.shape[1]} столбцов")
print(f"\nСтолбцы: {df.columns.tolist()}")

# Первичный осмотр данных
print(f"\n{'=' * 80}")
print("ПЕРВЫЕ 5 ЗАПИСЕЙ:")
print(df.head())

# Информация о типах данных и пропусках
print(f"\n{'=' * 80}")
print("ИНФОРМАЦИЯ О ДАННЫХ:")
print(df.info())

# Статистика пропусков
print(f"\n{'=' * 80}")
print("ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame(
    {
        "Столбец": missing.index,
        "Пропусков": missing.values,
        "Процент": missing_percent.values,
    }
)
print(missing_df[missing_df["Пропусков"] > 0])


# ============================================================================
# 2. ФУНКЦИИ ДЛЯ ПРЕДОБРАБОТКИ
# ============================================================================

print(f"\n{'=' * 80}")
print("ЭТАП 2: ОПРЕДЕЛЕНИЕ ФУНКЦИЙ ПРЕДОБРАБОТКИ")
print("=" * 80)


def normalize_rubrics(rubrics_str):
    """
    Нормализация рубрик: сортировка и дедупликация

    Args:
        rubrics_str: строка с рубриками, разделенными точкой с запятой

    Returns:
        нормализованная строка с рубриками
    """
    if pd.isna(rubrics_str):
        return rubrics_str

    # Разделяем по точке с запятой
    rubrics = str(rubrics_str).split(";")

    # Убираем пробелы, удаляем дубликаты и сортируем
    rubrics_cleaned = sorted(set(r.strip() for r in rubrics if r.strip()))

    # Собираем обратно
    return ";".join(rubrics_cleaned)


def get_rubrics_list(rubrics_str):
    """
    Преобразование строки рубрик в список

    Args:
        rubrics_str: строка с рубриками

    Returns:
        список рубрик
    """
    if pd.isna(rubrics_str):
        return []

    # Если это строковое представление списка
    if isinstance(rubrics_str, str) and rubrics_str.startswith("["):
        try:
            import ast

            parsed = ast.literal_eval(rubrics_str)
            return parsed if isinstance(parsed, list) else []
        except:
            pass

    # Если строка с разделителями
    return [r.strip() for r in str(rubrics_str).split(";") if r.strip()]


def clean_text(text):
    """
    Базовая очистка текста

    Args:
        text: исходный текст

    Returns:
        очищенный текст
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Удаляем URL
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Удаляем email
    text = re.sub(r"\S+@\S+", "", text)

    # Удаляем номера телефонов
    text = re.sub(r"\+?\d[\d\s\-\(\)]{7,}\d", "", text)

    # Убираем множественные пробелы и переносы
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


print("✓ Функции предобработки определены:")
print("  - normalize_rubrics(): нормализация рубрик")
print("  - get_rubrics_list(): преобразование в список")
print("  - clean_text(): очистка текста")


# ============================================================================
# 3. ОБРАБОТКА РУБРИК
# ============================================================================

print(f"\n{'=' * 80}")
print("ЭТАП 3: ОБРАБОТКА РУБРИК")
print("=" * 80)

df_processed = df.copy()

# Нормализация рубрик
print("\nНормализация рубрик...")
df_processed["rubrics_normalized"] = df_processed["rubrics"].apply(normalize_rubrics)

# Создание списков рубрик
print("Создание списков рубрик...")
df_processed["rubrics_list"] = df_processed["rubrics_normalized"].apply(
    get_rubrics_list
)

# Подсчет количества рубрик
df_processed["rubric_count"] = df_processed["rubrics_list"].apply(len)

# Статистика
print(f"\nСтатистика по количеству рубрик:")
print(df_processed["rubric_count"].value_counts().sort_index())

print(f"\nВсего записей: {len(df_processed):,}")
print(
    f"С одной рубрикой: {(df_processed['rubric_count'] == 1).sum():,} ({(df_processed['rubric_count'] == 1).sum() / len(df_processed) * 100:.1f}%)"
)
print(
    f"С множественными рубриками: {(df_processed['rubric_count'] > 1).sum():,} ({(df_processed['rubric_count'] > 1).sum() / len(df_processed) * 100:.1f}%)"
)


# ============================================================================
# 4. ОЧИСТКА ТЕКСТА
# ============================================================================

print(f"\n{'=' * 80}")
print("ЭТАП 4: ОЧИСТКА ТЕКСТА")
print("=" * 80)

print("\nОчистка текстов отзывов...")
df_processed["text_clean"] = df_processed["text"].apply(clean_text)

# Вычисление длины текста (в символах)
df_processed["text_length_chars"] = df_processed["text_clean"].str.len()

# Вычисление длины текста (в словах)
df_processed["text_length_words"] = df_processed["text_clean"].apply(
    lambda x: len(x.split()) if x else 0
)

# Статистика длины текста
print(f"\nСтатистика длины текста (символы):")
print(df_processed["text_length_chars"].describe())

print(f"\nСтатистика длины текста (слова):")
print(df_processed["text_length_words"].describe())


# ============================================================================
# 5. ФИЛЬТРАЦИЯ ДАННЫХ
# ============================================================================

print(f"\n{'=' * 80}")
print("ЭТАП 5: ФИЛЬТРАЦИЯ ДАННЫХ")
print("=" * 80)

initial_size = len(df_processed)

# Удаление записей без текста
print("\n1. Удаление пустых текстов...")
df_processed = df_processed[
    (df_processed["text_clean"].notna()) & (df_processed["text_clean"].str.len() > 0)
].copy()
removed = initial_size - len(df_processed)
print(f"   Удалено: {removed:,} записей")
print(f"   Осталось: {len(df_processed):,} записей")

# Удаление записей без рубрик
print("\n2. Удаление записей без рубрик...")
initial_size = len(df_processed)
df_processed = df_processed[df_processed["rubric_count"] > 0].copy()
removed = initial_size - len(df_processed)
print(f"   Удалено: {removed:,} записей")
print(f"   Осталось: {len(df_processed):,} записей")

# Удаление слишком коротких текстов (< 20 символов)
print("\n3. Удаление слишком коротких текстов (< 20 символов)...")
initial_size = len(df_processed)
df_processed = df_processed[df_processed["text_length_chars"] >= 20].copy()
removed = initial_size - len(df_processed)
print(f"   Удалено: {removed:,} записей")
print(f"   Осталось: {len(df_processed):,} записей")

# Удаление слишком длинных текстов (> 2000 символов)
print("\n4. Удаление слишком длинных текстов (> 2000 символов)...")
initial_size = len(df_processed)
df_processed = df_processed[df_processed["text_length_chars"] <= 2000].copy()
removed = initial_size - len(df_processed)
print(f"   Удалено: {removed:,} записей")
print(f"   Осталось: {len(df_processed):,} записей")


# ============================================================================
# 6. СОЗДАНИЕ ДОПОЛНИТЕЛЬНЫХ ПРИЗНАКОВ
# ============================================================================

print(f"\n{'=' * 80}")
print("ЭТАП 6: СОЗДАНИЕ ДОПОЛНИТЕЛЬНЫХ ПРИЗНАКОВ")
print("=" * 80)

# Флаг множественных рубрик
df_processed["is_multi_rubric"] = df_processed["rubric_count"] > 1

# Извлечение первой (главной) рубрики
df_processed["main_rubric"] = df_processed["rubrics_list"].apply(
    lambda x: x[0] if len(x) > 0 else None
)

# Количество слов
df_processed["word_count"] = df_processed["text_clean"].apply(lambda x: len(x.split()))

# Средняя длина слова
df_processed["avg_word_length"] = df_processed["text_clean"].apply(
    lambda x: np.mean([len(word) for word in x.split()])
    if x and len(x.split()) > 0
    else 0
)

# Количество уникальных слов
df_processed["unique_words"] = df_processed["text_clean"].apply(
    lambda x: len(set(x.split())) if x else 0
)

# Лексическое разнообразие (unique words / total words)
df_processed["lexical_diversity"] = df_processed.apply(
    lambda row: row["unique_words"] / row["word_count"] if row["word_count"] > 0 else 0,
    axis=1,
)

print("\n✓ Созданные признаки:")
print("  - is_multi_rubric: флаг множественных рубрик")
print("  - main_rubric: главная (первая) рубрика")
print("  - word_count: количество слов")
print("  - avg_word_length: средняя длина слова")
print("  - unique_words: количество уникальных слов")
print("  - lexical_diversity: лексическое разнообразие")


# ============================================================================
# 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================

print(f"\n{'=' * 80}")
print("ЭТАП 7: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 80)

# Сохранение в CSV
output_filename = "data_preprocessed.csv"
df_processed.to_csv(output_filename, index=False)
print(f"\n✓ Данные сохранены в CSV: {output_filename}")

# Сохранение метаданных
metadata = {
    "original_size": len(df),
    "processed_size": len(df_processed),
    "removed_records": len(df) - len(df_processed),
    "columns": df_processed.columns.tolist(),
    "single_label_records": (df_processed["rubric_count"] == 1).sum(),
    "multi_label_records": (df_processed["rubric_count"] > 1).sum(),
    "unique_rubrics": len(
        set([r for rubrics in df_processed["rubrics_list"] for r in rubrics])
    ),
    "preprocessing_date": datetime.now().isoformat(),
    "text_stats": {
        "min_length": int(df_processed["text_length_chars"].min()),
        "max_length": int(df_processed["text_length_chars"].max()),
        "mean_length": float(df_processed["text_length_chars"].mean()),
        "median_length": float(df_processed["text_length_chars"].median()),
    },
}

metadata_filename = "preprocessing_metadata.pkl"
with open(metadata_filename, "wb") as f:
    pickle.dump(metadata, f)
print(f"✓ Метаданные сохранены: {metadata_filename}")


# ============================================================================
# 8. ИТОГОВАЯ СТАТИСТИКА
# ============================================================================

print(f"\n{'=' * 80}")
print("ИТОГОВАЯ СТАТИСТИКА ПРЕДОБРАБОТКИ")
print("=" * 80)

summary = f"""
Исходный датасет:           {metadata["original_size"]:,} записей
Предобработанный датасет:   {metadata["processed_size"]:,} записей
Удалено:                    {metadata["removed_records"]:,} записей ({metadata["removed_records"] / metadata["original_size"] * 100:.1f}%)

Рубрики:
  - Уникальных рубрик:      {metadata["unique_rubrics"]:,}
  - Single-label записей:   {metadata["single_label_records"]:,} ({metadata["single_label_records"] / metadata["processed_size"] * 100:.1f}%)
  - Multi-label записей:    {metadata["multi_label_records"]:,} ({metadata["multi_label_records"] / metadata["processed_size"] * 100:.1f}%)

Длина текста (символы):
  - Минимум:                {metadata["text_stats"]["min_length"]}
  - Максимум:               {metadata["text_stats"]["max_length"]}
  - Среднее:                {metadata["text_stats"]["mean_length"]:.0f}
  - Медиана:                {metadata["text_stats"]["median_length"]:.0f}

Созданные файлы:
  ✓ {output_filename}
  ✓ {metadata_filename}

Следующий шаг: EDA (exploratory data analysis)
"""

print(summary)

# Сохранение итоговой статистики в текстовый файл
with open("preprocessing_summary.txt", "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("ОТЧЕТ О ПРЕДОБРАБОТКЕ ДАННЫХ\n")
    f.write("=" * 80 + "\n\n")
    f.write(summary)

print(f"\n✓ Итоговый отчет сохранен: preprocessing_summary.txt")

print("\n" + "=" * 80)
print("✅ ПРЕДОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
print("=" * 80)
