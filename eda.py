"""
eda.py

Exploratory Data Analysis (EDA) для задачи классификации рубрик.

Анализ включает:
1. Распределение таргета (рубрик)
2. Анализ длины текстов
3. Корреляция между длиной отзыва и рейтингом
4. Частотные слова и n-граммы
5. Облако слов
6. Визуализации
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import pickle
import warnings

warnings.filterwarnings("ignore")

# Настройка стиля графиков
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)


# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 1: ЗАГРУЗКА ПРЕДОБРАБОТАННЫХ ДАННЫХ")
print("=" * 80)

# Загрузка данных
df = pd.read_csv("data_preprocessed.csv")

# Загрузка метаданных
with open("preprocessing_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(f"\nДанные загружены успешно!")
print(f"Размер: {len(df):,} записей")
print(f"Столбцов: {len(df.columns)}")

# Преобразуем rubrics_list обратно в список
import ast

df["rubrics_list"] = df["rubrics_list"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)


# ============================================================================
# 2. РАСПРЕДЕЛЕНИЕ ТАРГЕТА (РУБРИК)
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 2: АНАЛИЗ РАСПРЕДЕЛЕНИЯ РУБРИК")
print("=" * 80)

# Подсчет всех рубрик
all_rubrics = []
for rubrics in df["rubrics_list"]:
    all_rubrics.extend(rubrics)

rubric_counts = Counter(all_rubrics)
print(f"\nВсего уникальных рубрик: {len(rubric_counts)}")
print(f"Общее количество рубрик (с повторами): {len(all_rubrics):,}")

# Топ-30 рубрик
print(f"\n{'=' * 80}")
print("ТОП-30 САМЫХ ЧАСТЫХ РУБРИК:")
print(f"{'=' * 80}")
print(f"{'Рубрика':<50} {'Количество':>10} {'Процент':>10}")
print("-" * 80)

top_30_rubrics = rubric_counts.most_common(30)
for rubric, count in top_30_rubrics:
    percent = (count / len(all_rubrics)) * 100
    print(f"{rubric:<50} {count:>10,} {percent:>9.2f}%")

# Визуализация топ-20 рубрик
fig, ax = plt.subplots(figsize=(14, 8))
top_20 = rubric_counts.most_common(20)
rubrics_names = [r[0] for r in top_20]
rubrics_values = [r[1] for r in top_20]

bars = ax.barh(
    rubrics_names, rubrics_values, color="skyblue", edgecolor="navy", alpha=0.7
)
ax.set_xlabel("Количество записей", fontsize=12, fontweight="bold")
ax.set_ylabel("Рубрика", fontsize=12, fontweight="bold")
ax.set_title("Топ-20 самых частых рубрик", fontsize=14, fontweight="bold", pad=20)
ax.invert_yaxis()

# Добавляем значения на столбцы
for i, (bar, value) in enumerate(zip(bars, rubrics_values)):
    ax.text(value, i, f" {value:,}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("eda_top20_rubrics.png", dpi=300, bbox_inches="tight")
print(f"\n✓ График сохранен: eda_top20_rubrics.png")
plt.close()

# Распределение по частоте встречаемости
print(f"\n{'=' * 80}")
print("РАСПРЕДЕЛЕНИЕ ПО ЧАСТОТЕ ВСТРЕЧАЕМОСТИ:")
print(f"{'=' * 80}")

frequency_bins = {
    "Очень редкие (1-10)": len([c for c in rubric_counts.values() if 1 <= c <= 10]),
    "Редкие (11-50)": len([c for c in rubric_counts.values() if 11 <= c <= 50]),
    "Средние (51-200)": len([c for c in rubric_counts.values() if 51 <= c <= 200]),
    "Частые (201-1000)": len([c for c in rubric_counts.values() if 201 <= c <= 1000]),
    "Очень частые (>1000)": len([c for c in rubric_counts.values() if c > 1000]),
}

for category, count in frequency_bins.items():
    percent = (count / len(rubric_counts)) * 100
    print(f"{category:<30} {count:>5} рубрик ({percent:>5.1f}%)")

# Pie chart распределения
fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#ff9999", "#ffcc99", "#ffff99", "#99ccff", "#99ff99"]
wedges, texts, autotexts = ax.pie(
    frequency_bins.values(),
    labels=frequency_bins.keys(),
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
    textprops={"fontsize": 11},
)

ax.set_title(
    "Распределение рубрик по частоте встречаемости",
    fontsize=14,
    fontweight="bold",
    pad=20,
)

for autotext in autotexts:
    autotext.set_color("black")
    autotext.set_fontweight("bold")

plt.tight_layout()
plt.savefig("eda_rubrics_frequency_distribution.png", dpi=300, bbox_inches="tight")
print(f"✓ График сохранен: eda_rubrics_frequency_distribution.png")
plt.close()


# ============================================================================
# 3. АНАЛИЗ ДЛИНЫ ТЕКСТОВ
# ============================================================================

print(f"\n{'=' * 80}")
print("ЭТАП 3: АНАЛИЗ ДЛИНЫ ТЕКСТОВ")
print("=" * 80)

print(f"\nСтатистика длины текста (символы):")
print(df["text_length_chars"].describe())

print(f"\nСтатистика длины текста (слова):")
print(df["word_count"].describe())

# Гистограмма длины текста
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Длина в символах
axes[0, 0].hist(
    df["text_length_chars"], bins=50, color="steelblue", edgecolor="black", alpha=0.7
)
axes[0, 0].set_xlabel("Длина текста (символы)", fontweight="bold")
axes[0, 0].set_ylabel("Количество", fontweight="bold")
axes[0, 0].set_title("Распределение длины текста (символы)", fontweight="bold")
axes[0, 0].axvline(
    df["text_length_chars"].median(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Медиана: {df['text_length_chars'].median():.0f}",
)
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Длина в словах
axes[0, 1].hist(df["word_count"], bins=50, color="coral", edgecolor="black", alpha=0.7)
axes[0, 1].set_xlabel("Длина текста (слова)", fontweight="bold")
axes[0, 1].set_ylabel("Количество", fontweight="bold")
axes[0, 1].set_title("Распределение длины текста (слова)", fontweight="bold")
axes[0, 1].axvline(
    df["word_count"].median(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Медиана: {df['word_count'].median():.0f}",
)
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Средняя длина слова
axes[1, 0].hist(
    df["avg_word_length"], bins=50, color="lightgreen", edgecolor="black", alpha=0.7
)
axes[1, 0].set_xlabel("Средняя длина слова", fontweight="bold")
axes[1, 0].set_ylabel("Количество", fontweight="bold")
axes[1, 0].set_title("Распределение средней длины слова", fontweight="bold")
axes[1, 0].axvline(
    df["avg_word_length"].median(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Медиана: {df['avg_word_length'].median():.2f}",
)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Лексическое разнообразие
axes[1, 1].hist(
    df["lexical_diversity"], bins=50, color="plum", edgecolor="black", alpha=0.7
)
axes[1, 1].set_xlabel("Лексическое разнообразие", fontweight="bold")
axes[1, 1].set_ylabel("Количество", fontweight="bold")
axes[1, 1].set_title("Распределение лексического разнообразия", fontweight="bold")
axes[1, 1].axvline(
    df["lexical_diversity"].median(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Медиана: {df['lexical_diversity'].median():.2f}",
)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("eda_text_length_distributions.png", dpi=300, bbox_inches="tight")
print(f"\n✓ График сохранен: eda_text_length_distributions.png")
plt.close()


# ============================================================================
# 4. КОРРЕЛЯЦИЯ МЕЖДУ ДЛИНОЙ ОТЗЫВА И РЕЙТИНГОМ
# ============================================================================

print(f"\n{'=' * 80}")
print("ЭТАП 4: КОРРЕЛЯЦИЯ ДЛИНЫ ОТЗЫВА И РЕЙТИНГА")
print("=" * 80)

# Проверяем наличие столбца rating
if "rating" in df.columns:
    # Удаляем пропуски в рейтинге
    df_rating = df[df["rating"].notna()].copy()

    print(f"\nЗаписей с рейтингом: {len(df_rating):,}")
    print(f"\nСтатистика рейтинга:")
    print(df_rating["rating"].describe())

    # Корреляция
    correlation_chars = df_rating["text_length_chars"].corr(df_rating["rating"])
    correlation_words = df_rating["word_count"].corr(df_rating["rating"])

    print(f"\nКорреляция Пирсона:")
    print(f"  Длина текста (символы) ↔ Рейтинг: {correlation_chars:.4f}")
    print(f"  Длина текста (слова) ↔ Рейтинг:   {correlation_words:.4f}")

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(
        df_rating["word_count"], df_rating["rating"], alpha=0.1, s=1, color="navy"
    )
    axes[0].set_xlabel("Длина текста (слова)", fontweight="bold")
    axes[0].set_ylabel("Рейтинг", fontweight="bold")
    axes[0].set_title(f"Корреляция: {correlation_words:.4f}", fontweight="bold")
    axes[0].grid(alpha=0.3)

    # Boxplot: длина текста по рейтингам
    df_rating.boxplot(column="word_count", by="rating", ax=axes[1])
    axes[1].set_xlabel("Рейтинг", fontweight="bold")
    axes[1].set_ylabel("Длина текста (слова)", fontweight="bold")
    axes[1].set_title("Длина текста по рейтингам", fontweight="bold")
    axes[1].get_figure().suptitle("")  # убираем автоматический заголовок

    plt.tight_layout()
    plt.savefig("eda_rating_correlation.png", dpi=300, bbox_inches="tight")
    print(f"✓ График сохранен: eda_rating_correlation.png")
    plt.close()
else:
    print("\n⚠ Столбец 'rating' не найден в данных")


# ============================================================================
# 5. ЧАСТОТНЫЕ СЛОВА И N-ГРАММЫ
# ============================================================================

print(f"\n{'=' * 80}")
print("ЭТАП 5: АНАЛИЗ ЧАСТОТНЫХ СЛОВ И N-ГРАММ")
print("=" * 80)

# Объединяем все тексты
all_texts = " ".join(df["text_clean"].astype(str))
words = all_texts.split()

print(f"\nВсего слов: {len(words):,}")
print(f"Уникальных слов: {len(set(words)):,}")

# Топ-30 слов
word_counts = Counter(words)
print(f"\n{'=' * 80}")
print("ТОП-30 САМЫХ ЧАСТЫХ СЛОВ:")
print(f"{'=' * 80}")
print(f"{'Слово':<30} {'Количество':>15} {'Процент':>10}")
print("-" * 80)

for word, count in word_counts.most_common(30):
    percent = (count / len(words)) * 100
    print(f"{word:<30} {count:>15,} {percent:>9.2f}%")

# Визуализация топ-20 слов
fig, ax = plt.subplots(figsize=(12, 8))
top_20_words = word_counts.most_common(20)
words_names = [w[0] for w in top_20_words]
words_values = [w[1] for w in top_20_words]

bars = ax.barh(
    words_names, words_values, color="lightcoral", edgecolor="darkred", alpha=0.7
)
ax.set_xlabel("Количество", fontsize=12, fontweight="bold")
ax.set_ylabel("Слово", fontsize=12, fontweight="bold")
ax.set_title("Топ-20 самых частых слов", fontsize=14, fontweight="bold", pad=20)
ax.invert_yaxis()

for i, (bar, value) in enumerate(zip(bars, words_values)):
    ax.text(value, i, f" {value:,}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("eda_top20_words.png", dpi=300, bbox_inches="tight")
print(f"\n✓ График сохранен: eda_top20_words.png")
plt.close()

# Биграммы
print(f"\n{'=' * 80}")
print("ТОП-20 БИГРАММ:")
print(f"{'=' * 80}")

from itertools import islice


def generate_ngrams(text, n=2):
    """Генерация n-грамм"""
    words = text.split()
    ngrams = zip(*[islice(words, i, None) for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


# Собираем биграммы
all_bigrams = []
for text in df["text_clean"].astype(str):
    all_bigrams.extend(generate_ngrams(text, n=2))

bigram_counts = Counter(all_bigrams)

print(f"{'Биграмма':<50} {'Количество':>15}")
print("-" * 80)
for bigram, count in bigram_counts.most_common(20):
    print(f"{bigram:<50} {count:>15,}")

# Триграммы
print(f"\n{'=' * 80}")
print("ТОП-20 ТРИГРАММ:")
print(f"{'=' * 80}")

all_trigrams = []
for text in df["text_clean"].astype(str):
    all_trigrams.extend(generate_ngrams(text, n=3))

trigram_counts = Counter(all_trigrams)

print(f"{'Триграмма':<50} {'Количество':>15}")
print("-" * 80)
for trigram, count in trigram_counts.most_common(20):
    print(f"{trigram:<50} {count:>15,}")


# ============================================================================
# 6. ОБЛАКО СЛОВ
# ============================================================================

print(f"\n{'=' * 80}")
print("ЭТАП 6: СОЗДАНИЕ ОБЛАКА СЛОВ")
print("=" * 80)

# Общее облако слов
print("\nСоздание облака слов...")
wordcloud = WordCloud(
    width=1600,
    height=800,
    background_color="white",
    colormap="viridis",
    max_words=200,
    relative_scaling=0.5,
    min_font_size=10,
).generate(all_texts)

plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Облако слов (все отзывы)", fontsize=16, fontweight="bold", pad=20)
plt.tight_layout(pad=0)
plt.savefig("eda_wordcloud_all.png", dpi=300, bbox_inches="tight")
print("✓ Облако слов сохранено: eda_wordcloud_all.png")
plt.close()

# Облака слов для топ-5 рубрик
print("\nСоздание облаков слов для топ-5 рубрик...")
top_5_rubrics = [r[0] for r in rubric_counts.most_common(5)]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, rubric in enumerate(top_5_rubrics):
    # Фильтруем тексты для этой рубрики
    rubric_texts = df[df["rubrics_list"].apply(lambda x: rubric in x)]["text_clean"]
    rubric_text_combined = " ".join(rubric_texts.astype(str))

    if rubric_text_combined:
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="tab10",
            max_words=100,
        ).generate(rubric_text_combined)

        axes[idx].imshow(wc, interpolation="bilinear")
        axes[idx].axis("off")
        axes[idx].set_title(
            f"{rubric}\n({rubric_counts[rubric]:,} записей)",
            fontsize=12,
            fontweight="bold",
        )

# Убираем лишнюю ось
axes[5].axis("off")

plt.tight_layout()
plt.savefig("eda_wordcloud_top5_rubrics.png", dpi=300, bbox_inches="tight")
print("✓ Облака слов для топ-5 рубрик сохранены: eda_wordcloud_top5_rubrics.png")
plt.close()


# ============================================================================
# 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ EDA
# ============================================================================

print(f"\n{'=' * 80}")
print("ЭТАП 7: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ EDA")
print("=" * 80)

eda_results = {
    "total_records": len(df),
    "unique_rubrics": len(rubric_counts),
    "top_30_rubrics": top_30_rubrics,
    "frequency_distribution": frequency_bins,
    "text_stats": {
        "mean_length_chars": float(df["text_length_chars"].mean()),
        "median_length_chars": float(df["text_length_chars"].median()),
        "mean_length_words": float(df["word_count"].mean()),
        "median_length_words": float(df["word_count"].median()),
    },
    "top_30_words": word_counts.most_common(30),
    "top_20_bigrams": bigram_counts.most_common(20),
    "top_20_trigrams": trigram_counts.most_common(20),
}

if "rating" in df.columns:
    eda_results["rating_correlation"] = {
        "correlation_chars": float(correlation_chars),
        "correlation_words": float(correlation_words),
    }

# Сохранение
with open("eda_results.pkl", "wb") as f:
    pickle.dump(eda_results, f)

print("\n✓ Результаты EDA сохранены: eda_results.pkl")

# Создание текстового отчета
with open("eda_summary.txt", "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("ОТЧЕТ ПО EXPLORATORY DATA ANALYSIS\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Всего записей: {len(df):,}\n")
    f.write(f"Уникальных рубрик: {len(rubric_counts):,}\n\n")

    f.write("Топ-10 рубрик:\n")
    for rubric, count in top_30_rubrics[:10]:
        f.write(f"  {rubric:<40} {count:>10,}\n")

    f.write(f"\nСредняя длина текста: {df['word_count'].mean():.1f} слов\n")
    f.write(f"Медианная длина текста: {df['word_count'].median():.0f} слов\n")

    f.write("\nТоп-10 слов:\n")
    for word, count in word_counts.most_common(10):
        f.write(f"  {word:<20} {count:>10,}\n")

    f.write("\nСозданные графики:\n")
    f.write("  ✓ eda_top20_rubrics.png\n")
    f.write("  ✓ eda_rubrics_frequency_distribution.png\n")
    f.write("  ✓ eda_text_length_distributions.png\n")
    if "rating" in df.columns:
        f.write("  ✓ eda_rating_correlation.png\n")
    f.write("  ✓ eda_top20_words.png\n")
    f.write("  ✓ eda_wordcloud_all.png\n")
    f.write("  ✓ eda_wordcloud_top5_rubrics.png\n")

print("✓ Текстовый отчет сохранен: eda_summary.txt")

print("\n" + "=" * 80)
print("✅ EDA ЗАВЕРШЕН УСПЕШНО!")
print("=" * 80)
print("\nСледующий шаг: Feature Engineering (feature_engineering.py)")
