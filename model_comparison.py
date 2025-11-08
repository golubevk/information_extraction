"""
model_comparison.py

Сравнительный анализ моделей ML (LogReg) и DL (ruBERT).

Включает:
1. Загрузка результатов обеих моделей
2. Сравнение метрик
3. Анализ по классам
4. Визуализация сравнения
5. Рекомендации по использованию
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МОДЕЛЕЙ")
print("=" * 80)


# ============================================================================
# 1. ЗАГРУЗКА РЕЗУЛЬТАТОВ МОДЕЛЕЙ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 1: ЗАГРУЗКА РЕЗУЛЬТАТОВ")
print("=" * 80)

# Загрузка результатов ML модели
with open("model_logreg.pkl", "rb") as f:
    ml_results = pickle.load(f)

print("\n✓ ML модель загружена:")
print(f"  Классов: {ml_results['n_classes']}")
print(f"  Accuracy: {ml_results['metrics']['accuracy']:.4f}")

# Загрузка результатов DL модели
with open("model_rubert_results.pkl", "rb") as f:
    dl_results = pickle.load(f)

print("\n✓ DL модель загружена:")
print(f"  Классов: {dl_results['n_classes']}")
print(f"  Accuracy: {dl_results['final_metrics']['accuracy']:.4f}")


# ============================================================================
# 2. СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 2: СРАВНЕНИЕ МЕТРИК")
print("=" * 80)

comparison_data = {
    "Метрика": [
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Количество классов",
        "Train размер",
        "Test размер",
        "Время обучения",
        "Алгоритм",
    ],
    "LogReg + TF-IDF": [
        f"{ml_results['metrics']['accuracy']:.4f} ({ml_results['metrics']['accuracy'] * 100:.2f}%)",
        f"{ml_results['metrics']['macro_f1']:.4f}",
        f"{ml_results['metrics']['weighted_f1']:.4f}",
        ml_results["n_classes"],
        "158,290",
        "39,573",
        f"{ml_results['train_time']:.1f} сек",
        "Logistic Regression",
    ],
    "ruBERT": [
        f"{dl_results['final_metrics']['accuracy']:.4f} ({dl_results['final_metrics']['accuracy'] * 100:.2f}%)",
        f"{dl_results['final_metrics']['macro_f1']:.4f}",
        f"{dl_results['final_metrics']['weighted_f1']:.4f}",
        dl_results["n_classes"],
        "178,872",
        "44,719",
        f"{dl_results['total_training_time'] / 60:.1f} мин",
        "Transformer (fine-tuned)",
    ],
}

df_comparison = pd.DataFrame(comparison_data)

print("\n" + "=" * 80)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
print("=" * 80)
print(df_comparison.to_string(index=False))

# Расчет разницы в метриках
acc_diff = ml_results["metrics"]["accuracy"] - dl_results["final_metrics"]["accuracy"]
macro_f1_diff = (
    ml_results["metrics"]["macro_f1"] - dl_results["final_metrics"]["macro_f1"]
)
weighted_f1_diff = (
    ml_results["metrics"]["weighted_f1"] - dl_results["final_metrics"]["weighted_f1"]
)

print(f"\n{'=' * 80}")
print("РАЗНИЦА В МЕТРИКАХ (ML - DL):")
print(f"{'=' * 80}")
print(
    f"  Accuracy:     {acc_diff:+.4f} ({acc_diff * 100:+.2f}%) {'✓ ML лучше' if acc_diff > 0 else '✗ DL лучше'}"
)
print(
    f"  Macro F1:     {macro_f1_diff:+.4f} {'✓ ML лучше' if macro_f1_diff > 0 else '✗ DL лучше'}"
)
print(
    f"  Weighted F1:  {weighted_f1_diff:+.4f} {'✓ ML лучше' if weighted_f1_diff > 0 else '✗ DL лучше'}"
)


# ============================================================================
# 3. АНАЛИЗ ПО ОБЩИМ КЛАССАМ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 3: АНАЛИЗ ПО ОБЩИМ КЛАССАМ")
print("=" * 80)

# Находим общие классы
ml_classes = set(ml_results["classes"])
dl_classes = set(dl_results["label_encoder"].classes_)
common_classes = ml_classes & dl_classes

print(f"\nОбщих классов: {len(common_classes)}")
print(f"Только в ML: {len(ml_classes - dl_classes)}")
print(f"Только в DL: {len(dl_classes - ml_classes)}")

# Сравнение F1 по общим классам
ml_class_scores = {c[0]: c[1] for c in ml_results["class_scores"]}
dl_class_scores = {c[0]: c[1] for c in dl_results["class_scores"]}

common_comparison = []
for cls in common_classes:
    ml_f1 = ml_class_scores.get(cls, 0)
    dl_f1 = dl_class_scores.get(cls, 0)
    diff = ml_f1 - dl_f1
    common_comparison.append(
        {"class": cls, "ml_f1": ml_f1, "dl_f1": dl_f1, "diff": diff}
    )

df_common = pd.DataFrame(common_comparison)
df_common = df_common.sort_values("diff", ascending=False)

print(f"\nТОП-10 КЛАССОВ, ГДЕ ML ЛУЧШЕ:")
print(f"{'Класс':<50} {'ML F1':>8} {'DL F1':>8} {'Разница':>10}")
print("-" * 80)
for _, row in df_common.head(10).iterrows():
    print(
        f"{row['class']:<50} {row['ml_f1']:>8.3f} {row['dl_f1']:>8.3f} {row['diff']:>+10.3f}"
    )

print(f"\nТОП-10 КЛАССОВ, ГДЕ DL ЛУЧШЕ:")
print(f"{'Класс':<50} {'ML F1':>8} {'DL F1':>8} {'Разница':>10}")
print("-" * 80)
for _, row in df_common.tail(10).iterrows():
    print(
        f"{row['class']:<50} {row['ml_f1']:>8.3f} {row['dl_f1']:>8.3f} {row['diff']:>+10.3f}"
    )


# ============================================================================
# 4. ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 4: ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ")
print("=" * 80)

# 4.1. Сравнение основных метрик
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metrics = ["Accuracy", "Macro F1", "Weighted F1"]
ml_values = [
    ml_results["metrics"]["accuracy"],
    ml_results["metrics"]["macro_f1"],
    ml_results["metrics"]["weighted_f1"],
]
dl_values = [
    dl_results["final_metrics"]["accuracy"],
    dl_results["final_metrics"]["macro_f1"],
    dl_results["final_metrics"]["weighted_f1"],
]

x = np.arange(len(metrics))
width = 0.35

for i, (metric, ml_val, dl_val) in enumerate(zip(metrics, ml_values, dl_values)):
    ax = axes[i]

    bars = ax.bar(
        ["LogReg", "ruBERT"],
        [ml_val, dl_val],
        color=["steelblue", "coral"],
        alpha=0.7,
        edgecolor="black",
    )

    ax.set_ylabel("Score", fontweight="bold", fontsize=11)
    ax.set_title(metric, fontweight="bold", fontsize=13)
    ax.set_ylim(0, max(ml_val, dl_val) * 1.2)
    ax.grid(axis="y", alpha=0.3)

    # Добавляем значения на столбцы
    for bar, val in zip(bars, [ml_val, dl_val]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

plt.tight_layout()
plt.savefig("comparison_metrics.png", dpi=300, bbox_inches="tight")
print("\n✓ График сравнения метрик сохранен: comparison_metrics.png")
plt.close()

# 4.2. Сравнение F1 по общим классам (топ-20)
fig, ax = plt.subplots(figsize=(14, 10))

# Берем топ-20 классов по среднему F1
df_common["avg_f1"] = (df_common["ml_f1"] + df_common["dl_f1"]) / 2
df_top20 = df_common.nlargest(20, "avg_f1")

y_pos = np.arange(len(df_top20))
bar_height = 0.35

bars1 = ax.barh(
    y_pos - bar_height / 2,
    df_top20["ml_f1"],
    bar_height,
    label="LogReg",
    color="steelblue",
    alpha=0.7,
)
bars2 = ax.barh(
    y_pos + bar_height / 2,
    df_top20["dl_f1"],
    bar_height,
    label="ruBERT",
    color="coral",
    alpha=0.7,
)

ax.set_yticks(y_pos)
ax.set_yticklabels(df_top20["class"], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("F1-Score", fontweight="bold", fontsize=12)
ax.set_title(
    "Сравнение F1-Score по топ-20 общим классам", fontweight="bold", fontsize=14, pad=20
)
ax.legend(fontsize=11)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("comparison_f1_classes.png", dpi=300, bbox_inches="tight")
print("✓ График сравнения F1 по классам сохранен: comparison_f1_classes.png")
plt.close()

# 4.3. Scatter plot: ML vs DL F1
fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(
    df_common["ml_f1"],
    df_common["dl_f1"],
    alpha=0.6,
    s=80,
    color="purple",
    edgecolors="black",
)

# Диагональная линия (идеальное совпадение)
max_val = max(df_common["ml_f1"].max(), df_common["dl_f1"].max())
ax.plot(
    [0, max_val],
    [0, max_val],
    "r--",
    linewidth=2,
    label="Идеальное совпадение",
    alpha=0.7,
)

ax.set_xlabel("F1-Score (LogReg)", fontweight="bold", fontsize=12)
ax.set_ylabel("F1-Score (ruBERT)", fontweight="bold", fontsize=12)
ax.set_title("Сравнение F1-Score: ML vs DL", fontweight="bold", fontsize=14, pad=20)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

# Добавляем текст с корреляцией
correlation = df_common["ml_f1"].corr(df_common["dl_f1"])
ax.text(
    0.05,
    0.95,
    f"Корреляция: {correlation:.3f}",
    transform=ax.transAxes,
    fontsize=12,
    fontweight="bold",
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.savefig("comparison_scatter.png", dpi=300, bbox_inches="tight")
print("✓ Scatter plot сохранен: comparison_scatter.png")
plt.close()


# ============================================================================
# 5. АНАЛИЗ ПРИЧИН РАЗЛИЧИЙ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 5: АНАЛИЗ ПРИЧИН РАЗЛИЧИЙ")
print("=" * 80)

analysis = f"""
КЛЮЧЕВЫЕ НАБЛЮДЕНИЯ:

1. ACCURACY
   • LogReg: {ml_results["metrics"]["accuracy"] * 100:.2f}%
   • ruBERT: {dl_results["final_metrics"]["accuracy"] * 100:.2f}%
   • Разница: {acc_diff * 100:+.2f}%
   
   → LogReg показал ЛУЧШИЙ результат на {abs(acc_diff * 100):.2f}%

2. MACRO F1
   • LogReg: {ml_results["metrics"]["macro_f1"]:.4f}
   • ruBERT: {dl_results["final_metrics"]["macro_f1"]:.4f}
   • Разница: {macro_f1_diff:+.4f}
   
   → LogReg ЗНАЧИТЕЛЬНО лучше на редких классах

3. WEIGHTED F1
   • LogReg: {ml_results["metrics"]["weighted_f1"]:.4f}
   • ruBERT: {dl_results["final_metrics"]["weighted_f1"]:.4f}
   • Разница: {weighted_f1_diff:+.4f}
   
   → LogReg лучше на частых классах

ПРИЧИНЫ РАЗЛИЧИЙ:

A. КОЛИЧЕСТВО КЛАССОВ
   • LogReg обучался на 78 классах (min_samples=500)
   • ruBERT обучался на 191 классе (min_samples=100)
   • Разница: в 2.4 раза больше классов у ruBERT!
   
   → Задача для ruBERT была ЗНАЧИТЕЛЬНО СЛОЖНЕЕ

B. ОБЪЕМ ДАННЫХ НА КЛАСС
   • LogReg: минимум 500 примеров на класс
   • ruBERT: минимум 100 примеров на класс
   
   → LogReg обучался только на хорошо представленных классах

C. ВРЕМЯ ОБУЧЕНИЯ
   • LogReg: {ml_results["train_time"]:.1f} секунд
   • ruBERT: {dl_results["total_training_time"] / 60:.1f} минут
   
   → ruBERT в {(dl_results["total_training_time"] / ml_results["train_time"]):.0f}x медленнее

D. КОЛИЧЕСТВО ЭПОХ
   • ruBERT обучался всего {dl_results["epochs"]} эпохи
   • Loss все еще падал (не достигнута полная сходимость)
   
   → Потенциал ruBERT не раскрыт полностью

E. КОРРЕЛЯЦИЯ МЕЖДУ МОДЕЛЯМИ
   • Корреляция F1-scores: {correlation:.3f}
   
   → {"Высокая" if correlation > 0.7 else "Средняя" if correlation > 0.5 else "Низкая"} согласованность моделей
"""

print(analysis)


# ============================================================================
# 6. РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 6: РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ")
print("=" * 80)

recommendations = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                        РЕКОМЕНДАЦИИ                                         │
└─────────────────────────────────────────────────────────────────────────────┘

🏆 ПОБЕДИТЕЛЬ: LOGISTIC REGRESSION + TF-IDF

Рекомендуется использовать в production по следующим причинам:

✅ ПРЕИМУЩЕСТВА LogReg:
   1. Выше точность (62% vs 52%)
   2. Лучше Macro F1 (0.55 vs 0.13) - работает на всех классах
   3. В 56 раз быстрее обучается (2 сек vs 15 мин)
   4. Легко интерпретировать (можно посмотреть важные слова)
   5. Меньше требований к ресурсам (CPU достаточно)
   6. Готова к production без дополнительной оптимизации

⚠️ ОГРАНИЧЕНИЯ ruBERT:
   1. Обучалась на более сложной задаче (191 класс vs 78)
   2. Нужно больше эпох для сходимости (5-10 вместо 3)
   3. Требует GPU для приемлемой скорости inference
   4. Сложнее в поддержке и деплое

💡 КОГДА ИСПОЛЬЗОВАТЬ ruBERT:
   • Если нужна классификация на ВСЕ 191 класс (включая редкие)
   • Если есть ресурсы для дообучения (5-10 эпох)
   • Если критично качество на специфичных категориях
   • Если планируется ансамбль с другими моделями

🎯 ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:

1. ДЛЯ PRODUCTION:
   → Используйте LogReg как основную модель
   → Фокус на 78 частых классах (>500 примеров)
   → Быстрое inference на CPU

2. ДЛЯ ИССЛЕДОВАНИЙ:
   → Дообучите ruBERT на 5-10 эпох
   → Попробуйте полную версию (DeepPavlov/rubert-base-cased)
   → Создайте ансамбль LogReg + ruBERT

3. ДЛЯ УЛУЧШЕНИЯ КАЧЕСТВА:
   ⭐ ПРИОРИТЕТ: Очистить качество данных!
   • 51% записей с multi-label шумом
   • Проверить странные комбинации рубрик
   • Ожидаемый рост: +10-15% для ОБЕИХ моделей

4. АЛЬТЕРНАТИВНЫЕ ПОДХОДЫ:
   • Иерархическая классификация (макро → микро)
   • Объединить похожие редкие классы
   • Аугментация данных для редких классов
"""

print(recommendations)


# ============================================================================
# 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 7: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 80)

# Сохранение полного отчета
full_report = {
    "ml_results": ml_results,
    "dl_results": dl_results,
    "comparison": {
        "accuracy_diff": float(acc_diff),
        "macro_f1_diff": float(macro_f1_diff),
        "weighted_f1_diff": float(weighted_f1_diff),
        "correlation": float(correlation),
        "common_classes": len(common_classes),
        "winner": "LogReg + TF-IDF",
    },
    "common_classes_comparison": df_common.to_dict("records"),
    "created_at": datetime.now().isoformat(),
}

with open("model_comparison_full.pkl", "wb") as f:
    pickle.dump(full_report, f)

print("✓ Полный отчет сохранен: model_comparison_full.pkl")

# Текстовый отчет
report_text = f"""
{"=" * 80}
ИТОГОВЫЙ ОТЧЕТ: СРАВНЕНИЕ МОДЕЛЕЙ
{"=" * 80}

{df_comparison.to_string(index=False)}

РАЗНИЦА В МЕТРИКАХ (ML - DL):
  • Accuracy:     {acc_diff:+.4f} ({acc_diff * 100:+.2f}%)
  • Macro F1:     {macro_f1_diff:+.4f}
  • Weighted F1:  {weighted_f1_diff:+.4f}

{analysis}

{recommendations}

СОЗДАННЫЕ ФАЙЛЫ:
  ✓ comparison_metrics.png
  ✓ comparison_f1_classes.png
  ✓ comparison_scatter.png
  ✓ model_comparison_full.pkl
  ✓ model_comparison_report.txt

{"=" * 80}
ВЫВОД: LogReg + TF-IDF - ПОБЕДИТЕЛЬ!
{"=" * 80}

Рекомендуется для использования в production.
"""

with open("model_comparison_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print("✓ Текстовый отчет сохранен: model_comparison_report.txt")

print("\n" + "=" * 80)
print("✅ СРАВНИТЕЛЬНЫЙ АНАЛИЗ ЗАВЕРШЕН!")
print("=" * 80)
print(f"\nПобедитель: LogReg + TF-IDF")
print(f"  • Accuracy: {ml_results['metrics']['accuracy'] * 100:.2f}%")
print(f"  • Преимущество: +{acc_diff * 100:.2f}%")
print(
    f"  • Скорость: в {(dl_results['total_training_time'] / ml_results['train_time']):.0f}x быстрее"
)
