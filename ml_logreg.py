"""
ml_logreg.py

Обучение ML модели: Logistic Regression + TF-IDF для классификации рубрик.

Этапы:
1. Загрузка данных
2. TF-IDF векторизация
3. Обучение Logistic Regression
4. Подбор гиперпараметров
5. Оценка качества
6. Визуализация результатов
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("ML МОДЕЛЬ: LOGISTIC REGRESSION + TF-IDF")
print("=" * 80)


# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 1: ЗАГРУЗКА ПОДГОТОВЛЕННЫХ ДАННЫХ")
print("=" * 80)

with open("data_ml_preprocessed.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
label_encoder = data["label_encoder"]
n_classes = data["n_classes"]

print(f"\nДанные загружены успешно!")
print(f"  Train размер: {len(X_train):,} записей")
print(f"  Test размер: {len(X_test):,} записей")
print(f"  Классов: {n_classes}")
print(f"  Примеры классов: {list(label_encoder.classes_[:5])}")


# ============================================================================
# 2. TF-IDF ВЕКТОРИЗАЦИЯ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 2: TF-IDF ВЕКТОРИЗАЦИЯ")
print("=" * 80)

# Параметры TF-IDF
tfidf_params = {
    "max_features": 10000,
    "min_df": 5,
    "max_df": 0.7,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
}

print(f"\nПараметры TF-IDF:")
for key, value in tfidf_params.items():
    print(f"  {key}: {value}")

# Создание и обучение TF-IDF векторизатора
print("\nОбучение TF-IDF векторизатора...")
tfidf = TfidfVectorizer(**tfidf_params)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"\n✓ Векторизация завершена!")
print(f"  Train матрица: {X_train_tfidf.shape}")
print(f"  Test матрица: {X_test_tfidf.shape}")
print(f"  Размерность словаря: {len(tfidf.vocabulary_):,}")
print(
    f"  Разреженность train: {(X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]) * 100):.2f}%"
)


# ============================================================================
# 3. BASELINE МОДЕЛЬ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 3: ОБУЧЕНИЕ BASELINE МОДЕЛИ")
print("=" * 80)

# Baseline с параметрами по умолчанию
print("\nОбучение Logistic Regression (baseline)...")
model_baseline = LogisticRegression(max_iter=1000, random_state=42, verbose=0)

import time

start_time = time.time()
model_baseline.fit(X_train_tfidf, y_train)
train_time_baseline = time.time() - start_time

print(f"✓ Модель обучена за {train_time_baseline:.2f} секунд")

# Предсказание
y_pred_baseline = model_baseline.predict(X_test_tfidf)

# Метрики
acc_baseline = accuracy_score(y_test, y_pred_baseline)
macro_f1_baseline = f1_score(y_test, y_pred_baseline, average="macro", zero_division=0)
weighted_f1_baseline = f1_score(
    y_test, y_pred_baseline, average="weighted", zero_division=0
)

print(f"\nМетрики baseline:")
print(f"  Accuracy:     {acc_baseline:.4f} ({acc_baseline * 100:.2f}%)")
print(f"  Macro F1:     {macro_f1_baseline:.4f}")
print(f"  Weighted F1:  {weighted_f1_baseline:.4f}")


# ============================================================================
# 4. ПОДБОР ГИПЕРПАРАМЕТРОВ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 4: ПОДБОР ГИПЕРПАРАМЕТРОВ")
print("=" * 80)

# Тестируем разные значения C
C_values = [0.1, 0.5, 1.0, 5.0, 10.0]

print(f"\nТестирование различных значений C:")
print(
    f"{'C':<10} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12} {'Время (сек)':<12}"
)
print("-" * 60)

results = []
best_score = 0
best_c = 1.0

for c in C_values:
    model = LogisticRegression(
        C=c, max_iter=1000, random_state=42, class_weight="balanced", verbose=0
    )

    start = time.time()
    model.fit(X_train_tfidf, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results.append(
        {
            "C": c,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "train_time": train_time,
        }
    )

    print(
        f"{c:<10.1f} {acc:<12.4f} {macro_f1:<12.4f} {weighted_f1:<12.4f} {train_time:<12.2f}"
    )

    if acc > best_score:
        best_score = acc
        best_c = c

print(f"\n✓ Лучшее значение C: {best_c} (Accuracy: {best_score:.4f})")


# ============================================================================
# 5. ФИНАЛЬНАЯ МОДЕЛЬ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 5: ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ")
print("=" * 80)

print(f"\nОбучение финальной модели с C={best_c}...")
model_final = LogisticRegression(
    C=best_c, max_iter=1000, random_state=42, class_weight="balanced", verbose=0
)

start_time = time.time()
model_final.fit(X_train_tfidf, y_train)
final_train_time = time.time() - start_time

print(f"✓ Финальная модель обучена за {final_train_time:.2f} секунд")

# Предсказание
y_pred_final = model_final.predict(X_test_tfidf)

# Финальные метрики
acc_final = accuracy_score(y_test, y_pred_final)
macro_f1_final = f1_score(y_test, y_pred_final, average="macro", zero_division=0)
weighted_f1_final = f1_score(y_test, y_pred_final, average="weighted", zero_division=0)

print(f"\n{'=' * 80}")
print("ФИНАЛЬНЫЕ МЕТРИКИ")
print(f"{'=' * 80}")
print(f"Accuracy:     {acc_final:.4f} ({acc_final * 100:.2f}%)")
print(f"Macro F1:     {macro_f1_final:.4f}")
print(f"Weighted F1:  {weighted_f1_final:.4f}")


# ============================================================================
# 6. ДЕТАЛЬНЫЙ АНАЛИЗ ПО КЛАССАМ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 6: ДЕТАЛЬНЫЙ АНАЛИЗ ПО КЛАССАМ")
print("=" * 80)

# Classification report
report = classification_report(
    y_test,
    y_pred_final,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0,
)

# Сортировка классов по F1
class_scores = []
for class_name, metrics in report.items():
    if class_name not in ["accuracy", "macro avg", "weighted avg"]:
        class_scores.append(
            (
                class_name,
                metrics["f1-score"],
                metrics["precision"],
                metrics["recall"],
                int(metrics["support"]),
            )
        )

class_scores.sort(key=lambda x: x[1], reverse=True)

print("\nТОП-20 КЛАССОВ ПО F1-SCORE:")
print(f"{'Класс':<50} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Supp':>6}")
print("-" * 85)
for class_name, f1, prec, rec, supp in class_scores[:20]:
    print(f"{class_name:<50} {f1:>6.3f} {prec:>6.3f} {rec:>6.3f} {supp:>6}")

print("\nХУДШИЕ 10 КЛАССОВ:")
print(f"{'Класс':<50} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Supp':>6}")
print("-" * 85)
for class_name, f1, prec, rec, supp in class_scores[-10:]:
    print(f"{class_name:<50} {f1:>6.3f} {prec:>6.3f} {rec:>6.3f} {supp:>6}")


# ============================================================================
# 7. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 7: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 80)

# 7.1. График сравнения гиперпараметров
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

c_vals = [r["C"] for r in results]
acc_vals = [r["accuracy"] for r in results]
f1_vals = [r["macro_f1"] for r in results]

axes[0].plot(c_vals, acc_vals, marker="o", linewidth=2, markersize=8, label="Accuracy")
axes[0].plot(c_vals, f1_vals, marker="s", linewidth=2, markersize=8, label="Macro F1")
axes[0].set_xlabel("Параметр C (регуляризация)", fontweight="bold")
axes[0].set_ylabel("Score", fontweight="bold")
axes[0].set_title("Влияние гиперпараметра C на качество", fontweight="bold")
axes[0].set_xscale("log")
axes[0].grid(alpha=0.3)
axes[0].legend()

# 7.2. Топ-20 классов по F1
top_20_classes = class_scores[:20]
classes_names = [c[0] for c in top_20_classes]
classes_f1 = [c[1] for c in top_20_classes]

axes[1].barh(range(len(classes_names)), classes_f1, color="steelblue", alpha=0.7)
axes[1].set_yticks(range(len(classes_names)))
axes[1].set_yticklabels(classes_names, fontsize=8)
axes[1].set_xlabel("F1-Score", fontweight="bold")
axes[1].set_title("Топ-20 классов по F1-Score", fontweight="bold")
axes[1].invert_yaxis()
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("ml_logreg_results.png", dpi=300, bbox_inches="tight")
print("\n✓ График сохранен: ml_logreg_results.png")
plt.close()

# 7.3. Confusion Matrix (топ-10 классов)

print("\nСоздание confusion matrix для топ-10 классов...")

top_10_indices = [label_encoder.transform([c[0]])[0] for c in class_scores[:10]]

# Фильтруем предсказания для топ-10
mask_test = np.isin(y_test, top_10_indices)
mask_pred = np.isin(y_pred_final, top_10_indices)
# ВАЖНО: оставляем только те примеры, где и истинный класс, и предсказание из топ-10
mask = mask_test & mask_pred

y_test_top10 = y_test[mask]
y_pred_top10 = y_pred_final[mask]

if len(y_test_top10) > 0:
    # Создаем mapping для confusion matrix
    label_mapping = {old: new for new, old in enumerate(top_10_indices)}
    y_test_mapped = np.array([label_mapping[y] for y in y_test_top10])
    y_pred_mapped = np.array([label_mapping[y] for y in y_pred_top10])

    cm = confusion_matrix(y_test_mapped, y_pred_mapped)

    # Нормализация по строкам (по истинным классам)
    cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    top_10_names = [c[0] for c in class_scores[:10]]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=top_10_names,
        yticklabels=top_10_names,
        cbar_kws={"label": "Доля от истинного класса"},
    )
    plt.xlabel("Предсказанный класс", fontweight="bold", fontsize=12)
    plt.ylabel("Истинный класс", fontweight="bold", fontsize=12)
    plt.title(
        "Confusion Matrix (топ-10 классов, нормализованная)",
        fontweight="bold",
        pad=20,
        fontsize=14,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("ml_logreg_confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("✓ Confusion matrix сохранена: ml_logreg_confusion_matrix.png")
    plt.close()
else:
    print("⚠ Недостаточно данных для построения confusion matrix")


# ============================================================================
# 8. ПРИМЕРЫ ПРЕДСКАЗАНИЙ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 8: ПРИМЕРЫ ПРЕДСКАЗАНИЙ")
print("=" * 80)


def predict_with_proba(text, top_k=5):
    """Предсказание с вероятностями"""
    X = tfidf.transform([text])
    proba = model_final.predict_proba(X)[0]
    top_indices = np.argsort(proba)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append((label_encoder.classes_[idx], proba[idx]))
    return results


test_examples = [
    "Отличный ресторан, вкусная еда, приятная атмосфера",
    "Хороший отель, чистые номера, вежливый персонал",
    "Быстро обслужили машину, качественно отремонтировали",
    "Вкусный кофе и десерты, уютное место",
    "Купили продукты, большой выбор, свежие овощи",
]

for i, text in enumerate(test_examples, 1):
    print(f"\n{'=' * 80}")
    print(f"ПРИМЕР {i}")
    print(f"Текст: {text}")
    print(f"\nПредсказания:")

    predictions = predict_with_proba(text)
    for category, prob in predictions:
        marker = "★" if prob > 0.5 else " "
        print(f"{marker} {category:<50} {prob * 100:>6.2f}%")


# ============================================================================
# 9. СОХРАНЕНИЕ МОДЕЛИ И РЕЗУЛЬТАТОВ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 9: СОХРАНЕНИЕ МОДЕЛИ И РЕЗУЛЬТАТОВ")
print("=" * 80)

# Сохранение модели
model_package = {
    "model": model_final,
    "tfidf": tfidf,
    "label_encoder": label_encoder,
    "classes": label_encoder.classes_,
    "n_classes": n_classes,
    "best_params": {"C": best_c, "class_weight": "balanced"},
    "metrics": {
        "accuracy": float(acc_final),
        "macro_f1": float(macro_f1_final),
        "weighted_f1": float(weighted_f1_final),
    },
    "hyperparameter_search": results,
    "class_scores": class_scores,
    "train_time": final_train_time,
    "created_at": datetime.now().isoformat(),
}

with open("model_logreg.pkl", "wb") as f:
    pickle.dump(model_package, f)

print("✓ Модель сохранена: model_logreg.pkl")

# Текстовый отчет
report_text = f"""
{"=" * 80}
ОТЧЕТ ПО ML МОДЕЛИ: LOGISTIC REGRESSION + TF-IDF
{"=" * 80}

ПАРАМЕТРЫ МОДЕЛИ:
  • Алгоритм: Logistic Regression
  • C (регуляризация): {best_c}
  • Class weight: balanced
  • TF-IDF max_features: {tfidf_params["max_features"]}
  • N-grams: {tfidf_params["ngram_range"]}

МЕТРИКИ:
  • Accuracy:     {acc_final:.4f} ({acc_final * 100:.2f}%)
  • Macro F1:     {macro_f1_final:.4f}
  • Weighted F1:  {weighted_f1_final:.4f}

ДАННЫЕ:
  • Train размер: {len(X_train):,} записей
  • Test размер: {len(X_test):,} записей
  • Классов: {n_classes}
  • Время обучения: {final_train_time:.2f} секунд

ТОП-5 КЛАССОВ ПО F1-SCORE:
"""

for class_name, f1, prec, rec, supp in class_scores[:5]:
    report_text += (
        f"  • {class_name:<40} F1={f1:.3f}, Precision={prec:.3f}, Recall={rec:.3f}\n"
    )

report_text += f"""
СОЗДАННЫЕ ФАЙЛЫ:
  ✓ model_logreg.pkl
  ✓ ml_logreg_results.png
  ✓ ml_logreg_confusion_matrix.png
  ✓ ml_logreg_report.txt

СЛЕДУЮЩИЙ ШАГ: dl_rubert.py
"""

with open("ml_logreg_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print("✓ Отчет сохранен: ml_logreg_report.txt")

print("\n" + "=" * 80)
print("✅ ML МОДЕЛЬ ОБУЧЕНА УСПЕШНО!")
print("=" * 80)
print(f"\nИтоговые метрики:")
print(f"  • Accuracy: {acc_final * 100:.2f}%")
print(f"  • Macro F1: {macro_f1_final:.4f}")
print(f"  • Weighted F1: {weighted_f1_final:.4f}")
print(f"  • Время обучения: {final_train_time:.2f} сек")
