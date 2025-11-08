"""
dl_rubert.py

Обучение DL модели: ruBERT для классификации рубрик.

Этапы:
1. Загрузка данных
2. Токенизация
3. Создание Dataset и DataLoader
4. Обучение ruBERT
5. Оценка качества
6. Визуализация loss и метрик по эпохам
7. Сохранение результатов
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
from tqdm.auto import tqdm
import time
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("DL МОДЕЛЬ: ruBERT FINE-TUNING")
print("=" * 80)


# ============================================================================
# 1. ПРОВЕРКА УСТРОЙСТВА
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 1: ПРОВЕРКА ДОСТУПНЫХ УСТРОЙСТВ")
print("=" * 80)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Используется Apple Silicon GPU (MPS)")
    print(f"  Модель чипа: Apple Silicon")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✓ Используется CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠ Используется CPU (будет медленнее)")

print(f"\nВыбранное устройство: {device}")


# ============================================================================
# 2. ЗАГРУЗКА ДАННЫХ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 2: ЗАГРУЗКА ПОДГОТОВЛЕННЫХ ДАННЫХ")
print("=" * 80)

with open("data_bert_preprocessed.pkl", "rb") as f:
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
# 3. ТОКЕНИЗАЦИЯ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 3: ТОКЕНИЗАЦИЯ")
print("=" * 80)

# Выбор модели
# MODEL_NAME = "cointegrated/rubert-tiny"  # Легковесная версия для быстрого обучения
MODEL_NAME = 'DeepPavlov/rubert-base-cased'  # Полная версия (медленнее, но точнее)

print(f"\nИспользуемая модель: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Параметры
MAX_LENGTH = 128  # Максимальная длина последовательности
BATCH_SIZE = 64  # Размер батча (можно увеличить до 64-128 на M4 Pro)
EPOCHS = 10  # Количество эпох (можно увеличить до 5-10)
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

print(f"\nПараметры обучения:")
print(f"  Max length: {MAX_LENGTH}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")


# ============================================================================
# 4. DATASET И DATALOADER
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 4: СОЗДАНИЕ DATASET И DATALOADER")
print("=" * 80)


class TextDataset(Dataset):
    """Dataset для текстов"""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


print("\nСоздание датасетов...")
train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LENGTH)
test_dataset = TextDataset(X_test, y_test, tokenizer, MAX_LENGTH)

# DataLoader (num_workers=0 для избежания проблем на macOS)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

print(f"✓ Train batches: {len(train_loader)}")
print(f"✓ Test batches: {len(test_loader)}")


# ============================================================================
# 5. МОДЕЛЬ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 5: ИНИЦИАЛИЗАЦИЯ МОДЕЛИ")
print("=" * 80)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=n_classes, problem_type="single_label_classification"
)

model = model.to(device)

print(f"\n✓ Модель загружена на {device}")
print(f"Параметров в модели: {sum(p.numel() for p in model.parameters()):,}")
print(
    f"Обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
)


# ============================================================================
# 6. ОПТИМИЗАТОР И SCHEDULER
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 6: НАСТРОЙКА ОПТИМИЗАЦИИ")
print("=" * 80)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

print(f"  Total training steps: {total_steps}")
print(f"  Warmup steps: {warmup_steps}")


# ============================================================================
# 7. ФУНКЦИИ ДЛЯ ОБУЧЕНИЯ И ОЦЕНКИ
# ============================================================================


def train_epoch(model, data_loader, optimizer, scheduler, device):
    """Обучение на одной эпохе"""
    model.train()

    losses = []
    correct_predictions = 0
    total_predictions = 0

    progress_bar = tqdm(data_loader, desc="Training")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        total_predictions += labels.size(0)

        progress_bar.set_postfix(
            {
                "loss": f"{np.mean(losses):.4f}",
                "acc": f"{correct_predictions / total_predictions:.4f}",
            }
        )

    return np.mean(losses), correct_predictions / total_predictions


def eval_model(model, data_loader, device):
    """Оценка модели"""
    model.eval()

    losses = []
    predictions = []
    true_labels = []

    progress_bar = tqdm(data_loader, desc="Evaluation")

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(
        true_labels, predictions, average="weighted", zero_division=0
    )

    return {
        "loss": np.mean(losses),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "predictions": predictions,
        "true_labels": true_labels,
    }


# ============================================================================
# 8. ЦИКЛ ОБУЧЕНИЯ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 7: ОБУЧЕНИЕ МОДЕЛИ")
print("=" * 80)

best_accuracy = 0
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "val_macro_f1": [],
    "val_weighted_f1": [],
    "epoch_time": [],
}

total_start_time = time.time()

for epoch in range(EPOCHS):
    print(f"\n{'=' * 80}")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"{'=' * 80}")

    epoch_start_time = time.time()

    # Обучение
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, scheduler, device
    )

    epoch_time = time.time() - epoch_start_time

    print(f"\nTrain Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Epoch Time: {epoch_time:.1f}s")

    # Валидация
    val_metrics = eval_model(model, test_loader, device)

    print(f"\nValidation Loss: {val_metrics['loss']:.4f}")
    print(
        f"Validation Accuracy: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy'] * 100:.2f}%)"
    )
    print(f"Macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {val_metrics['weighted_f1']:.4f}")

    # Сохранение истории
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_metrics["loss"])
    history["val_acc"].append(val_metrics["accuracy"])
    history["val_macro_f1"].append(val_metrics["macro_f1"])
    history["val_weighted_f1"].append(val_metrics["weighted_f1"])
    history["epoch_time"].append(epoch_time)

    # Сохранение лучшей модели
    if val_metrics["accuracy"] > best_accuracy:
        best_accuracy = val_metrics["accuracy"]
        torch.save(model.state_dict(), "model_rubert_best.pt")
        print(f"\n✓ Сохранена лучшая модель (accuracy: {best_accuracy:.4f})")

total_time = time.time() - total_start_time

print("\n" + "=" * 80)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print("=" * 80)
print(f"Лучшая accuracy: {best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")
print(f"Общее время обучения: {total_time / 60:.1f} минут")
print(f"Среднее время на эпоху: {np.mean(history['epoch_time']):.1f}s")


# ============================================================================
# 9. ФИНАЛЬНАЯ ОЦЕНКА
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 8: ФИНАЛЬНАЯ ОЦЕНКА")
print("=" * 80)

# Загружаем лучшую модель
model.load_state_dict(torch.load("model_rubert_best.pt"))
final_metrics = eval_model(model, test_loader, device)

print(f"\n{'=' * 80}")
print("ИТОГОВЫЕ МЕТРИКИ")
print(f"{'=' * 80}")
print(
    f"Accuracy:     {final_metrics['accuracy']:.4f} ({final_metrics['accuracy'] * 100:.2f}%)"
)
print(f"Macro F1:     {final_metrics['macro_f1']:.4f}")
print(f"Weighted F1:  {final_metrics['weighted_f1']:.4f}")


# ============================================================================
# 10. ДЕТАЛЬНЫЙ АНАЛИЗ ПО КЛАССАМ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 9: ДЕТАЛЬНЫЙ АНАЛИЗ ПО КЛАССАМ")
print("=" * 80)

y_pred = final_metrics["predictions"]
y_true = final_metrics["true_labels"]

report = classification_report(
    y_true,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0,
)

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
# 11. ВИЗУАЛИЗАЦИЯ ОБУЧЕНИЯ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 10: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 80)

# 11.1. Графики Loss и Accuracy по эпохам
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, EPOCHS + 1)

# Loss
axes[0].plot(
    epochs_range,
    history["train_loss"],
    "o-",
    linewidth=2,
    markersize=8,
    label="Train Loss",
    color="steelblue",
)
axes[0].plot(
    epochs_range,
    history["val_loss"],
    "s-",
    linewidth=2,
    markersize=8,
    label="Validation Loss",
    color="coral",
)
axes[0].set_xlabel("Epoch", fontweight="bold", fontsize=12)
axes[0].set_ylabel("Loss", fontweight="bold", fontsize=12)
axes[0].set_title("Loss по эпохам", fontweight="bold", fontsize=14)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Accuracy
axes[1].plot(
    epochs_range,
    history["train_acc"],
    "o-",
    linewidth=2,
    markersize=8,
    label="Train Accuracy",
    color="steelblue",
)
axes[1].plot(
    epochs_range,
    history["val_acc"],
    "s-",
    linewidth=2,
    markersize=8,
    label="Validation Accuracy",
    color="coral",
)
axes[1].set_xlabel("Epoch", fontweight="bold", fontsize=12)
axes[1].set_ylabel("Accuracy", fontweight="bold", fontsize=12)
axes[1].set_title("Accuracy по эпохам", fontweight="bold", fontsize=14)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("dl_rubert_training_curves.png", dpi=300, bbox_inches="tight")
print("\n✓ График обучения сохранен: dl_rubert_training_curves.png")
plt.close()

# 11.2. Топ-20 классов по F1
fig, ax = plt.subplots(figsize=(12, 8))

top_20_classes = class_scores[:20]
classes_names = [c[0] for c in top_20_classes]
classes_f1 = [c[1] for c in top_20_classes]

bars = ax.barh(range(len(classes_names)), classes_f1, color="lightcoral", alpha=0.7)
ax.set_yticks(range(len(classes_names)))
ax.set_yticklabels(classes_names, fontsize=9)
ax.set_xlabel("F1-Score", fontweight="bold", fontsize=12)
ax.set_title("Топ-20 классов по F1-Score (ruBERT)", fontweight="bold", fontsize=14)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)

# Добавляем значения
for i, (bar, f1) in enumerate(zip(bars, classes_f1)):
    ax.text(f1, i, f" {f1:.3f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("dl_rubert_top20_classes.png", dpi=300, bbox_inches="tight")
print("✓ График топ-20 классов сохранен: dl_rubert_top20_classes.png")
plt.close()


# ============================================================================
# 12. ПРИМЕРЫ ПРЕДСКАЗАНИЙ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 11: ПРИМЕРЫ ПРЕДСКАЗАНИЙ")
print("=" * 80)


def predict_text(
    text, model, tokenizer, label_encoder, device, max_length=128, top_k=5
):
    """Предсказание с вероятностями"""
    model.eval()

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=1)[0]
    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))

    results = []
    for prob, idx in zip(top_probs, top_indices):
        category = label_encoder.classes_[idx.item()]
        probability = prob.item()
        results.append((category, probability))

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

    predictions = predict_text(
        text, model, tokenizer, label_encoder, device, MAX_LENGTH
    )

    for category, prob in predictions:
        marker = "★" if prob > 0.5 else " "
        print(f"{marker} {category:<50} {prob * 100:>6.2f}%")


# ============================================================================
# 13. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================

print("\n" + "=" * 80)
print("ЭТАП 12: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 80)

results_package = {
    "model_name": MODEL_NAME,
    "device": str(device),
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "best_accuracy": best_accuracy,
    "final_metrics": {
        "accuracy": float(final_metrics["accuracy"]),
        "macro_f1": float(final_metrics["macro_f1"]),
        "weighted_f1": float(final_metrics["weighted_f1"]),
    },
    "history": history,
    "label_encoder": label_encoder,
    "n_classes": n_classes,
    "class_scores": class_scores,
    "total_training_time": total_time,
    "created_at": datetime.now().isoformat(),
}

with open("model_rubert_results.pkl", "wb") as f:
    pickle.dump(results_package, f)

print("✓ Результаты сохранены: model_rubert_results.pkl")
print("✓ Модель сохранена: model_rubert_best.pt")

# Текстовый отчет
report_text = f"""
{"=" * 80}
ОТЧЕТ ПО DL МОДЕЛИ: ruBERT
{"=" * 80}

ПАРАМЕТРЫ МОДЕЛИ:
  • Модель: {MODEL_NAME}
  • Устройство: {device}
  • Max sequence length: {MAX_LENGTH}
  • Batch size: {BATCH_SIZE}
  • Epochs: {EPOCHS}
  • Learning rate: {LEARNING_RATE}

МЕТРИКИ:
  • Accuracy:     {final_metrics["accuracy"]:.4f} ({final_metrics["accuracy"] * 100:.2f}%)
  • Macro F1:     {final_metrics["macro_f1"]:.4f}
  • Weighted F1:  {final_metrics["weighted_f1"]:.4f}

ДАННЫЕ:
  • Train размер: {len(X_train):,} записей
  • Test размер: {len(X_test):,} записей
  • Классов: {n_classes}
  • Время обучения: {total_time / 60:.1f} минут

ИСТОРИЯ ОБУЧЕНИЯ:
"""

for epoch in range(EPOCHS):
    report_text += (
        f"  Epoch {epoch + 1}: Train Loss={history['train_loss'][epoch]:.4f}, "
    )
    report_text += f"Val Loss={history['val_loss'][epoch]:.4f}, "
    report_text += f"Val Acc={history['val_acc'][epoch]:.4f}\n"

report_text += f"""
ТОП-5 КЛАССОВ ПО F1-SCORE:
"""

for class_name, f1, prec, rec, supp in class_scores[:5]:
    report_text += (
        f"  • {class_name:<40} F1={f1:.3f}, Precision={prec:.3f}, Recall={rec:.3f}\n"
    )

report_text += f"""
СОЗДАННЫЕ ФАЙЛЫ:
  ✓ model_rubert_best.pt
  ✓ model_rubert_results.pkl
  ✓ dl_rubert_training_curves.png
  ✓ dl_rubert_top20_classes.png
  ✓ dl_rubert_report.txt

СЛЕДУЮЩИЙ ШАГ: model_comparison.py
"""

with open("dl_rubert_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print("✓ Отчет сохранен: dl_rubert_report.txt")

print("\n" + "=" * 80)
print("✅ DL МОДЕЛЬ ОБУЧЕНА УСПЕШНО!")
print("=" * 80)
print(f"\nИтоговые метрики:")
print(f"  • Accuracy: {final_metrics['accuracy'] * 100:.2f}%")
print(f"  • Macro F1: {final_metrics['macro_f1']:.4f}")
print(f"  • Weighted F1: {final_metrics['weighted_f1']:.4f}")
print(f"  • Время обучения: {total_time / 60:.1f} мин")
