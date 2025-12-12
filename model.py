import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=6, suppress=True)

videos_df = pd.read_csv("file_names.csv", sep=";")

all_keypoints, all_exercises, all_technique = [], [], []

for exercise in videos_df["exercise"].unique():
    for label in [0, 1]:
        label_folder = os.path.join(f"{exercise} keypoints", str(label))
        if not os.path.isdir(label_folder):
            continue

        video_indices = sorted(os.listdir(label_folder))
        for video_idx in video_indices:
            video_folder = os.path.join(label_folder, video_idx)
            if not os.path.isdir(video_folder):
                continue
            frames = sorted(os.listdir(video_folder))
            video_keypoints = []
            for frame in frames:
                npy_file = os.path.join(video_folder, frame)
                if not os.path.isfile(npy_file):
                    continue
                keypoints = np.load(npy_file)
                video_keypoints.append(keypoints)

            if len(video_keypoints) == 0:
                continue

            video_keypoints_array = np.array(video_keypoints)
            all_keypoints.append(video_keypoints_array)
            all_exercises.append(exercise)
            all_technique.append(label)

y_ex_int, exercises = pd.factorize(all_exercises)
x = np.array(all_keypoints, dtype=np.float32)
y_ex = np.array(y_ex_int, dtype=np.int64)
y_tech = np.array(all_technique, dtype=np.int64)

combined_strata = y_ex * 2 + y_tech
x_train, x_test, y_ex_train, y_ex_test, y_tech_train, y_tech_test = train_test_split(
    x, y_ex, y_tech, test_size=0.1, random_state=42, stratify=combined_strata
)

classes_ex = np.unique(y_ex_train)
cw_ex_arr = compute_class_weight(class_weight='balanced', classes=classes_ex, y=y_ex_train)
class_weight_ex = {int(c): float(w) for c, w in zip(classes_ex, cw_ex_arr)}
print("class_weight (exercise):", class_weight_ex)

classes_tech = np.unique(y_tech_train)
cw_tech_arr = compute_class_weight(class_weight='balanced', classes=classes_tech, y=y_tech_train)
class_weight_tech = {int(c): float(w) for c, w in zip(classes_tech, cw_tech_arr)}
print("class_weight (technique):", class_weight_tech)

K = len(exercises)
input_shape = (237, 132)

inp = Input(shape=input_shape, name="kp_seq")
h = LSTM(32, return_sequences=True, activation='tanh')(inp)
h = LSTM(16, return_sequences=False, activation='tanh')(h)
h = Dense(64, activation='tanh')(h)

out_ex = Dense(K, activation='softmax', name='exercise')(h)
out_tech = Dense(1, activation='sigmoid', name='technique')(h)

model = Model(inputs=inp, outputs={'exercise': out_ex, 'technique': out_tech})
model.compile(
    optimizer='adam',
    loss={'exercise': 'sparse_categorical_crossentropy', 'technique': 'binary_crossentropy'},
    metrics={'exercise': ['accuracy'], 'technique': ['accuracy']},
)

sample_weight_ex = np.array([class_weight_ex[int(c)] for c in y_ex_train])
sample_weight_tech = np.array([class_weight_tech[int(c)] for c in y_tech_train])

history = model.fit(
    x_train,
    {'exercise': y_ex_train, 'technique': y_tech_train},
    validation_data=(x_test, {"exercise": y_ex_test, "technique": y_tech_test}),
    epochs=20,
    sample_weight={'exercise': sample_weight_ex, 'technique': sample_weight_tech},
    verbose=1
)

preds = model.predict(x_test)
probs_ex = preds['exercise']
probs_tech = preds['technique'].ravel()

y_pred_ex = np.argmax(probs_ex, axis=1)
y_pred_tech = (probs_tech >= 0.5).astype(int)

print("\n=== Ćwiczenia (multi-class) ===")
print(classification_report(
    y_ex_test, y_pred_ex,
    labels=np.arange(K), target_names=exercises, zero_division=0
))

cm_ex = confusion_matrix(y_ex_test, y_pred_ex, labels=np.arange(K))
plt.figure(figsize=(6, 5))
sns.heatmap(cm_ex, annot=True, fmt='d', cmap='Blues',
            xticklabels=exercises, yticklabels=exercises)
plt.xlabel("Przewidywane ćwiczenie")
plt.ylabel("Rzeczywiste ćwiczenie")
plt.title("Macierz pomyłek – ćwiczenia")
plt.tight_layout()
plt.show()

print("\n=== Technika (0=niepoprawna, 1=poprawna) ===")
print(classification_report(
    y_tech_test, y_pred_tech,
    labels=[0, 1], target_names=["niepoprawna", "poprawna"], zero_division=0
))

cm_tech = confusion_matrix(y_tech_test, y_pred_tech, labels=[0, 1])
plt.figure(figsize=(4.5, 4))
sns.heatmap(cm_tech, annot=True, fmt='d', cmap='Blues',
            xticklabels=["niepoprawna", "poprawna"],
            yticklabels=["niepoprawna", "poprawna"])
plt.xlabel("Przewidywana technika")
plt.ylabel("Rzeczywista technika")
plt.title("Macierz pomyłek – technika")
plt.tight_layout()
plt.show()

print("Dostępne metryki w historii uczenia:", history.history.keys())

plt.figure()
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.xlabel("Epoka")
plt.ylabel("Strata")
plt.title("Przebieg funkcji straty (loss)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(history.history["exercise_accuracy"], label="train accuracy")
plt.plot(history.history["val_exercise_accuracy"], label="val accuracy")
plt.xlabel("Epoka")
plt.ylabel("Dokładność")
plt.title("Dokładność klasyfikacji ćwiczeń w czasie uczenia")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(history.history["technique_accuracy"], label="train accuracy")
plt.plot(history.history["val_technique_accuracy"], label="val accuracy")
plt.xlabel("Epoka")
plt.ylabel("Dokładność")
plt.title("Dokładność klasyfikacji techniki w czasie uczenia")
plt.legend()
plt.tight_layout()
plt.show()

unique_ex, counts_ex = np.unique(y_ex, return_counts=True)
labels_ex = [exercises[i] for i in unique_ex]

plt.figure()
plt.bar(labels_ex, counts_ex)
plt.xlabel("Klasa ćwiczenia")
plt.ylabel("Liczba próbek (cały zbiór)")
plt.title("Rozkład liczności klas ćwiczeń w całym zbiorze danych")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

unique_tech, counts_tech = np.unique(y_tech, return_counts=True)

tech_labels = ["niepoprawna", "poprawna"]  # 0, 1
plt.figure()
plt.bar(tech_labels, counts_tech)
plt.xlabel("Technika")
plt.ylabel("Liczba próbek (cały zbiór)")
plt.title("Rozkład liczności klas techniki w całym zbiorze danych")
plt.tight_layout()
plt.show()

model_accuracy_ex = accuracy_score(y_ex_test, y_pred_ex)
print("Dokładność modelu LSTM (ćwiczenia):", model_accuracy_ex)

values_ex, counts_ex = np.unique(y_ex_train, return_counts=True)
majority_class_ex = values_ex[np.argmax(counts_ex)]
y_pred_majority_ex = np.full_like(y_ex_test, fill_value=majority_class_ex)
majority_accuracy_ex = accuracy_score(y_ex_test, y_pred_majority_ex)
print("Dokładność majority baseline (ćwiczenia):", majority_accuracy_ex)

num_classes_ex = len(values_ex)
rng = np.random.default_rng(seed=42)
y_pred_random_ex = rng.integers(low=0, high=num_classes_ex, size=len(y_ex_test))
random_accuracy_ex = accuracy_score(y_ex_test, y_pred_random_ex)
print("Dokładność random baseline (ćwiczenia):", random_accuracy_ex)

labels = ["LSTM", "Majority", "Random"]
accuracies = [model_accuracy_ex, majority_accuracy_ex, random_accuracy_ex]

plt.figure()
plt.bar(labels, accuracies)
plt.ylabel("Dokładność (accuracy)")
plt.title("Porównanie dokładności modelu LSTM i modeli bazowych (ćwiczenia)")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.show()

model_accuracy_tech = accuracy_score(y_tech_test, y_pred_tech)
print("Dokładność modelu LSTM (technika):", model_accuracy_tech)

values_tech, counts_tech = np.unique(y_tech_train, return_counts=True)
majority_class_tech = values_tech[np.argmax(counts_tech)]
y_pred_majority_tech = np.full_like(y_tech_test, fill_value=majority_class_tech)
majority_accuracy_tech = accuracy_score(y_tech_test, y_pred_majority_tech)
print("Dokładność majority baseline (technika):", majority_accuracy_tech)

num_classes_tech = len(values_tech)
rng = np.random.default_rng(seed=42)
y_pred_random_tech = rng.integers(low=0, high=num_classes_tech, size=len(y_tech_test))
random_accuracy_tech = accuracy_score(y_tech_test, y_pred_random_tech)
print("Dokładność random baseline (technika):", random_accuracy_tech)

labels_tech = ["LSTM", "Majority", "Random"]
accuracies_tech = [model_accuracy_tech, majority_accuracy_tech, random_accuracy_tech]

plt.figure()
plt.bar(labels_tech, accuracies_tech)
plt.ylabel("Dokładność (accuracy)")
plt.ylim(0, 1.0)
plt.title("Porównanie dokładności modelu LSTM i modeli bazowych (technika)")
plt.tight_layout()
plt.show()

model.save('model.h5')
