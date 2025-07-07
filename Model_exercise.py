import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense
np.set_printoptions(precision=6, suppress=True)
videos_df = pd.read_csv("file_names.csv", sep = ";")
all_keypoints = []
all_exercises = []
for exercise in videos_df["exercise"].unique():
    for label in [0, 1]:
        label_folder = os.path.join(f"{exercise} keypoints", str(label))

        video_indices = sorted(os.listdir(label_folder))

        for video_idx in video_indices:
            video_folder = os.path.join(label_folder, video_idx)

            frames = sorted(os.listdir(video_folder))
            video_keypoints = []
            for frame in frames:

                npy_file = os.path.join(video_folder, frame)

                keypoints = np.load(npy_file)

                video_keypoints.append(keypoints)
            video_keypoints_array = np.array(video_keypoints)

            all_keypoints.append(video_keypoints_array)
            all_exercises.append(exercise)
all_labels, exercises = pd.factorize(all_exercises)
x = np.array(all_keypoints)
labels = np.array(all_labels)

y=to_categorical(labels)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
model = Sequential()
model.add(LSTM(265, input_shape=(237,132)))
# model.add(LSTM(128,return_sequences = True,activation = 'tanh', input_shape=(237,132)))
# model.add(LSTM(64,return_sequences = True,activation = 'tanh'))
# model.add(LSTM(32,return_sequences = False,activation = 'tanh'))
# model.add(Dense(64,activation = 'tanh'))
# model.add(Dense(4,activation = 'softmax'))
model.add(Dense(4,activation = 'softmax'))
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train,y_train,epochs = 50)
test = model.predict(x_test)
model.summary()
print("Przewidywane wartości:", test)

y_test_labels = np.argmax(y_test, axis=1)
test_labels = np.argmax(test, axis=1)

for i in range(len(y_test)):
    print(f"Próbka {i}: Rzeczywiste = {y_test_labels[i]}, Przewidywane = {test_labels[i]}")

correct_predictions = np.sum(y_test_labels == test_labels[:len(y_test_labels)])
total_predictions = len(y_test_labels)
accuracy = correct_predictions / total_predictions * 100

print(f"Procent poprawnych predykcji: {accuracy:.2f}%")

model.save('model2.h5')

