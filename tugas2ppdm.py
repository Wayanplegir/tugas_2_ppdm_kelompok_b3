import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import librosa
import librosa.display
import librosa.effects as le
import IPython.display as ipd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
import joblib
from itertools import cycle
from sklearn.metrics import classification_report, confusion_matrix
import pickle

from google.colab import drive
drive.mount('/content/drive')

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

target_shape = (128, 128)

emotions = ["ANG", "SAD", "DIS", "FEA", "HAP", "NEU"]
emotion_dir_names = {
    "ANG": "/content/drive/MyDrive/Colab Notebooks/Emotions/Angry",
    "SAD": "/content/drive/MyDrive/Colab Notebooks/Emotions/Sad",
    "DIS": "/content/drive/MyDrive/Colab Notebooks/Emotions/Disgusted",
    "FEA": "/content/drive/MyDrive/Colab Notebooks/Emotions/Fearful",
    "HAP": "/content/drive/MyDrive/Colab Notebooks/Emotions/Happy",
    "NEU": "/content/drive/MyDrive/Colab Notebooks/Emotions/Neutral"
}
label_dict = {
    "ANG": 0, "SAD": 1, "DIS": 2, "FEA": 3, "HAP": 4, "NEU": 5
}

for emotion in emotions:
    emotion_dir = emotion_dir_names[emotion]
    files = glob(f'{emotion_dir}/*.wav')
    num_files = len(files)
    print(f"Emotion {emotion}: {num_files} files")

    def load_emotion_files(emotion, emotion_dir_name, max_files=1000):
    files = glob(f'{emotion_dir_name}/*.wav')
    files = sorted(files)
    num_files = len(files)
    print(f"Total files for emotion {emotion}: {num_files}")
    if num_files > max_files:
        files = random.sample(files, max_files)
    return files

    def process_audio_files(files, label, target_shape=(128, 128)):
    train_data = []
    labels = []
    with tqdm(total=len(files), desc=f'Processing {label} files') as pbar:
        for file in files:
            y, sr = librosa.load(file, sr=None)
            y_stretched = le.time_stretch(y, rate=1)
            mfcc = librosa.feature.mfcc(y=y_stretched, sr=sr, n_mfcc=13)
            mfcc = tf.image.resize(np.expand_dims(mfcc, axis=-1), target_shape)
            train_data.append(mfcc.numpy())
            labels.append(label)
            pbar.update(1)
    return train_data, labels

    def process_files(emotions, emotion_dir_names, label_dict, target_shape=(128, 128), max_files=1000):
    all_train_data = []
    all_labels = []

    with tqdm(total=len(emotions), desc='Processing dataset') as pbar:
        for emotion in emotions:
            files = load_emotion_files(emotion, emotion_dir_names[emotion], max_files)
            if not files:
                print(f"No files found for emotion: {emotion}")
            num_files = len(files)
            for file in files:
                y, sr = librosa.load(file, sr=None)
                y_stretched = le.time_stretch(y, rate=1)
                mfcc = librosa.feature.mfcc(y=y_stretched, sr=sr, n_mfcc=13)
                mfcc = tf.image.resize(np.expand_dims(mfcc, axis=-1), target_shape)
                all_train_data.append(mfcc.numpy())
                all_labels.append(label_dict[emotion])
                # Additional augmentation for certain emotions
                if emotion in ["ANG", "SAD", "DIS", "FEA", "HAP", "NEU"]:
                    y_stretched = le.time_stretch(y, rate=1.3)
                    mfcc = librosa.feature.mfcc(y=y_stretched, sr=sr, n_mfcc=13)
                    mfcc = tf.image.resize(np.expand_dims(mfcc, axis=-1), target_shape)
                    all_train_data.append(mfcc.numpy())
                    all_labels.append(label_dict[emotion])
            pbar.update(1)

    return np.array(all_train_data), np.array(all_labels)

    all_train_data, all_labels = process_files(emotions, emotion_dir_names, label_dict, max_files=1000)

def load_and_process_random_audio(emotions, emotion_dir_names, max_files=1000):
    all_files = []
    for emotion in emotions:
        emotion_files = load_emotion_files(emotion, emotion_dir_names[emotion], max_files)
        all_files.extend(emotion_files)

    random_file_path = random.choice(all_files)

    if not os.path.exists(random_file_path):
        print(f"File {random_file_path} does not exist.")
        return None, None

    print(f"File {random_file_path} found. Proceeding with processing.")

    y, sr = librosa.load(random_file_path, sr=None)
    ipd.display(ipd.Audio(data=y, rate=sr))

    return y, sr

print("All data processing complete!")
print(f"Total samples: {all_train_data.shape[0]}")
print(f"Shape of each sample: {all_train_data.shape[1:]}")

all_files = []
for emotion in emotions:
    emotion_files = load_emotion_files(emotion, emotion_dir_names[emotion], max_files=1000)
    all_files.extend(emotion_files)

random_file_path = random.choice(all_files)

if not os.path.exists(random_file_path):
    print(f"File {random_file_path} does not exist.")
else:
    print(f"File {random_file_path} found. Proceeding with processing.")

    y, sr = librosa.load(random_file_path, sr=None)
ipd.display(ipd.Audio(data=y, rate=sr))

plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Original Audio Waveform')
plt.show()

y_stretched = le.time_stretch(y, rate=1)
mfcc = librosa.feature.mfcc(y=y_stretched, sr=sr, n_mfcc=13)
mfcc_resized = tf.image.resize(np.expand_dims(mfcc, axis=-1), target_shape)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
librosa.display.specshow(np.squeeze(mfcc_resized), x_axis='time')
plt.colorbar()
plt.title('Resized MFCC')
plt.tight_layout()
plt.show()

y_preprocessed = librosa.feature.inverse.mfcc_to_audio(mfcc)
ipd.display(ipd.Audio(data=y_preprocessed, rate=sr))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_train_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

# Convert labels to categorical format
y_train = to_categorical(y_train, num_classes=len(emotions))
y_test = to_categorical(y_test, num_classes=len(emotions))

# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile the model
input_shape = (target_shape[0], target_shape[1], 1)
num_classes = len(emotions)
model = create_cnn_model(input_shape, num_classes)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate average accuracy
accuracy = np.mean(y_pred_classes == y_true)
print(f'Average Accuracy: {accuracy}')

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Calculate evaluation metrics
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=emotions))


print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=emotions, yticklabels=emotions)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs[::2], history.history['accuracy'][::2], marker='o', linestyle='-', color='b')
plt.plot(epochs[::2], history.history['val_accuracy'][::2], marker='o', linestyle='-', color='g')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 2)
epochs = range(1, len(history.history['loss']) + 1)
plt.plot(epochs[::2], history.history['loss'][::2], marker='o', linestyle='-', color='b')
plt.plot(epochs[::2], history.history['val_loss'][::2], marker='o', linestyle='-', color='g')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

