import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import librosa
import numpy as np


def extract_mfcc_fixed_duration(audio_path, duration=10, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)

    target_length = int(duration * sr)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs


csv_file_path = '/Users/bhanutejavaravenkata/Desktop/audio_d.csv'
df = pd.read_csv(csv_file_path)

features = []
labels = []

for index, row in df.iterrows():
    file_path = row['File Path']
    label = row['Label']

    mfcc_features = extract_mfcc_fixed_duration(file_path, n_mfcc=13)

    features.append(mfcc_features)
    labels.append(label)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)


max_sequence_length = max(len(seq[0]) for seq in features)
features_padded = [np.pad(seq, ((0, 0), (0, max_sequence_length - len(seq[0]))), mode='constant') for seq in features]

X = np.array(features_padded)
y = np.array(encoded_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train.reshape((*X_train.shape, 1)), y_train, epochs=50, batch_size=64, validation_split=0.2)

# Save the trained model
model.save('Language_recognition_CNN.h5')

# Load the trained model
model = load_model('Language_recognition_CNN.h5')

predictions_probabilities = model.predict(X_test.reshape((*X_test.shape, 1)))
predictions = np.argmax(predictions_probabilities, axis=1)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)
