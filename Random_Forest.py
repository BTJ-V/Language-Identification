import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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
    return mfccs.flatten()


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

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=56)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)


