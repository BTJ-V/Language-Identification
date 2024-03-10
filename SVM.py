import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import librosa
import numpy as np

csv_file_path = '/Users/bhanutejavaravenkata/Desktop/audio_d.csv'
df = pd.read_csv(csv_file_path)


def extract_mfcc_features(audio_path, duration=10, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None, duration=duration)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs


df['MFCCs'] = df['File Path'].apply(lambda x: extract_mfcc_features(x))

X = np.array(df['MFCCs'].tolist())
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

svm_classifier = SVC(kernel='rbf', C=10, gamma='scale')
svm_classifier.fit(X_train_flat, y_train)

y_pred = svm_classifier.predict(X_test_flat)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))
