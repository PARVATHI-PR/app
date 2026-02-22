
import pickle
import librosa
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--offline', action='store_true')
args = parser.parse_args()

print("=== Binary Brains Cough Detection Demo ===")
print("Loading model...")
model = pickle.load(open('model.pkl', 'rb'))

print("Loading audio...")
y, sr = librosa.load('005_No_Talking_In.wav', sr=750)

print("Extracting features...")
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
features = np.concatenate([mfccs.mean(axis=1), mfccs.std(axis=1)]).reshape(1,-1)

print("Predicting...")
prob = model.predict_proba(features)[0][1]
prediction = "COUGH DETECTED" if prob > 0.5 else "NO COUGH"

result = {
    "prediction": prediction,
    "confidence": round(float(prob)*100, 2),
    "model_accuracy": "91.67%",
    "model_auc": "0.957"
}

with open('outputs/demo_result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nResult:", prediction)
print("Confidence:", round(float(prob)*100, 2), "%")
print("Output saved to outputs/demo_result.json")
print("\n=== Demo Complete ===")
