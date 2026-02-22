import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pickle

st.title("🫁 Cough Detection System")
st.subheader("YUGO Hackathon 2026 | Team Binary Brains")
st.warning("⚠️ This is a research prototype. Not a medical device.")

uploaded_file = st.file_uploader("Upload a WAV file", type=['wav'])

if uploaded_file:
    st.audio(uploaded_file)

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Analyzing audio..."):
        # sr=750 exactly as model was trained
        y, sr = librosa.load("temp.wav", sr=750)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.concatenate([mfccs.mean(axis=1), mfccs.std(axis=1)]).reshape(1, -1)

        model = pickle.load(open("model.pkl", "rb"))
        prob = model.predict_proba(features)[0][1]

    prediction = "🔴 COUGH DETECTED" if prob > 0.5 else "🟢 NO COUGH"
    severity = "mild" if prob < 0.7 else "moderate" if prob < 0.9 else "severe"

    st.metric("Prediction", prediction)
    st.metric("Confidence", f"{prob*100:.1f}%")
    if prob > 0.5:
        st.metric("Severity", severity.upper())

    fig, ax = plt.subplots()
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y[:sr*30])), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", ax=ax)
    ax.set_title("Spectrogram (first 30 seconds)")
    st.pyplot(fig)

st.markdown("---")
st.caption("Binary Brains | SSN College of Engineering | 2026")
