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

    # Load full audio
    y, sr = librosa.load("temp.wav", sr=22050)

    # Split audio into 1-second chunks
    chunk_size = sr * 1  # 1 second per chunk
    chunks = [y[i:i+chunk_size] for i in range(0, len(y), chunk_size)]

    model = pickle.load(open("model.pkl", "rb"))

    cough_count = 0
    max_prob = 0.0
    cough_times = []

    for idx, chunk in enumerate(chunks):
        if len(chunk) < chunk_size // 2:
            continue  # skip very short chunks
        mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
        features = np.concatenate([mfccs.mean(axis=1), mfccs.std(axis=1)]).reshape(1, -1)
        prob = model.predict_proba(features)[0][1]
        if prob > max_prob:
            max_prob = prob
        if prob > 0.5:
            cough_count += 1
            cough_times.append(f"{idx}s - {idx+1}s (confidence: {prob*100:.1f}%)")

    # Final result
    if cough_count > 0:
        prediction = "🔴 COUGH DETECTED"
    else:
        prediction = "🟢 NO COUGH"

    st.metric("Prediction", prediction)
    st.metric("Max Confidence", f"{max_prob*100:.1f}%")
    st.metric("Cough Segments Found", cough_count)

    if cough_times:
        st.subheader("⏱️ Cough detected at:")
        for t in cough_times:
            st.write(f"• {t}")

    # Spectrogram
    fig, ax = plt.subplots()
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", ax=ax)
    ax.set_title("Spectrogram")
    st.pyplot(fig)

st.markdown("---")
st.caption("Binary Brains | SSN College of Engineering | 2026")
