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

    model = pickle.load(open("model.pkl", "rb"))

    with st.spinner("Analyzing audio..."):
        # Load full audio at sr=750 (exactly as trained)
        y, sr = librosa.load("temp.wav", sr=750)

        # Scan in short segments (0.5s) like training data
        segment_duration = 0.5  # seconds
        segment_samples = int(sr * segment_duration)
        step_samples = int(sr * 0.3)  # 0.3s step (overlap)

        cough_count = 0
        max_prob = 0.0
        cough_times = []

        for i in range(0, len(y) - segment_samples, step_samples):
            chunk = y[i:i + segment_samples]
            if len(chunk) < segment_samples:
                continue

            mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
            features = np.concatenate([mfccs.mean(axis=1), mfccs.std(axis=1)]).reshape(1, -1)
            prob = model.predict_proba(features)[0][1]

            if prob > max_prob:
                max_prob = prob
            if prob > 0.5:
                cough_count += 1
                t_start = round(i / sr, 1)
                t_end = round((i + segment_samples) / sr, 1)
                severity = "mild" if prob < 0.7 else "moderate" if prob < 0.9 else "severe"
                cough_times.append(f"{t_start}s - {t_end}s | confidence: {prob*100:.1f}% | {severity}")

    prediction = "🔴 COUGH DETECTED" if cough_count > 0 else "🟢 NO COUGH"

    st.metric("Prediction", prediction)
    st.metric("Max Confidence", f"{max_prob*100:.1f}%")
    st.metric("Cough Segments Found", cough_count)

    if cough_times:
        st.subheader("⏱️ Cough detected at:")
        for t in cough_times[:20]:  # show max 20
            st.write(f"• {t}")

    # Spectrogram
    fig, ax = plt.subplots()
    y_short = y[:sr*30]
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_short)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", ax=ax)
    ax.set_title("Spectrogram (first 30 seconds)")
    st.pyplot(fig)

st.markdown("---")
st.caption("Binary Brains | SSN College of Engineering | 2026")
