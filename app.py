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

    st.info("Loading audio...")
    # Load at lower sample rate to save memory
    y, sr = librosa.load("temp.wav", sr=16000)

    # 2-second chunks, stepping every 2 seconds (no overlap) = fewer chunks
    chunk_size = sr * 2
    chunks = [y[i:i+chunk_size] for i in range(0, len(y), chunk_size)]

    model = pickle.load(open("model.pkl", "rb"))

    cough_count = 0
    max_prob = 0.0
    cough_times = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, chunk in enumerate(chunks):
        progress_bar.progress(int((idx + 1) / len(chunks) * 100))
        status_text.text(f"Scanning segment {idx+1} of {len(chunks)}...")

        if len(chunk) < sr:
            continue

        mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
        features = np.concatenate([mfccs.mean(axis=1), mfccs.std(axis=1)]).reshape(1, -1)
        prob = model.predict_proba(features)[0][1]

        if prob > max_prob:
            max_prob = prob
        if prob > 0.5:
            cough_count += 1
            cough_times.append(f"{idx*2}s - {idx*2+2}s (confidence: {prob*100:.1f}%)")

    status_text.text("✅ Analysis complete!")
    progress_bar.progress(100)

    # Final result
    prediction = "🟢 NO COUGH" if cough_count < 0 else "🔴 COUGH DETECTED"

    st.metric("Prediction", prediction)
    st.metric("Max Confidence", f"{max_prob*100:.1f}%")
    st.metric("Cough Segments Found", cough_count)

    if cough_times:
        st.subheader("⏱️ Cough detected at:")
        for t in cough_times:
            st.write(f"• {t}")

    # Spectrogram of first 30 seconds only
    fig, ax = plt.subplots()
    y_short = y[:sr*30]
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_short)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", ax=ax)
    ax.set_title("Spectrogram (first 30 seconds)")
    st.pyplot(fig)

st.markdown("---")
st.caption("Binary Brains | SSN College of Engineering | 2026")
