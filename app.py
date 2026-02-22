import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pickle

st.title("🫁 Cough Detection System")
st.subheader("YUGO Hackathon 2026 | Team Binary Brains")
st.warning("⚠️ This is a research prototype. Not a medical device.")

# API Documentation
with st.expander("📡 API Documentation"):
    st.markdown("""
    ### REST API Endpoint
    **Endpoint:** `POST /predict`
    
    **Example:**
    ```bash
    curl -X POST http://localhost:8000/predict \\
      -F "file=@your_audio.wav"
    ```
    
    **Response:**
    ```json
    {
      "prediction": "COUGH DETECTED",
      "confidence": 95.2,
      "cough_segments_found": 12,
      "cough_events": [
        {"start": 1.2, "end": 1.7, "confidence": 95.2, "severity": "severe"}
      ]
    }
    ```
    """)

uploaded_file = st.file_uploader("Upload a WAV file", type=['wav'])

if uploaded_file:
    st.audio(uploaded_file)

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Load model - handles both old and new format
    raw = pickle.load(open("model.pkl", "rb"))
    if isinstance(raw, dict):
        model = raw['model']
        threshold = raw['threshold']
    else:
        model = raw
        threshold = 0.5

    with st.spinner("Analyzing audio..."):
        y, sr = librosa.load("temp.wav", sr=750)

        segment_duration = 0.5
        segment_samples = int(sr * segment_duration)
        step_samples = int(sr * 0.3)

        cough_count = 0
        max_prob = 0.0
        cough_times = []
        cough_events = []

        for i in range(0, len(y) - segment_samples, step_samples):
            chunk = y[i:i + segment_samples]
            if len(chunk) < segment_samples:
                continue
            mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
            features = np.concatenate([mfccs.mean(axis=1), mfccs.std(axis=1)]).reshape(1, -1)
            prob = model.predict_proba(features)[0][1]

            if prob > max_prob:
                max_prob = prob
            if prob >= threshold:
                cough_count += 1
                t_start = round(i / sr, 1)
                t_end = round((i + segment_samples) / sr, 1)
                severity = "mild" if prob < 0.7 else "moderate" if prob < 0.9 else "severe"
                cough_times.append(f"{t_start}s - {t_end}s | confidence: {prob*100:.1f}% | {severity}")
                cough_events.append({
                    "start": t_start,
                    "end": t_end,
                    "confidence": round(prob*100, 1),
                    "severity": severity
                })

    prediction = "🔴 COUGH DETECTED" if cough_count > 0 else "🟢 NO COUGH"

    st.metric("Prediction", prediction)
    st.metric("Max Confidence", f"{max_prob*100:.1f}%")
    st.metric("Cough Segments Found", cough_count)

    if cough_times:
        st.subheader("⏱️ Cough detected at:")
        for t in cough_times[:20]:
            st.write(f"• {t}")

    # JSON Output
    st.subheader("📄 JSON Output")
    result_json = {
        "prediction": "COUGH DETECTED" if cough_count > 0 else "NO COUGH",
        "confidence": round(max_prob*100, 1),
        "cough_segments_found": cough_count,
        "cough_events": cough_events[:20]
    }
    st.json(result_json)

    # Spectrogram
    fig, ax = plt.subplots()
    y_short = y[:sr*30]
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_short)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", ax=ax)
    ax.set_title("Spectrogram (first 30 seconds)")
    st.pyplot(fig)

st.markdown("---")
st.caption("Binary Brains | SSN College of Engineering | 2026")
