import streamlit as st
import pickle
import librosa
import numpy as np

# Load your pre-trained model with pickle
with open('modelForPrediction1.sav', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to extract audio features
def extract_feature(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(y)), sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack((mfccs, chroma, mel))

# Streamlit app
def main():
    st.title("Audio Analysis App")

    # Upload file through Streamlit
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav", start_time=0)

        # Extract features and make prediction
        features = extract_feature(uploaded_file)
        prediction = model.predict([features])

        # Convert the predicted class to string before applying 'upper()'
        predicted_class = str(prediction[0])

        # Display the analysis results
        st.subheader("Analysis Results")
        st.markdown(f"<h1 style='text-align: center; color: Red;'>Predicted Class: {predicted_class.upper()}</h1>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
