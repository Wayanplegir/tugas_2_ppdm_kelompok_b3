import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# Load the pre-trained CNN model
model = load_model('cnn_model.h5')

# Define the emotions and corresponding labels
emotions = ["ANG", "SAD", "DIS", "FEA", "HAP", "NEU"]
label_dict = {"ANG": 0, "SAD": 1, "DIS": 2, "FEA": 3, "HAP": 4, "NEU": 5}

# Define a function to extract and resize MFCC features
def extract_mfcc(file_path, n_mfcc=13, target_shape=(128, 128)):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = tf.image.resize(mfcc, target_shape)
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
    return mfcc, y, sr

# Streamlit app
st.title('Emotion Recognition from Audio')

# Session state to manage file uploader reset
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = None

# File uploader
uploaded_files = st.file_uploader('Upload audio files', type=['wav'], accept_multiple_files=True, key='file_uploader')

# Buttons
button_col1, _, button_col2 = st.columns([1, 4, 1])
with button_col1:
    refresh_button = st.button('Refresh', key='refresh_button')
with button_col2:
    analyze_button = st.button('Analyze', key='analyze_button')

if refresh_button:
    # Clear the uploaded files and reset the session state
    st.session_state['uploaded_files'] = None
    st.experimental_rerun()

if analyze_button and uploaded_files:
    st.session_state['uploaded_files'] = uploaded_files
    for uploaded_file in uploaded_files:
        # Save the uploaded file temporarily
        file_path = uploaded_file.name
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract MFCC features
        mfcc, y, sr = extract_mfcc(file_path)
        
        # Display the audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Display the MFCC
        fig, ax = plt.subplots()
        mfcc_display = np.squeeze(mfcc, axis=0)  # Remove batch dimension for display
        librosa.display.specshow(mfcc_display[:, :, 0], sr=sr, ax=ax, x_axis='time')
        ax.set(title='MFCC')
        st.pyplot(fig)
        
        # Predict emotion
        prediction = model.predict(mfcc)
        predicted_emotion = emotions[np.argmax(prediction)]
        accuracy = np.max(prediction) * 100  # Convert to percentage
        
        # Display the prediction and accuracy
        st.markdown(f'**<span style="font-size:24px;">Predicted Emotion: {predicted_emotion} with accuracy {accuracy:.2f}%</span>**', unsafe_allow_html=True)
        
        # Clean up the temporary file
        os.remove(file_path)
