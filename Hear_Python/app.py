import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import joblib
import gradio as gr

# Load the saved model and LabelEncoder
model = load_model('D:/Sound_Recognition/New_Model/df/Sound_Classifier_CNN.h5')
le = joblib.load('D:/Sound_Recognition/New_Model/df/Sound_label.pkl')

# Function to extract features from audio files
def extract_features(file_name, target_size=195):
    try:
        audio, sample_rate = librosa.load(file_name, sr=44100)
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, fmax=8000)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)

        features = np.hstack((
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(mel.T, axis=0),
            np.mean(contrast.T, axis=0),
            np.mean(tonnetz.T, axis=0)
        ))

        if len(features) < target_size:
            features = np.pad(features, (0, target_size - len(features)), 'constant')
        elif len(features) > target_size:
            features = features[:target_size]
        
        return features
    except Exception as e:
        return None, str(e)

# Function to plot spectrogram
def plot_spectrogram(file_name):
    y, sr = librosa.load(file_name, sr=44100)
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='coolwarm')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    
    # Return the image as a plot, not saving to file
    return plt.gcf()

# Function to predict the label of a sound file
def predict_sound_label(file_name):
    feature = extract_features(file_name)
    if feature is None:
        return "Unable to extract features from the audio file.", None, None
    
    feature = feature.reshape(1, feature.shape[0], 1, 1)
    predicted_vector = model.predict(feature)
    
    predicted_label = le.inverse_transform(np.argmax(predicted_vector, axis=1))
    class_probabilities = predicted_vector.flatten()
    
    return predicted_label[0], plot_spectrogram(file_name), class_probabilities

# Function to create a probability bar chart
def create_probability_chart(class_probabilities):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=le.classes_, y=class_probabilities, ax=ax)
    plt.title('Class Probabilities')
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    
    # Return the plot without saving
    return fig

# Function to get audio file metadata
def get_file_metadata(file_name):
    audio, sr = librosa.load(file_name, sr=None)
    duration = librosa.get_duration(y=audio, sr=sr)
    return {"Duration (s)": duration, "Sample Rate (Hz)": sr}

# Main function to process audio
def process_audio(file_name):
    predicted_label, spectrogram_img, class_probabilities = predict_sound_label(file_name)
    metadata = get_file_metadata(file_name)
    prob_chart = create_probability_chart(class_probabilities)
    
    return predicted_label, spectrogram_img, prob_chart, metadata

# Create the Gradio interface
iface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath"),
    outputs=["text", "plot", "plot", "json"],  # Change to 'plot' for displaying charts
    title="Sound Recognition Model",
    description="Upload an audio file to recognize the sound category.",
)

iface.launch()
