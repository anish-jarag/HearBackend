import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, Conv2D, MaxPooling2D, Flatten
import joblib

# Load metadata from both datasets
metadata_hear4u = pd.read_csv('D:/Sound_Recognition/archive1/Hear4U_dataset/meta/hear4u.csv')
metadata_urbansound = pd.read_csv('D:/Sound_Recognition/archive/UrbanSound8K.csv')

# Function to extract features from audio files
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, sr=44100)
        n_fft = min(1024, len(audio))  # Ensure n_fft does not exceed the length of the audio
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, fmax=8000, n_fft=n_fft)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_fft=n_fft)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_fft=n_fft)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)

        # Add additional features
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        rmse = librosa.feature.rms(y=audio)

        # Combine features into a single array
        features = np.hstack((
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(mel.T, axis=0),
            np.mean(contrast.T, axis=0),
            np.mean(tonnetz.T, axis=0),
            np.mean(zcr.T, axis=0),
            np.mean(rmse.T, axis=0)
        ))
        
        return features
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None

# Function to process metadata and extract features
def process_metadata(metadata, base_path, file_column, label_column, fold_column=None):
    features = []
    labels = []
    
    for index, row in metadata.iterrows():
        # Check if fold_column exists (used for UrbanSound8K dataset)
        if fold_column:
            file_name = os.path.join(base_path, f'fold{row[fold_column]}', row[file_column])
        else:
            file_name = os.path.join(base_path, row[file_column])
        
        if not os.path.exists(file_name):
            print(f"File does not exist: {file_name}")
            continue
        
        print(f"Processing file: {file_name}")
        feature = extract_features(file_name)
        
        if feature is not None:
            features.append(feature)
            labels.append(row[label_column])
        else:
            print(f"Failed to extract features from {file_name}")
    
    return features, labels

# Process Hear4U dataset    
features_hear4u, labels_hear4u = process_metadata(
    metadata_hear4u, 
    'D:/Sound_Recognition/archive1/Hear4U_dataset', 
    "filename", 
    "category"
)

# Process UrbanSound8K dataset
features_urbansound, labels_urbansound = process_metadata(
    metadata_urbansound, 
    'D:/Sound_Recognition/archive', 
    "slice_file_name", 
    "class", 
    "fold"  # Adding 'fold' column for UrbanSound8K dataset
)

# Combine features and labels from both datasets
all_features = np.vstack((features_hear4u, features_urbansound))
all_labels = labels_hear4u + labels_urbansound

# Encode labels
le = LabelEncoder()
encoded_labels = to_categorical(le.fit_transform(all_labels))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_features, encoded_labels, test_size=0.2, random_state=42)

# Reshape for CNN input
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)  # Reshape for CNN
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

# Build the CNN model
model_cnn = Sequential()
model_cnn.add(Conv2D(32, (3, 1), activation='relu', input_shape=(X_train_cnn.shape[1], 1, 1)))
model_cnn.add(MaxPooling2D(pool_size=(2, 1)))
model_cnn.add(Conv2D(64, (3, 1), activation='relu'))
model_cnn.add(MaxPooling2D(pool_size=(2, 1)))
model_cnn.add(Flatten())
model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(Dropout(0.5))
model_cnn.add(Dense(len(le.classes_), activation='softmax'))

# Compile the model
model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history_cnn = model_cnn.fit(X_train_cnn, y_train, batch_size=32, epochs=100, validation_data=(X_test_cnn, y_test), verbose=1)

# Evaluate the model
score = model_cnn.evaluate(X_test_cnn, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Save the trained model and LabelEncoder
model_cnn.save('D:/Sound_Recognition/New_Model/Sound_Classifier_CNN.h5')
joblib.dump(le, 'D:/Sound_Recognition/New_Model/Sound_label.pkl')

print("Model and LabelEncoder have been saved successfully.")

# Function to predict label of a sound file
def predict_sound_label(file_name):
    feature = extract_features(file_name)
    if feature is None:
        print(f"Unable to extract features from {file_name}")
        return None
    
    feature = feature.reshape(1, -1)
    predicted_vector = model_cnn.predict(feature.reshape(1, feature.shape[0], 1, 1))
    predicted_label = le.inverse_transform(np.argmax(predicted_vector, axis=1))
    return predicted_label[0]

# Test with a new or random sound file
random_file_name = 'D:/Sound_Recognition/archive/fold1/7061-6-0-0.wav'
predicted_label = predict_sound_label(random_file_name)
print(f"The predicted label for the sound is: {predicted_label}")
