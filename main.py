from fastapi import FastAPI, WebSocket
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import librosa
import io
import wave
import json

app = FastAPI()

# Load YAMNet model from TF Hub
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Load class labels
import requests
label_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
labels_txt = requests.get(label_url).text
yamnet_labels = [line.split(',')[2].strip().strip('"') for line in labels_txt.splitlines()[1:]]

# Constants
CHUNK_SIZE = 88200  # 1 second of 44.1kHz audio
AUDIO_BUFFER = {}

USEFUL_LABELS = {
    "Dog", "Bark", "Dog bark", "Baby cry, infant cry", "Siren", "Car horn", "Gunshot, gunfire", "Fire alarm", "Speech"
}

def pcm_to_wav(pcm_bytes, sample_rate=44100, num_channels=1, sample_width=2):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buffer.seek(0)
    return buffer

@app.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    client_id = str(id(websocket))
    AUDIO_BUFFER[client_id] = b""
    print(f"🟢 Client {client_id} connected.")

    try:
        while True:
            data = await websocket.receive_bytes()
            AUDIO_BUFFER[client_id] += data

            while len(AUDIO_BUFFER[client_id]) >= CHUNK_SIZE:
                chunk = AUDIO_BUFFER[client_id][:CHUNK_SIZE]
                AUDIO_BUFFER[client_id] = AUDIO_BUFFER[client_id][CHUNK_SIZE:]

                try:
                    # Convert PCM to WAV and load
                    wav_buffer = pcm_to_wav(chunk)
                    y, sr = librosa.load(wav_buffer, sr=16000)

                    # Predict with YAMNet
                    scores, embeddings, spectrogram = yamnet_model(y)
                    mean_scores = tf.reduce_mean(scores, axis=0).numpy()

                    # Top 3 predictions
                    top_indices = mean_scores.argsort()[-3:][::-1]
                    top_labels = [(yamnet_labels[i], float(mean_scores[i])) for i in top_indices]

                    # Filter to useful labels
                    filtered = [x for x in top_labels if any(label in x[0] for label in USEFUL_LABELS)]

                    await websocket.send_text(json.dumps([
                    {"label": top_labels[0][0], "confidence": float(top_labels[0][1])}
                    ]))


                except Exception as e:
                    print("❌ Inference error:", e)
                    await websocket.send_text(f"❌ Error: {str(e)}")

    except Exception as e:
        print(f"🔴 Client {client_id} disconnected: {e}")
    finally:
        AUDIO_BUFFER.pop(client_id, None)
