# Audio and Speech Processing Complete Guide

Comprehensive guide to audio and speech processing with deep learning.

## Table of Contents

- [Audio Signal Fundamentals](#audio-signal-fundamentals)
- [Speech Recognition (ASR)](#speech-recognition-asr)
- [Text-to-Speech (TTS)](#text-to-speech-tts)
- [Audio Classification](#audio-classification)
- [Music Generation](#music-generation)
- [Audio Libraries](#audio-libraries)
- [Practice Exercises](#practice-exercises)

---

## Audio Signal Fundamentals

### Digital Audio Representation

**Audio** is a continuous signal that must be digitized for computer processing.

**Key Concepts:**
- **Sampling Rate**: How many samples per second (Hz)
  - **44.1 kHz**: CD quality
  - **22.05 kHz**: Speech quality
  - **16 kHz**: Common for ASR
  - **8 kHz**: Telephone quality
- **Bit Depth**: Resolution of each sample (bits)
  - **16-bit**: Standard (65,536 levels)
  - **24-bit**: High quality
  - **32-bit**: Professional
- **Channels**: Mono (1) or Stereo (2)
- **Nyquist Theorem**: Sampling rate must be at least 2× the highest frequency

### Audio Formats

**Common Formats:**
- **WAV**: Uncompressed, high quality
- **MP3**: Compressed, widely used
- **FLAC**: Lossless compression
- **OGG**: Open-source compressed

### Understanding Audio Signals

**Time Domain**: Amplitude vs Time
- Shows how signal changes over time
- Useful for detecting silence, amplitude changes

**Frequency Domain**: Amplitude vs Frequency
- Shows frequency content
- Useful for identifying pitch, harmonics

**Time-Frequency Domain**: Spectrogram
- Shows both time and frequency
- Most useful for ML applications

### Waveform

**Waveform** is the time-domain representation of audio.

```python
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio
audio, sr = librosa.load('audio.wav', sr=22050)

# Plot waveform
plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(audio)) / sr, audio)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Waveform')
plt.show()
```

### Spectrogram

**Spectrogram** is the time-frequency representation showing frequency content over time.

```python
# Compute spectrogram
D = librosa.stft(audio)
magnitude = np.abs(D)
spectrogram = librosa.amplitude_to_db(magnitude)

# Plot spectrogram
plt.figure(figsize=(12, 6))
librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()
```

### Mel Spectrogram

**Mel Spectrogram** uses mel scale, which is perceptually more relevant to human hearing.

```python
# Compute mel spectrogram
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Plot
plt.figure(figsize=(12, 6))
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.show()
```

### MFCC (Mel-Frequency Cepstral Coefficients)

**MFCCs** are commonly used features for speech recognition, capturing the spectral envelope.

**Why MFCCs?**
- Perceptually relevant (mel scale matches human hearing)
- Compact representation (13-39 coefficients)
- Robust to noise
- Captures vocal tract characteristics

```python
# Extract MFCCs
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

# Plot
plt.figure(figsize=(12, 6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
plt.title('MFCCs')
plt.show()

# Delta and Delta-Delta (capture temporal dynamics)
mfcc_delta = librosa.feature.delta(mfccs)
mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
```

### Chroma Features

**Chroma** represents the 12 pitch classes (C, C#, D, ..., B).

```python
# Extract chroma features
chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

# Plot
plt.figure(figsize=(12, 6))
librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
plt.colorbar()
plt.title('Chroma Features')
plt.show()
```

### Tempo and Beat Tracking

```python
# Detect tempo and beats
tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
print(f"Tempo: {tempo:.2f} BPM")

# Plot beats on waveform
plt.figure(figsize=(12, 4))
times = librosa.frames_to_time(beats, sr=sr)
plt.plot(np.arange(len(audio)) / sr, audio)
plt.vlines(times, -1, 1, color='r', alpha=0.5, label='Beats')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Waveform with Beat Tracking')
plt.legend()
plt.show()
```

### Zero Crossing Rate (ZCR)

**ZCR** measures how often the signal crosses zero, useful for voice activity detection.

```python
# Calculate ZCR
zcr = librosa.feature.zero_crossing_rate(audio)[0]

# Plot
plt.figure(figsize=(12, 4))
times = librosa.frames_to_time(np.arange(len(zcr)), sr=sr)
plt.plot(times, zcr)
plt.xlabel('Time (s)')
plt.ylabel('ZCR')
plt.title('Zero Crossing Rate')
plt.show()
```

### Spectral Features

```python
# Spectral centroid (brightness)
centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]

# Spectral rolloff (frequency below which 85% of energy is contained)
rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]

# Spectral bandwidth
bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
```

---

## Speech Recognition (ASR)

### Overview

**Automatic Speech Recognition (ASR)** converts speech audio to text.

### Connectionist Temporal Classification (CTC)

**CTC** handles alignment between audio and text sequences.

```python
import torch
import torch.nn as nn

class CTCASR(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(CTCASR, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
    
    def forward(self, x):
        # x: [batch, time, features]
        out, _ = self.lstm(x)
        out = self.fc(out)  # [batch, time, vocab_size]
        return out
```

### Attention-Based ASR

**Attention-based ASR** uses attention mechanism for sequence-to-sequence learning.

```python
class AttentionASR(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(AttentionASR, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, 
                              num_layers=2, batch_first=True, 
                              bidirectional=True)
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim, 
                              num_layers=2, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        # Encoder
        encoder_out, _ = self.encoder(x)
        
        # Decoder with attention
        decoder_out, _ = self.decoder(encoder_out)
        attn_out, _ = self.attention(decoder_out, encoder_out, encoder_out)
        
        # Output
        out = self.fc(attn_out)
        return out
```

### Wav2Vec 2.0

**Wav2Vec 2.0** is a self-supervised pre-trained model for speech recognition.

```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio

# Load model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load and preprocess audio
audio, sr = torchaudio.load("speech.wav")
# Resample to 16kHz if needed
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    audio = resampler(audio)
audio = audio.squeeze().numpy()

# Process
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

# Decode
transcription = processor.batch_decode(predicted_ids)[0]
print(transcription)
```

### Using Pre-trained Models (Whisper)

**OpenAI Whisper** is a state-of-the-art ASR model with multilingual support.

**Whisper Models:**
- `whisper-tiny`: Fastest, smallest
- `whisper-base`: Balanced
- `whisper-small`: Better accuracy
- `whisper-medium`: High accuracy
- `whisper-large`: Best accuracy

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# Load model (choose based on speed/accuracy tradeoff)
model_name = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.eval()

# Load audio
audio, sr = torchaudio.load("speech.wav")
# Resample to 16kHz
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    audio = resampler(audio)
audio = audio.squeeze().numpy()

# Process
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    generated_ids = model.generate(
        inputs["input_features"],
        max_length=448,
        num_beams=5,
        language="en"  # Specify language
    )

# Decode
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Transcription: {transcription}")

# Multilingual transcription
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    generated_ids = model.generate(
        inputs["input_features"],
        max_length=448,
        num_beams=5
        # No language specified - auto-detects
    )
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### Real-time ASR

**Streaming ASR** processes audio in real-time as it arrives.

```python
import numpy as np
from collections import deque

class StreamingASR:
    def __init__(self, model_name="openai/whisper-base", chunk_size=16000):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.chunk_size = chunk_size  # 1 second at 16kHz
        self.buffer = deque(maxlen=5)  # Keep last 5 seconds
    
    def process_chunk(self, audio_chunk):
        """Process a chunk of audio"""
        self.buffer.extend(audio_chunk)
        
        # Process when buffer is full enough
        if len(self.buffer) >= self.chunk_size * 2:
            audio_array = np.array(self.buffer)
            inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt")
            
            with torch.no_grad():
                generated_ids = self.model.generate(inputs["input_features"])
            
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return transcription
        return None
```

---

## Text-to-Speech (TTS)

### Overview

**Text-to-Speech (TTS)** converts text to speech audio.

### Neural TTS (Tacotron)

**Tacotron** is an end-to-end neural TTS system.

```python
class Tacotron(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encoder_dim, 
                 decoder_dim, n_mels):
        super(Tacotron, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, encoder_dim, 
                              batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(encoder_dim * 2, decoder_dim, 
                              batch_first=True)
        self.mel_linear = nn.Linear(decoder_dim, n_mels)
    
    def forward(self, text):
        # Embed text
        embedded = self.embedding(text)
        
        # Encode
        encoded, _ = self.encoder(embedded)
        
        # Decode to mel spectrogram
        decoded, _ = self.decoder(encoded)
        mel = self.mel_linear(decoded)
        
        return mel
```

### Tacotron 2

**Tacotron 2** is an improved version with better quality.

**Key Improvements:**
- Location-sensitive attention
- Post-net for waveform refinement
- Better prosody control

### FastSpeech

**FastSpeech** generates speech faster with parallel decoding.

```python
# FastSpeech generates mel spectrogram in parallel
# Then uses vocoder to generate waveform
```

### Using Pre-trained TTS Models

#### SpeechT5

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
import torch
import torchaudio

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

# Text input
text = "Hello, this is a text to speech example."
inputs = processor(text=text, return_tensors="pt")

# Get speaker embeddings (for multi-speaker)
speaker_embeddings = torch.randn((1, 512))  # Or extract from reference audio

# Generate speech
with torch.no_grad():
    speech = model.generate_speech(
        inputs["input_ids"], 
        speaker_embeddings,
        vocoder=None  # Use built-in vocoder
    )

# Save audio
torchaudio.save("output.wav", speech.unsqueeze(0), 16000)
```

#### Coqui TTS

```python
# Coqui TTS provides high-quality TTS models
from TTS.api import TTS

# Initialize TTS
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

# Generate speech
tts.tts_to_file(
    text="Hello, this is a text to speech example.",
    file_path="output.wav"
)
```

### Vocoders

**Vocoders** convert mel spectrograms to waveforms.

**Popular Vocoders:**
- **WaveNet**: Original neural vocoder
- **WaveGlow**: Flow-based vocoder
- **HiFi-GAN**: Fast, high-quality vocoder
- **MelGAN**: GAN-based vocoder

```python
from transformers import SpeechT5HifiGan

# Load vocoder
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Generate waveform from mel spectrogram
waveform = vocoder(mel_spectrogram)
```

---

## Audio Classification

### Audio Event Detection

**Audio Event Detection** identifies events or sounds in audio recordings.

**Applications:**
- Environmental sound classification
- Gunshot detection
- Baby crying detection
- Doorbell detection
- Emergency sound detection

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class AudioEventClassifier(nn.Module):
    def __init__(self, num_classes, n_mels=128):
        super(AudioEventClassifier, self).__init__()
        # CNN for spectrogram
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # x: [batch, 1, mel_bins, time_frames]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
```

### Music Genre Classification

**Music Genre Classification** categorizes music by genre (rock, jazz, classical, etc.).

```python
class MusicGenreClassifier(nn.Module):
    def __init__(self, num_genres):
        super(MusicGenreClassifier, self).__init__()
        # Use pre-trained audio models
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(512, num_genres)
    
    def forward(self, x):
        return self.backbone(x)
```

### Audio-based Emotion Recognition

**Emotion Recognition** identifies emotions from speech.

**Emotions:**
- Happy, Sad, Angry, Fearful, Surprised, Disgusted, Neutral

```python
class EmotionClassifier(nn.Module):
    def __init__(self, num_emotions=7):
        super(EmotionClassifier, self).__init__()
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=13, hidden_size=128, 
                           num_layers=2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_emotions)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, mfccs):
        # mfccs: [batch, time, 13]
        lstm_out, _ = self.lstm(mfccs)
        # Use last time step
        x = lstm_out[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
```

### Voice Activity Detection (VAD)

**VAD** detects when speech is present in audio.

```python
import webrtcvad

class VAD:
    def __init__(self, aggressiveness=2):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_duration_ms = 30
        self.sample_rate = 16000
    
    def is_speech(self, audio_frame):
        """Check if frame contains speech"""
        return self.vad.is_speech(audio_frame, self.sample_rate)
    
    def detect_speech_segments(self, audio, sr):
        """Detect all speech segments in audio"""
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        frame_size = int(0.03 * 16000)  # 30ms frames
        speech_segments = []
        in_speech = False
        start = 0
        
        for i in range(0, len(audio), frame_size):
            frame = audio[i:i+frame_size]
            if len(frame) < frame_size:
                break
            
            frame_bytes = (frame * 32767).astype('int16').tobytes()
            is_speech = self.vad.is_speech(frame_bytes, 16000)
            
            if is_speech and not in_speech:
                start = i / 16000
                in_speech = True
            elif not is_speech and in_speech:
                end = i / 16000
                speech_segments.append((start, end))
                in_speech = False
        
        if in_speech:
            speech_segments.append((start, len(audio) / 16000))
        
        return speech_segments
```

### Speaker Identification and Verification

**Speaker Identification**: Identify who is speaking
**Speaker Verification**: Verify if speaker matches claimed identity

```python
class SpeakerEncoder(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, embedding_dim=128):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, 
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)
    
    def forward(self, x):
        # x: [batch, time, features] (e.g., MFCCs)
        lstm_out, _ = self.lstm(x)
        # Average pooling over time
        x = torch.mean(lstm_out, dim=1)
        x = self.fc(x)
        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        return x

class SpeakerVerifier:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
    
    def verify(self, audio1, audio2):
        """Verify if two audio samples are from same speaker"""
        emb1 = self.model(audio1)
        emb2 = self.model(audio2)
        similarity = F.cosine_similarity(emb1, emb2)
        return similarity > self.threshold
```

---

## Music Generation

### Music Generation with RNN

**RNN-based Music Generation** generates music sequences note by note.

```python
class MusicGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):
        super(MusicGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        return out, hidden
    
    def generate(self, start_token, max_length=500, temperature=1.0):
        """Generate music sequence"""
        self.eval()
        generated = [start_token]
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length):
                x = torch.tensor([[generated[-1]]])
                out, hidden = self.forward(x, hidden)
                probs = F.softmax(out[0, -1] / temperature, dim=0)
                next_token = torch.multinomial(probs, 1).item()
                generated.append(next_token)
        
        return generated
```

### Music Generation with Transformers

**Transformer-based Music Generation** uses attention for long-range dependencies.

```python
class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super(MusicTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask=None):
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x, mask=mask)
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]
```

### Music Generation with GANs

**GAN-based Music Generation** uses adversarial training.

```python
class MusicGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(MusicGenerator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z, seq_length):
        x = F.relu(self.fc1(z))
        x = x.unsqueeze(1).repeat(1, seq_length, 1)
        out, _ = self.lstm(x)
        out = self.fc2(out)
        return torch.softmax(out, dim=-1)

class MusicDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MusicDiscriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)
```

### MIDI Generation

**MIDI Generation** creates MIDI files that can be converted to audio.

```python
from mido import MidiFile, MidiTrack, Message

def generate_midi(notes, output_path="generated.mid"):
    """Generate MIDI file from note sequence"""
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    for note, duration in notes:
        track.append(Message('note_on', note=note, velocity=64, time=0))
        track.append(Message('note_off', note=note, velocity=64, time=duration))
    
    mid.save(output_path)
```

---

## Audio Libraries

### librosa

```python
import librosa

# Load audio
audio, sr = librosa.load('audio.wav', sr=22050)

# Extract features
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
```

### torchaudio

```python
import torchaudio

# Load audio
waveform, sample_rate = torchaudio.load('audio.wav')

# Transform
transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=2048,
    n_mels=128
)
mel_spec = transform(waveform)
```

---

## Speech Enhancement and Denoising

### Noise Reduction

**Speech Enhancement** improves quality of noisy speech.

```python
class SpeechDenoiser(nn.Module):
    def __init__(self):
        super(SpeechDenoiser, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, noisy_audio):
        # noisy_audio: [batch, 1, time]
        encoded = self.encoder(noisy_audio)
        enhanced = self.decoder(encoded)
        return enhanced
```

### Spectral Subtraction

**Spectral Subtraction** removes noise in frequency domain.

```python
def spectral_subtraction(noisy_audio, noise_profile, alpha=2.0):
    """Remove noise using spectral subtraction"""
    # Compute STFT
    stft = librosa.stft(noisy_audio)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise magnitude
    noise_magnitude = np.abs(librosa.stft(noise_profile))
    noise_mean = np.mean(noise_magnitude, axis=1, keepdims=True)
    
    # Subtract noise
    enhanced_magnitude = magnitude - alpha * noise_mean
    enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)  # Floor
    
    # Reconstruct
    enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
    enhanced_audio = librosa.istft(enhanced_stft)
    
    return enhanced_audio
```

## Practice Exercises

1. **Speech Recognition**: Build ASR with Whisper for multiple languages
2. **Text-to-Speech**: Generate speech from text with voice cloning
3. **Music Genre Classification**: Classify music by genre using CNNs
4. **Audio Event Detection**: Detect events (gunshots, alarms) in recordings
5. **Voice Activity Detection**: Implement VAD to detect speech segments
6. **Speaker Verification**: Build system to verify speaker identity
7. **Speech Enhancement**: Denoise noisy speech recordings
8. **Music Generation**: Generate MIDI sequences with RNN/Transformer
9. **Emotion Recognition**: Classify emotions from speech
10. **Real-time ASR**: Build streaming speech recognition system

## Resources and Further Learning

### Books and Papers

1. **"Speech and Language Processing"** - Jurafsky & Martin
   - Comprehensive NLP and speech processing textbook
   - [Free Online](https://web.stanford.edu/~jurafsky/slp3/)

2. **"Robust Speech Recognition via Large-Scale Weak Supervision"** - Radford et al., 2022 (Whisper)

3. **"wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"** - Baevski et al., 2020

4. **"Tacotron: Towards End-to-End Speech Synthesis"** - Wang et al., 2017

5. **"FastSpeech: Fast, Robust and Controllable Text to Speech"** - Ren et al., 2019

### Online Courses

1. **Speech and Language Processing** - Stanford (Jurafsky & Martin)
   - [Course Website](https://web.stanford.edu/~jurafsky/slp3/)
   - Comprehensive coverage of speech and NLP

2. **Deep Learning for Audio** - Coursera
   - Audio classification and generation

### Datasets

1. **LibriSpeech**: Large-scale English ASR dataset
2. **Common Voice**: Multilingual speech dataset (Mozilla)
3. **GTZAN**: Music genre classification dataset
4. **UrbanSound8K**: Environmental sound classification
5. **RAVDESS**: Emotional speech dataset
6. **VoxCeleb**: Speaker recognition dataset

### Tools and Libraries

1. **librosa**: Audio analysis and feature extraction
2. **torchaudio**: PyTorch audio processing
3. **transformers**: Pre-trained ASR/TTS models (Whisper, Wav2Vec, SpeechT5)
4. **Coqui TTS**: High-quality TTS models
5. **webrtcvad**: Voice activity detection
6. **soundfile**: Audio I/O
7. **pyaudio**: Real-time audio processing

### Pre-trained Models

1. **Whisper**: OpenAI's multilingual ASR
2. **Wav2Vec 2.0**: Facebook's self-supervised ASR
3. **SpeechT5**: Microsoft's TTS model
4. **Coqui TTS**: High-quality TTS models
5. **YAMNet**: Audio event classification

## Key Takeaways

1. **Audio is Complex**: Requires time-frequency representation (spectrograms)
2. **Feature Extraction**: MFCCs, mel spectrograms, chroma features
3. **Pre-trained Models**: Use Whisper for ASR, SpeechT5/Coqui for TTS
4. **Libraries**: librosa for analysis, torchaudio for deep learning
5. **Applications**: ASR, TTS, music classification, emotion recognition, VAD
6. **Real-time Processing**: Requires streaming and buffering techniques
7. **Voice Cloning**: Few-shot and zero-shot voice cloning possible
8. **Music Generation**: RNNs, Transformers, GANs for music generation
9. **Speech Enhancement**: Denoising and noise reduction techniques
10. **Speaker Recognition**: Identification and verification systems

---

**Next Steps**: Explore [Advanced Topics](audio-speech-processing-advanced-topics.md) for voice cloning, music generation with transformers, real-time processing, and speech enhancement.

