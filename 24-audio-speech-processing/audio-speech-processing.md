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
- **Bit Depth**: Resolution of each sample (bits)
- **Channels**: Mono (1) or Stereo (2)

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

**MFCCs** are commonly used features for speech recognition.

```python
# Extract MFCCs
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

# Plot
plt.figure(figsize=(12, 6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
plt.title('MFCCs')
plt.show()
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

### Using Pre-trained Models (Whisper)

**OpenAI Whisper** is a state-of-the-art ASR model.

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# Load model
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# Load audio
audio, sr = torchaudio.load("speech.wav")
audio = audio.squeeze().numpy()

# Process
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
with torch.no_grad():
    generated_ids = model.generate(inputs["input_features"])

# Decode
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)
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

### Using Pre-trained TTS (Tacotron 2)

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
import torch

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

# Text input
text = "Hello, this is a text to speech example."
inputs = processor(text=text, return_tensors="pt")

# Generate speech
with torch.no_grad():
    speech = model.generate_speech(inputs["input_ids"], 
                                   speaker_embeddings)
```

---

## Audio Classification

### Audio Event Detection

```python
import torch
import torch.nn as nn
import torchaudio

class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        # CNN for spectrogram
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x: [batch, 1, mel_bins, time_frames]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

---

## Music Generation

### Music Generation with RNN

```python
class MusicGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MusicGenerator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 
                           num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out
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

## Practice Exercises

1. **Speech Recognition**: Build ASR with Whisper
2. **Text-to-Speech**: Generate speech from text
3. **Music Classification**: Classify music by genre
4. **Audio Event Detection**: Detect events in recordings
5. **Voice Cloning**: Clone voice for TTS

---

## Key Takeaways

1. **Audio is Complex**: Time-frequency representation needed
2. **Spectrograms**: Key representation for deep learning
3. **Pre-trained Models**: Use Whisper for ASR, Tacotron for TTS
4. **Libraries**: librosa for analysis, torchaudio for deep learning
5. **Applications**: ASR, TTS, music, audio classification

---

**Next Steps**: Explore [Advanced Topics](audio-speech-processing-advanced-topics.md) for voice cloning, music generation, and real-time processing.

