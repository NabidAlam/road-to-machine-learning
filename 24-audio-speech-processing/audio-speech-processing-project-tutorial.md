# Audio and Speech Processing Project Tutorial

Step-by-step tutorial: Speech Recognition with Whisper.

## Project: Speech Recognition with Whisper

### Objective

Build a speech recognition system using OpenAI Whisper.

Copy the cells below onto your machine after installing `transformers` and `torchaudio`. They are tagged so the Study Hub accuracy suite does not download Whisper weights or require a `speech.wav` file.

### Step 1: Setup

```python snippet-skip
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
```

### Step 2: Load Model

```python snippet-skip
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
model.eval()
```

### Step 3: Load Audio

```python snippet-skip
audio_path = "speech.wav"
waveform, sample_rate = torchaudio.load(audio_path)

# Resample to 16kHz if needed
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)
    sample_rate = 16000

# Convert to numpy
audio = waveform.squeeze().numpy()
```

### Step 4: Process and Transcribe

```python snippet-skip
# Process audio
inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")

# Generate transcription
with torch.no_grad():
    generated_ids = model.generate(inputs["input_features"])

# Decode
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Transcription: {transcription}")
```

### Step 5: Batch Processing

```python snippet-skip
def transcribe_audio_files(audio_files):
    transcriptions = []
    for audio_file in audio_files:
        waveform, sr = torchaudio.load(audio_file)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        audio = waveform.squeeze().numpy()
        
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            generated_ids = model.generate(inputs["input_features"])
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        transcriptions.append(transcription)
    return transcriptions
```

## Extensions

1. **Real-time ASR**: Process streaming audio
2. **Language Detection**: Detect language automatically
3. **Speaker Diarization**: Identify different speakers
4. **Translation**: Translate speech to different languages

---

## Tiny local smoke (no Whisper download)

Synthetic waveform features with NumPy only. Useful before you wire Whisper.

```python
import numpy as np

rng = np.random.default_rng(0)
sr = 16000
t = np.arange(sr) / sr
# 440 Hz tone + noise (stand-in for a short clip)
wave = 0.2 * np.sin(2 * np.pi * 440 * t) + 0.01 * rng.normal(size=sr)

frame = 512
hop = 256
frames = []
for start in range(0, len(wave) - frame, hop):
    chunk = wave[start : start + frame]
    energy = float(np.mean(chunk ** 2))
    zcr = float(np.mean(np.abs(np.diff(np.sign(chunk)))) / 2)
    frames.append((energy, zcr))

feat = np.asarray(frames)
print(f"frames={feat.shape[0]} energy_mean={feat[:, 0].mean():.6f} zcr_mean={feat[:, 1].mean():.4f}")
assert feat.ndim == 2 and feat.shape[1] == 2
```
