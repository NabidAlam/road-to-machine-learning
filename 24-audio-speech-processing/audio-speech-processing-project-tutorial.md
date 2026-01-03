# Audio and Speech Processing Project Tutorial

Step-by-step tutorial: Speech Recognition with Whisper.

## Project: Speech Recognition with Whisper

### Objective

Build a speech recognition system using OpenAI Whisper.

### Step 1: Setup

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
```

### Step 2: Load Model

```python
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
model.eval()
```

### Step 3: Load Audio

```python
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

```python
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

```python
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

