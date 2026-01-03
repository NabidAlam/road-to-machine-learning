# Audio and Speech Processing Quick Reference

Quick reference for audio processing, ASR, and TTS.

## Audio Processing

### Load Audio
```python
import librosa
audio, sr = librosa.load('audio.wav', sr=22050)
```

### Spectrogram
```python
D = librosa.stft(audio)
magnitude = np.abs(D)
spectrogram = librosa.amplitude_to_db(magnitude)
```

### Mel Spectrogram
```python
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec)
```

### MFCC
```python
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
```

## ASR

### Whisper
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
inputs = processor(audio, return_tensors="pt")
generated_ids = model.generate(inputs["input_features"])
transcription = processor.batch_decode(generated_ids)[0]
```

## TTS

### SpeechT5
```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
inputs = processor(text=text, return_tensors="pt")
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings)
```

## Libraries

- **librosa**: Audio analysis
- **torchaudio**: PyTorch audio processing
- **transformers**: Pre-trained ASR/TTS models

