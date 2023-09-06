# ReArabic


---


A Flask web application that serves multiple purposes:

1. **Video Upload & Transcription**: Users can upload videos, from which the audio is extracted, transcribed using the Whisper model, and paraphrased using the GPT model.
2. **VTT File Processing**: Users can upload VTT (Web Video Text Tracks) files for direct paraphrasing of their content.
3. **Transcription Review & Overdubbing**: Users can view transcriptions in a visually appealing interface with timestamp badges. Following the review, they can modify or record over transcribed segments and replace the original video's audio with their recording.

## Features:

- **Video Processing**: Upload videos, transcribe audio, and receive the paraphrased version displayed with time-stamped segments.

- **VTT File Handling**: Upload VTT files for direct processing and paraphrasing using GPT.

- **Interactive Transcription Display**: Review transcriptions in a styled interface with time-stamped segments, courtesy of Bootstrap.

- **Audio Recording**: After reviewing transcriptions, users can record over segments. Their audio can then replace the original video's audio.

## Requirements:

- Flask
- moviepy
- pydub
