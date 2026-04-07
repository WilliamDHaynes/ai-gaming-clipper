# AI Gaming Clipper

An automated, two-stage Python pipeline that takes raw, multi-track gameplay footage, finds the best moments using local AI transcription (Whisper), and uses OpenAI to generate cut vertical clips for TikTok/Shorts.

## Prerequisites
1. **FFmpeg**: Must be installed on your system
2. **OpenAI API Key**: For the Stage 2 clipping

## Setup
1. Clone this repository and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the root folder and add your API key:
   ```env
   OPENAI_API_KEY=sk-your-api-key-here
   ```
3. Create a folder named `raw_videos` in the root directory and drop your `.mp4` gaming session inside. *(Note: The script assumes your video has game audio on Track 2 and mic audio on Track 3. Adjust the FFmpeg map settings in Stage 1 if your OBS setup is different).*

## How to Run

### Stage 1: Audio Extraction & AI Transcription
Extracts the audio, finds high-volume action spikes, and generates a timestamped transcript using `faster-whisper`.
```bash
python stage1_ai.py
```

### Stage 2: Scene Ranking & Video Export
Feeds the scenes to OpenAI (gpt-4o-mini) to find the funniest/most intense moments, then uses FFmpeg to export perfectly timed, vertical 9:16 clips into a new `clips/` folder.
```bash
python stage2_clipper.py
```