import os, librosa, json, subprocess, gc, torch, glob
import numpy as np
from faster_whisper import WhisperModel

# --- CONFIG ---
list_of_files = glob.glob('./raw_videos/*.mp4')
if not list_of_files:
    raise Exception("CRITICAL ERROR: No .mp4 files found in ./raw_videos/")

VIDEO_PATH = max(list_of_files, key=os.path.getctime)
COMMS_AUDIO = "comms_temp.wav" # Mic + Discord
GAME_AUDIO = "game_temp.wav"
CACHE_FILE = "transcript_cache.json"

def extract_audio():
    print(">>> Stage 1: Multi-Track Audio Extraction & Mix...")
    abs_video_path = os.path.abspath(VIDEO_PATH)
    
    # 1. Extract Game Audio for spike detection (Assuming Track 2 / 0:a:1)
    cmd_game = (
        f'ffmpeg -i "{abs_video_path}" -map 0:a:1 -vn '
        f'-ar 16000 -ac 1 -y "{GAME_AUDIO}" -loglevel error'
    )
    subprocess.run(cmd_game, shell=True)

    # 2. Mix Discord (0:a:0) and Mic (0:a:2) for Whisper Transcription
    # *Note: If your OBS track layout is different, adjust the [0:a:0] and [0:a:2] here*
    cmd_comms = (
        f'ffmpeg -i "{abs_video_path}" -filter_complex "[0:a:0][0:a:2]amix=inputs=2:duration=longest" '
        f'-vn -ar 16000 -ac 1 -y "{COMMS_AUDIO}" -loglevel error'
    )
    subprocess.run(cmd_comms, shell=True)

def detect_spikes(audio_path):
    print(">>> Analyzing audio for adaptive spikes...")
    y, sr = librosa.load(audio_path, sr=16000)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.frames_to_time(range(len(onset_env)), sr=sr)
    
    # ADAPTIVE THRESHOLD: Mean + 2.5x Standard Deviation
    mean_onset = np.mean(onset_env)
    std_onset = np.std(onset_env)
    adaptive_threshold = mean_onset + (2.5 * std_onset)
    
    print(f"    -> Calculated dynamic threshold: {round(adaptive_threshold, 2)}")
    spikes = [round(t, 2) for t, vol in zip(times, onset_env) if vol > adaptive_threshold]
    
    del y
    gc.collect()
    return spikes

def run_ai():
    extract_audio()
    spikes = detect_spikes(GAME_AUDIO)
    
    print(">>> Starting AI Transcription (Squad Comms Mix)...")
    model = WhisperModel("base", device="cuda", compute_type="float16")
    
    # We pass an initial prompt to bias the model towards gamer slang/callouts
    gaming_hints = "PMC, target down, push, flanking, dead, heal, contact, got him, armor, raid, extract, haha, lol, wait, bro."
    
    segments, info = model.transcribe(
        COMMS_AUDIO, 
        beam_size=5, 
        vad_filter=True,
        initial_prompt=gaming_hints,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    full_transcript = []
    for s in segments:
        full_transcript.append({
            "start": round(s.start, 2), 
            "text": s.text.strip()
        })
        print(f"[{round(s.start, 2)}s] {s.text.strip()}")

    output_data = {
        "transcript": full_transcript,
        "spikes": spikes[:1000] 
    }

    print(f"\n>>> Saving results to {CACHE_FILE}...")
    with open(CACHE_FILE, "w") as f:
        json.dump(output_data, f, indent=4)
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print(f">>> SUCCESS: Processing complete.")

if __name__ == "__main__":
    run_ai()