import os, json, subprocess, glob
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# --- CONFIG ---
list_of_files = glob.glob('./raw_videos/*.mp4')
if not list_of_files:
    raise Exception("CRITICAL ERROR: No .mp4 files found in ./raw_videos/")
VIDEO_PATH = max(list_of_files, key=os.path.getctime)
CACHE_FILE = "transcript_cache.json"

def make_clips():
    abs_video_path = os.path.abspath(VIDEO_PATH)
    print(f">>> Stage 2: Analyzing Candidate Scenes from {os.path.basename(abs_video_path)}...")
    
    if not os.path.exists(CACHE_FILE):
        print(f"ERROR: {CACHE_FILE} not found!")
        return

    with open(CACHE_FILE, "r") as f:
        data_cache = json.load(f)
        
    transcript = data_cache.get("transcript", [])
    spikes = data_cache.get("spikes", [])

    video_duration_sec = transcript[-1]['start'] if transcript else 3600 
    target_clips = max(3, min(12, int(video_duration_sec / 900)))

    # --- 1. CANDIDATE SCENE GENERATION (Python handles the grouping) ---
    active_segments = []
    # Pad dialogue by 20s
    for t in transcript:
        active_segments.append([max(0, t['start'] - 20), t['start'] + 20])
    # Pad audio spikes by 30s
    for s in spikes:
        active_segments.append([max(0, s - 30), s + 30])
        
    if not active_segments:
        active_segments = [[0, video_duration_sec]]

    # Merge overlapping active windows into distinct "Scenes"
    active_segments.sort(key=lambda x: x[0])
    scenes = []
    for seg in active_segments:
        if not scenes:
            scenes.append(seg)
        else:
            prev = scenes[-1]
            if seg[0] <= prev[1] + 15: # Merge if scenes are within 15s of each other
                prev[1] = max(prev[1], seg[1])
            else:
                scenes.append(seg)
                
    # Build the scene string for the LLM
    formatted_scenes = ""
    for i, scene in enumerate(scenes):
        s_start, s_end = scene
        s_spikes = [s for s in spikes if s_start <= s <= s_end]
        s_trans = [t for t in transcript if s_start <= t['start'] <= s_end]
        
        # Skip scenes with dead air
        if not s_trans and not s_spikes:
            continue
            
        formatted_scenes += f"\n--- SCENE {i+1} ({round(s_start)}s to {round(s_end)}s) ---\n"
        if s_spikes:
            formatted_scenes += f"ACTION SPIKES: {s_spikes}\n"
        formatted_scenes += "TRANSCRIPT:\n"
        for t in s_trans:
            formatted_scenes += f"[{t['start']}s] {t['text']}\n"

    # --- 2. LLM FINAL RANKING ---
    prompt = (
        "You are an expert gaming video editor. I am providing you with pre-filtered 'Candidate Scenes' from a gaming session. "
        "These scenes contain game audio spikes (gunfights/explosions) and/or squad communications. "
        f"Your goal is to select the {target_clips} BEST moments from these scenes and define the exact start and end times for final clips.\n\n"
        
        "SELECTION RULES:\n"
        "1. LENGTH: Clips must be tightly paced, between 45 and 120 seconds.\n"
        "2. THE GOLD: Look for high-energy tactical callouts, funny squad banter with a punchline, or sudden panics.\n"
        "3. THE JUNK FILTER: Ignore scenes where the transcript is just boring menu management or random background noise spikes.\n"
        "f4. BOUNDARIES: Use the timestamps provided to set the exact 'start' and 'end' for the clip. Add a few seconds of bufer before the action starts.\n\n"
        
        f"{formatted_scenes[:100000]}\n\n"
        
        "STRICT FORMATTING:\n"
        "- 'name' MUST be an exciting 3-5 word title summarizing the event.\n"
        "- Return ONLY a raw, valid JSON array. Do not include markdown formatting.\n"
        "OUTPUT FORMAT:\n"
        "[{\"start\": 120.5, \"end\": 180.0, \"name\": \"Title_Goes_Here\"}]"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You strictly output raw, valid JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        raw_text = response.choices[0].message.content.strip().replace('```json', '').replace('```', '').strip()
        data = json.loads(raw_text)

        # De-duplication: Ensure clips are at least 45 seconds apart
        seen_times = []
        unique_clips = []
        for clip in data:
            if not any(abs(clip['start'] - t) < 45 for t in seen_times):
                seen_times.append(clip['start'])
                unique_clips.append(clip)

        final_selection = unique_clips[:target_clips]

        # --- 3. FINAL EXPORT ---
        os.makedirs("./clips", exist_ok=True)
        for i, clip in enumerate(final_selection):
            safe_name = "".join([x if x.isalnum() else "_" for x in clip['name']])
            out = f"./clips/{safe_name}_{i}.mp4"
            
            start_time = max(0, clip['start'] - 8)
            duration = (clip['end'] - clip['start']) + 8
            if duration > 150: duration = 150
            
            print(f">>> Exporting Clip {i+1}/{len(final_selection)} [{round(duration)}s long]: {out}")
            
            # Map all 3 tracks (Discord, Game, Mic) to the final output
            cmd = (
                f'ffmpeg -ss {start_time} -t {duration} -i "{abs_video_path}" '
                f'-map 0:v:0 -map 0:a:0 -map 0:a:1 -map 0:a:2 '
                f'-vf "crop=ih*(9/16):ih:(iw-ow)/2:0,scale=1080:1920" '
                f'-c:v libx264 -crf 18 -preset ultrafast -pix_fmt yuv420p '
                f'-c:a aac -b:a 192k -y "{out}" -loglevel error'
            )
            subprocess.run(cmd, shell=True)

        print(f"\n>>> SUCCESS: {len(final_selection)} final vertical clips ready in ./clips.")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    make_clips()