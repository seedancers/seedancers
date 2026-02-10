#!/usr/bin/env python3
"""
X Video Automation with Seedance 1.5 Pro
=========================================
Generates AI micro-stories as 48-second videos and posts them to X every 30 minutes.

Architecture:
1. Claude API → generates 4-scene story with visual prompts + atmospheric narration
2. Kokoro TTS → generates narrator voiceover (EN or CN)
3. fal.ai Seedance 1.5 Pro → generates 4 x 12s video clips
4. ffmpeg → stitches clips, mixes voiceover + BGM
5. Tweepy → uploads video and posts to X

For visual consistency across 48 seconds:
- Scene 1: Text-to-Video (establishes look)
- Scene 2-4: Image-to-Video (last frame of previous scene → input)
"""

import os
import sys
import json
import time
import re
import logging
import subprocess
import tempfile
import hashlib
import requests
import schedule
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ─── Setup ────────────────────────────────────────────────────────────────────
load_dotenv("config.env")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("automation.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("x_video_bot")

# ─── Config ───────────────────────────────────────────────────────────────────
CLIPS_PER_VIDEO = int(os.getenv("CLIPS_PER_VIDEO", 4))
CLIP_DURATION = os.getenv("CLIP_DURATION", "12")
RESOLUTION = os.getenv("RESOLUTION", "720p")
ASPECT_RATIO = os.getenv("ASPECT_RATIO", "16:9")
GENERATE_AUDIO = os.getenv("GENERATE_AUDIO", "true").lower() == "true"
BGM_STYLE = os.getenv("BACKGROUND_MUSIC", "cinematic")
BGM_VOLUME = float(os.getenv("BGM_VOLUME", "0.25"))
STORY_GENRE = os.getenv("STORY_GENRE", "mixed")

# Rotating visual styles - each video gets a different one
VISUAL_STYLES = [
    {"name": "realistic_cinematic", "prompt": "photorealistic, cinematic lighting, 35mm film, shallow depth of field, dramatic shadows, movie quality"},
    {"name": "anime", "prompt": "anime style, vibrant colors, Makoto Shinkai lighting, detailed backgrounds, Studio Ghibli inspired, hand-drawn feel"},
    {"name": "cartoon_fun", "prompt": "3D Pixar cartoon style, bright saturated colors, exaggerated expressions, playful, rounded shapes, family animation"},
    {"name": "dark_anime", "prompt": "dark anime, moody atmosphere, cyberpunk neon, sharp lines, Attack on Titan aesthetic, dramatic action"},
    {"name": "watercolor", "prompt": "Chinese watercolor painting style, ink wash, ethereal, traditional brush strokes, misty mountains, poetic"},
    {"name": "noir", "prompt": "black and white film noir, high contrast, venetian blind shadows, 1940s detective aesthetic, smoke and rain"},
    {"name": "fantasy", "prompt": "epic fantasy, golden hour magic, enchanted world, glowing particles, Lord of the Rings atmosphere"},
    {"name": "wuxia", "prompt": "Chinese wuxia martial arts, flowing robes, bamboo forests, sword fighting, Hero movie aesthetic, wire-fu"},
    {"name": "cute_chibi", "prompt": "chibi kawaii style, oversized heads, tiny bodies, pastel colors, adorable expressions, manga cute"},
    {"name": "horror", "prompt": "psychological horror, desaturated colors, fog, unsettling atmosphere, flickering lights, uncanny valley"},
    {"name": "retro_80s", "prompt": "synthwave retro 80s, neon grids, VHS aesthetic, chrome text, sunset gradients, outrun style"},
    {"name": "surreal", "prompt": "surrealist art, Salvador Dali inspired, dreamlike, melting clocks, impossible architecture, floating objects"},
]

# Story moods to rotate through
STORY_MOODS = ["感人", "搞笑", "悬疑", "温馨", "悲伤", "励志", "恐怖", "浪漫", "讽刺", "奇幻"]
POST_INTERVAL = int(os.getenv("POST_INTERVAL_MINUTES", 30))

WORK_DIR = Path("./output")
WORK_DIR.mkdir(exist_ok=True)
ARCHIVE_DIR = Path("./archive")
ARCHIVE_DIR.mkdir(exist_ok=True)

# Track posted stories to avoid repeats
HISTORY_FILE = Path("./story_history.json")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: STORY GENERATION (Claude API)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_story() -> dict:
    """
    Uses Claude API to generate a Chinese micro-story with rotating visual styles.
    Returns dict with visual prompts (English for Seedance) + Chinese story text for tweet.
    """
    import random
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in config.env")

    # Pick visual style - currently locked to anime
    forced_style = os.environ.get("FORCE_STYLE", "anime")
    if forced_style:
        style = next((s for s in VISUAL_STYLES if s["name"] == forced_style), VISUAL_STYLES[1])
    else:
        style = VISUAL_STYLES[1]  # anime
    mood = random.choice(STORY_MOODS)
    
    # Load history to avoid repeats
    used_concepts = []
    if HISTORY_FILE.exists():
        history = json.loads(HISTORY_FILE.read_text())
        used_concepts = [h.get("title", "") for h in history[-50:]]

    # Strictly alternate between Chinese and English
    lang_file = Path("last_language.txt")
    if lang_file.exists():
        last_lang = lang_file.read_text().strip()
        caption_language = "en" if last_lang == "zh" else "zh"
    else:
        caption_language = "en"  # Start with English
    
    if caption_language == "zh":
        caption_instruction = """配文语言：中文
写一段氛围感很强的旁白。不要描述具体动作（因为视频画面可能不完全一致），而是描述情绪、氛围、感觉。像诗一样，但不要太文艺。像在给朋友低声讲一个故事的感觉。
例子：「雨。一间咖啡馆。一个等待的人。有些人等了一辈子，等一个永远不会来的人。但偶尔，就那么偶尔，门真的会被推开。」"""
    else:
        caption_instruction = """Caption language: English
Write an atmospheric, mood-driven narration. Don't describe specific actions (the video may not match exactly), instead describe emotions, atmosphere, feelings. Poetic but not pretentious. Like whispering a story to a friend.
Example: "Rain. A café. A man who waits. Some people spend their whole lives waiting for someone who never comes. But sometimes, just sometimes, the door opens." """

    prompt = f"""你现在是一个讲故事的人。你要给一个{int(CLIPS_PER_VIDEO) * int(CLIP_DURATION)}秒AI短片写旁白配文。

视觉风格：{style['name']} ({style['prompt']})
情绪：{mood}

{caption_instruction}

配文规则：
- 描述氛围和情感，不要描述具体动作
- 不要用破折号「——」或 em dashes
- 不要用省略号来制造悬念
- 句子长短不一
- 有画面感但是抽象的，不具体到某个动作
- 让人感受到故事的情绪就够了
- 100-200字/words，短而有力

故事要求：
- 完整的微型故事，4个场景，有开头发展高潮结尾
- 最好有反转或意想不到的结局
- 视频prompts要详细具体，但配文要抽象氛围化

避免这些已用过的概念：{json.dumps(used_concepts[-20:])}

严格按以下JSON格式回复（不要markdown不要代码块）：
{{
    "title": "故事标题2到5个字",
    "chinese_story": "{'中文氛围旁白，不要hashtag' if caption_language == 'zh' else 'English atmospheric narration, no hashtags'}",
    "caption_language": "{caption_language}",
    "visual_style": "{style['name']}",
    "mood": "{mood}",
    "character_description": "Detailed English description of main character(s) - exact appearance, clothing, features. Keep consistent across all scenes.",
    "setting_description": "Detailed English description of the setting/environment",
    "scenes": [
        {{
            "scene_number": 1,
            "narrative_cn": "场景1 开头",
            "visual_prompt": "DETAILED English visual prompt for Seedance. Style: {style['prompt']}. NO dialogue, NO talking, NO speech, silent characters. Clean simple composition, smooth camera movement. Include character, action, camera angle, lighting. Max 150 words."
        }},
        {{
            "scene_number": 2,
            "narrative_cn": "场景2 发展",
            "visual_prompt": "DETAILED English visual prompt continuing story. Same character and style: {style['prompt']}. NO dialogue, NO talking, NO speech, silent characters. Clean simple composition, smooth camera movement. Max 150 words."
        }},
        {{
            "scene_number": 3,
            "narrative_cn": "场景3 高潮",
            "visual_prompt": "DETAILED English visual prompt for the climax. Same character and style: {style['prompt']}. NO dialogue, NO talking, NO speech, silent characters. Clean simple composition, smooth camera movement. Max 150 words."
        }},
        {{
            "scene_number": 4,
            "narrative_cn": "场景4 结局反转",
            "visual_prompt": "DETAILED English visual prompt for the finale/twist. Same character and style: {style['prompt']}. NO dialogue, NO talking, NO speech, silent characters. Clean simple composition, smooth camera movement. Max 150 words."
        }}
    ]
}}"""

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 2500,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=60
    )
    response.raise_for_status()
    
    result_text = response.json()["content"][0]["text"]
    
    # Clean up potential markdown formatting
    result_text = result_text.strip()
    if result_text.startswith("```"):
        result_text = re.sub(r"^```(?:json)?\s*", "", result_text)
        result_text = re.sub(r"\s*```$", "", result_text)
    
    story = json.loads(result_text)
    
    # Validate structure
    assert "scenes" in story and len(story["scenes"]) >= 3
    # Trim or pad to exactly CLIPS_PER_VIDEO scenes
    story["scenes"] = story["scenes"][:int(CLIPS_PER_VIDEO)]
    assert all("visual_prompt" in s for s in story["scenes"])
    assert "chinese_story" in story
    
    log.info(f"Generated story: '{story['title']}' | Style: {style['name']} | Mood: {mood}")
    return story


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: VIDEO GENERATION (fal.ai Seedance 1.5 Pro)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_video_clip_t2v(prompt: str, clip_index: int) -> str:
    """
    Generate a video clip using Text-to-Video (for first scene).
    Returns path to downloaded video file.
    """
    import fal_client
    os.environ['FAL_KEY'] = os.getenv('FAL_KEY')
    
    log.info(f"Generating clip {clip_index + 1} (text-to-video)...")
    
    result = fal_client.subscribe(
        "fal-ai/bytedance/seedance/v1.5/pro/text-to-video",
        arguments={
            "prompt": prompt,
            "duration": CLIP_DURATION,
            "resolution": RESOLUTION,
            "aspect_ratio": ASPECT_RATIO,
            "generate_audio": GENERATE_AUDIO,
            "enable_safety_checker": True
        },
        with_logs=True
    )
    
    video_url = result["video"]["url"]
    output_path = str(WORK_DIR / f"clip_{clip_index:02d}.mp4")
    _download_file(video_url, output_path)
    log.info(f"Clip {clip_index + 1} downloaded: {output_path}")
    return output_path


def generate_video_clip_i2v(prompt: str, image_path: str, clip_index: int) -> str:
    """
    Generate a video clip using Image-to-Video (for scenes 2+).
    Uses the last frame of the previous clip as the starting image.
    Returns path to downloaded video file.
    """
    import fal_client
    os.environ['FAL_KEY'] = os.getenv('FAL_KEY')
    
    log.info(f"Generating clip {clip_index + 1} (image-to-video for consistency)...")
    
    # Upload image to fal
    log.info(f"  Uploading last frame...")
    image_url = fal_client.upload_file(image_path)
    log.info(f"  Image uploaded: {image_url[:80]}...")
    
    result = fal_client.subscribe(
        "fal-ai/bytedance/seedance/v1.5/pro/image-to-video",
        arguments={
            "prompt": prompt,
            "image_url": image_url,
            "duration": CLIP_DURATION,
            "resolution": RESOLUTION,
            "aspect_ratio": ASPECT_RATIO,
            "generate_audio": GENERATE_AUDIO,
            "enable_safety_checker": True
        },
        with_logs=True
    )
    
    video_url = result["video"]["url"]
    output_path = str(WORK_DIR / f"clip_{clip_index:02d}.mp4")
    _download_file(video_url, output_path)
    log.info(f"Clip {clip_index + 1} downloaded: {output_path}")
    return output_path


def _download_file(url: str, output_path: str):
    """Download a file from URL to local path."""
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def generate_voiceover(text: str, language: str = "zh") -> str:
    """
    Generate narrator voiceover using Kokoro TTS via fal.ai.
    Uses a consistent calm female voice across all videos for brand identity.
    
    Models:
        - Chinese:  fal-ai/kokoro/mandarin-chinese (zf_xiaobei)
        - English:  fal-ai/kokoro/american-english (af_heart)
    
    Cost: ~$0.02 per 1000 characters ≈ $0.004 per video
    
    Returns path to audio file, or None if generation fails.
    """
    import fal_client
    os.environ['FAL_KEY'] = os.getenv('FAL_KEY')
    
    log.info(f"Generating voiceover ({language}, {len(text)} chars)...")
    
    # Model + voice selection based on language
    # SIGNATURE VOICE: consistent calm female narrator for brand identity
    tts_config = {
        "zh": {
            "model": "fal-ai/kokoro/mandarin-chinese",
            "voice": "zf_xiaobei",   # Chinese female narrator (calm, clear)
        },
        "en": {
            "model": "fal-ai/kokoro/american-english",
            "voice": "af_heart",     # English female narrator (warm, calm)
        }
    }
    
    config = tts_config.get(language, tts_config["en"])
    
    try:
        result = fal_client.subscribe(
            config["model"],
            arguments={
                "prompt": text,
                "voice": config["voice"],
                "speed": 0.85,       # Slightly slower for narrator effect
            },
            with_logs=True
        )
        
        # Extract audio URL from result
        audio_url = None
        if isinstance(result.get("audio"), dict):
            audio_url = result["audio"].get("url")
        elif isinstance(result.get("audio_url"), str):
            audio_url = result["audio_url"]
        elif isinstance(result.get("output"), dict):
            audio_url = result["output"].get("url")
        
        if not audio_url:
            log.warning(f"No audio URL found in TTS result: {list(result.keys())}")
            return None
        
        output_path = str(WORK_DIR / "voiceover.wav")
        _download_file(audio_url, output_path)
        
        # Verify audio file is valid
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", output_path],
            capture_output=True, text=True
        )
        if probe.returncode != 0:
            log.warning("Voiceover audio file invalid, skipping")
            return None
        
        vo_duration = float(json.loads(probe.stdout)["format"]["duration"])
        log.info(f"Voiceover generated: {vo_duration:.1f}s")
        return output_path
        
    except Exception as e:
        log.warning(f"TTS generation failed: {e}")
        log.warning("Continuing without voiceover")
        return None


def extract_last_frame(video_path: str, output_path: str) -> str:
    """Extract the last frame from a video using ffmpeg."""
    # Get video duration
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    
    # Extract last frame (0.1s before end to be safe)
    last_frame_time = max(0, duration - 0.1)
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(last_frame_time),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        output_path
    ], check=True, capture_output=True)
    
    log.info(f"Extracted last frame at {last_frame_time:.1f}s → {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: VIDEO STITCHING (ffmpeg)
# ═══════════════════════════════════════════════════════════════════════════════

def stitch_clips(clip_paths: list, output_path: str) -> str:
    """
    Combine multiple video clips into one seamless video using ffmpeg.
    Optionally adds background music for consistency across all clips.
    """
    log.info(f"Stitching {len(clip_paths)} clips into final video...")
    
    # Create concat file
    concat_file = str(WORK_DIR / "concat_list.txt")
    
    # First: normalize all clips to same format/codec/resolution
    normalized_clips = []
    for i, clip in enumerate(clip_paths):
        norm_path = str(WORK_DIR / f"norm_{i:02d}.mp4")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", clip,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            "-ac", "2",
            "-r", "24",  # consistent frame rate
            "-pix_fmt", "yuv420p",
            norm_path
        ], check=True, capture_output=True)
        normalized_clips.append(norm_path)
    
    # Write concat file with normalized clips
    with open(concat_file, "w") as f:
        for clip in normalized_clips:
            f.write(f"file '{os.path.abspath(clip)}'\n")
    
    # Concatenate clips
    concat_output = str(WORK_DIR / "concat_raw.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        concat_output
    ], check=True, capture_output=True)
    
    # Add background music if enabled
    if BGM_STYLE and BGM_STYLE != "none":
        log.info(f"Adding background music ({BGM_STYLE})...")
        add_background_music(concat_output, output_path)
        os.remove(concat_output)
    else:
        os.rename(concat_output, output_path)
    
    # Verify final duration
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", output_path],
        capture_output=True, text=True
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    log.info(f"Final video: {output_path} ({duration:.1f}s)")
    
    # Cleanup temp files
    for clip in normalized_clips:
        if os.path.exists(clip):
            os.remove(clip)
    os.remove(concat_file)
    
    return output_path


def add_background_music(video_path: str, output_path: str):
    """
    Generate ambient background music with ffmpeg and mix it under the video audio.
    Uses ffmpeg audio filters to create mood-appropriate BGM.
    """
    # Get video duration
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    
    # Generate BGM as separate file first, then mix
    bgm_file = video_path.replace(".mp4", "_bgm.wav")
    
    # Simple ambient pad generation
    bgm_presets = {
        "ambient": f"sine=f=220:d={duration},lowpass=f=800,tremolo=f=0.5:d=0.3,afade=t=in:d=3,afade=t=out:st={max(0,duration-3)}:d=3,volume=0.3",
        "cinematic": f"sine=f=110:d={duration},lowpass=f=600,tremolo=f=0.3:d=0.4,afade=t=in:d=4,afade=t=out:st={max(0,duration-4)}:d=4,volume=0.3",
        "emotional": f"sine=f=261:d={duration},lowpass=f=1000,tremolo=f=0.8:d=0.2,afade=t=in:d=3,afade=t=out:st={max(0,duration-3)}:d=3,volume=0.3",
        "lofi": f"sine=f=196:d={duration},lowpass=f=500,highpass=f=100,tremolo=f=2:d=0.1,afade=t=in:d=2,afade=t=out:st={max(0,duration-2)}:d=2,volume=0.3",
        "epic": f"sine=f=55:d={duration},lowpass=f=400,tremolo=f=0.2:d=0.5,afade=t=in:d=5,afade=t=out:st={max(0,duration-4)}:d=4,volume=0.3",
        "chinese_traditional": f"sine=f=293:d={duration},lowpass=f=1200,tremolo=f=1.5:d=0.15,afade=t=in:d=3,afade=t=out:st={max(0,duration-3)}:d=3,volume=0.3",
    }
    
    bgm_filter = bgm_presets.get(BGM_STYLE, bgm_presets["cinematic"])
    
    # Step 1: Generate BGM audio file
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", bgm_filter,
        "-ac", "2",
        "-ar", "44100",
        bgm_file
    ], check=True, capture_output=True)
    
    # Step 2: Mix BGM under video audio
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", bgm_file,
        "-filter_complex",
        f"[0:a]volume=1.0[va];[1:a]volume={BGM_VOLUME}[bgm];[va][bgm]amix=inputs=2:duration=first[out]",
        "-map", "0:v",
        "-map", "[out]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output_path
    ], check=True, capture_output=True)
    
    # Cleanup
    if os.path.exists(bgm_file):
        os.remove(bgm_file)
    
    log.info(f"Background music added ({BGM_STYLE}, volume: {BGM_VOLUME})")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: POST TO X (Twitter)
# ═══════════════════════════════════════════════════════════════════════════════

def post_to_x(video_path: str, story: dict) -> dict:
    """
    Upload video to X and post a tweet with Chinese story text.
    Uses OAuth1 for media upload + Client for tweet creation.
    """
    import tweepy
    
    caption = story.get("chinese_story", story.get("caption", ""))
    
    # Auth setup
    auth = tweepy.OAuth1UserHandler(
        os.getenv("X_API_KEY"),
        os.getenv("X_API_SECRET"),
        os.getenv("X_ACCESS_TOKEN"),
        os.getenv("X_ACCESS_TOKEN_SECRET")
    )
    api = tweepy.API(auth, wait_on_rate_limit=True)
    
    client = tweepy.Client(
        bearer_token=os.getenv("X_BEARER_TOKEN"),
        consumer_key=os.getenv("X_API_KEY"),
        consumer_secret=os.getenv("X_API_SECRET"),
        access_token=os.getenv("X_ACCESS_TOKEN"),
        access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET")
    )
    
    # Upload video (chunked upload for videos)
    log.info("Uploading video to X...")
    media = api.chunked_upload(
        filename=video_path,
        media_category="tweet_video",
        wait_for_async_finalize=True
    )
    
    media_id = media.media_id
    log.info(f"Video uploaded. Media ID: {media_id}")
    
    # Compose tweet text - narrator style story + competition hashtag
    chinese_story = story.get("chinese_story", caption)
    hashtag = "#BinanceAIShortDramaContest"
    tweet_text = f"{chinese_story}\n\n{hashtag}"
    if len(tweet_text) > 280:
        max_story = 280 - len(hashtag) - 5  # 5 for newlines and "..."
        tweet_text = f"{chinese_story[:max_story]}...\n\n{hashtag}"
    
    # Post tweet
    response = client.create_tweet(
        text=tweet_text,
        media_ids=[media_id]
    )
    
    tweet_id = response.data["id"]
    log.info(f"Tweet posted! ID: {tweet_id}")
    log.info(f"URL: https://x.com/i/status/{tweet_id}")
    
    return {"tweet_id": tweet_id, "media_id": media_id}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

LAST_POST_FILE = WORK_DIR / "last_post_time.txt"


def get_last_post_time():
    """Read the last post timestamp from file."""
    if LAST_POST_FILE.exists():
        try:
            ts = float(LAST_POST_FILE.read_text().strip())
            return ts
        except:
            pass
    return 0


def save_last_post_time():
    """Save current time as last post timestamp."""
    LAST_POST_FILE.write_text(str(time.time()))


def minutes_since_last_post():
    """How many minutes since the last post."""
    last = get_last_post_time()
    if last == 0:
        return float('inf')  # Never posted
    return (time.time() - last) / 60


def run_pipeline(test_mode=False):
    """Execute the full pipeline: story → video → stitch → post."""
    # Safety check: don't post if last post was too recent
    if not test_mode:
        mins = minutes_since_last_post()
        if mins < POST_INTERVAL - 1:  # 1 min tolerance
            log.info(f"Skipping: last post was only {int(mins)} min ago (need {POST_INTERVAL})")
            return True
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info(f"{'='*60}")
    log.info(f"Starting pipeline run: {timestamp} {'[TEST MODE - no posting]' if test_mode else ''}")
    log.info(f"{'='*60}")
    
    try:
        # ── Step 1: Generate Story ──
        story = generate_story()
        log.info(f"Story: {story['title']}")
        log.info(f"Tweet text: {story.get('chinese_story', '')[:100]}...")
        
        # ── Step 2: Generate Video Clips ──
        clip_paths = []
        
        for i, scene in enumerate(story["scenes"]):
            if i == 0:
                # First scene: Text-to-Video
                clip_path = generate_video_clip_t2v(
                    prompt=scene["visual_prompt"],
                    clip_index=i
                )
            else:
                # Subsequent scenes: Image-to-Video (using last frame of previous clip)
                last_frame_path = str(WORK_DIR / f"last_frame_{i-1:02d}.png")
                extract_last_frame(clip_paths[-1], last_frame_path)
                
                clip_path = generate_video_clip_i2v(
                    prompt=scene["visual_prompt"],
                    image_path=last_frame_path,
                    clip_index=i
                )
                
                # Cleanup frame
                if os.path.exists(last_frame_path):
                    os.remove(last_frame_path)
            
            clip_paths.append(clip_path)
        
        # ── Step 3: Stitch Clips ──
        final_video = str(WORK_DIR / f"final_{timestamp}.mp4")
        stitch_clips(clip_paths, final_video)
        
        # ── Step 3.5: Generate & Mix Voiceover ──
        narrator_text = story.get("chinese_story", "")
        caption_lang = story.get("caption_language", "zh")
        voiceover_path = generate_voiceover(narrator_text, caption_lang)
        
        if voiceover_path and os.path.exists(voiceover_path):
            log.info("Mixing voiceover into video...")
            video_with_vo = str(WORK_DIR / f"final_vo_{timestamp}.mp4")
            
            # Get durations to adjust timing
            vid_probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", final_video],
                capture_output=True, text=True
            )
            vid_duration = float(json.loads(vid_probe.stdout)["format"]["duration"])
            
            vo_probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", voiceover_path],
                capture_output=True, text=True
            )
            vo_duration = float(json.loads(vo_probe.stdout)["format"]["duration"])
            
            # Calculate tempo adjustment if voiceover is longer than video
            # Leave 2s padding at start and end
            target_duration = vid_duration - 4.0
            if vo_duration > target_duration and target_duration > 0:
                tempo = vo_duration / target_duration
                tempo = min(tempo, 1.5)  # Don't speed up more than 1.5x
                vo_filter = f"atempo={tempo:.2f},adelay=2000|2000,afade=t=in:d=0.5,afade=t=out:st={vid_duration-2}:d=1.5"
            else:
                vo_filter = f"adelay=2000|2000,afade=t=in:d=0.5,afade=t=out:st={vid_duration-2}:d=1.5"
            
            try:
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", final_video,
                    "-i", voiceover_path,
                    "-filter_complex",
                    f"[0:a]volume=0.25[bg];"
                    f"[1:a]{vo_filter},volume=1.2[vo];"
                    f"[bg][vo]amix=inputs=2:duration=first:dropout_transition=2[a]",
                    "-map", "0:v",
                    "-map", "[a]",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-movflags", "+faststart",
                    video_with_vo
                ], check=True, capture_output=True)
                os.remove(final_video)
                os.rename(video_with_vo, final_video)
                log.info(f"Voiceover mixed! (VO: {vo_duration:.1f}s → Video: {vid_duration:.1f}s)")
            except Exception as e:
                log.warning(f"Voiceover mixing failed: {e}")
                log.warning("Keeping video without voiceover")
        
        # Save language for alternation
        Path("last_language.txt").write_text(story.get("caption_language", "en"))
        
        if test_mode:
            # Keep video in output folder for preview, don't post
            log.info(f"")
            log.info(f"{'='*60}")
            log.info(f"TEST MODE - Video ready for preview!")
            log.info(f"Video: {os.path.abspath(final_video)}")
            log.info(f"Tweet would be:")
            log.info(f"{story.get('chinese_story', '')}")
            log.info(f"Language: {story.get('caption_language', 'zh')}")
            log.info(f"{'='*60}")
            log.info(f"")
            log.info(f"Download to your PC with:")
            log.info(f"scp root@YOUR_IP:{os.path.abspath(final_video)} ~/Downloads/")
            return True
        
        # ── Step 4: Post to X ──
        result = post_to_x(
            video_path=final_video,
            story=story
        )
        
        # ── Step 5: Archive & Log ──
        archive_path = str(ARCHIVE_DIR / f"{timestamp}_{story['title'].replace(' ', '_')[:30]}.mp4")
        os.rename(final_video, archive_path)
        
        # Save to history
        history = []
        if HISTORY_FILE.exists():
            history = json.loads(HISTORY_FILE.read_text())
        history.append({
            "timestamp": timestamp,
            "title": story["title"],
            "chinese_story": story.get("chinese_story", ""),
            "visual_style": story.get("visual_style", ""),
            "mood": story.get("mood", ""),
            "tweet_id": result["tweet_id"],
            "video_path": archive_path
        })
        HISTORY_FILE.write_text(json.dumps(history, indent=2))
        
        # Cleanup clips
        for clip in clip_paths:
            if os.path.exists(clip):
                os.remove(clip)
        
        log.info(f"Pipeline complete! Tweet: https://x.com/i/status/{result['tweet_id']}")
        save_last_post_time()
        log.info(f"Last post time saved.")
        
    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        # Don't crash the scheduler
        return False
    
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Entry point: run immediately, then schedule every N minutes."""
    test_mode = "--test" in sys.argv
    
    # Support forced style: --anime, --noir, --wuxia etc.
    for arg in sys.argv:
        if arg.startswith("--style="):
            os.environ["FORCE_STYLE"] = arg.split("=")[1]
    if "--anime" in sys.argv:
        os.environ["FORCE_STYLE"] = "anime"
    
    if test_mode:
        log.info("="*60)
        log.info("TEST MODE - will generate 1 video without posting to X")
        log.info("="*60)
        run_pipeline(test_mode=True)
        log.info("Done! Check the output/ folder for your video.")
        return
    
    log.info(f"X Video Bot started. Posting every {POST_INTERVAL} minutes.")
    log.info(f"Styles: {len(VISUAL_STYLES)} rotating styles (realistic, anime, cartoon, wuxia, ...)")
    log.info(f"Moods: {', '.join(STORY_MOODS)}")
    log.info(f"Language: Random EN/CN")
    log.info(f"Video: {CLIPS_PER_VIDEO} clips × {CLIP_DURATION}s = {int(CLIPS_PER_VIDEO) * int(CLIP_DURATION)}s")
    
    # Check when we last posted - don't spam after restart
    mins_since = minutes_since_last_post()
    if mins_since < POST_INTERVAL:
        wait_mins = int(POST_INTERVAL - mins_since)
        log.info(f"Last post was {int(mins_since)} min ago. Waiting {wait_mins} min before next post...")
    else:
        log.info(f"Last post was {int(mins_since)} min ago (or never). Running now...")
        run_pipeline()
    
    # Schedule recurring runs
    schedule.every(POST_INTERVAL).minutes.do(run_pipeline)
    
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
