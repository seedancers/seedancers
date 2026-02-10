# 🎬 SeeDancers

**A fully autonomous AI agent that writes, directs, and produces a new short film every 30 minutes.**

No human input needed. The agent picks a story, chooses a visual style, generates the film with Seedance, adds narrator voiceover, and posts it to X/Twitter automatically.

## How It Works

1. **Story Generation** — Claude AI writes an original atmospheric narrative
2. **Visual Style** — The agent selects from styles like anime, wuxia, watercolor, noir, fantasy
3. **Video Generation** — 4 clips are generated with Seedance and stitched into a 48 second film
4. **Voiceover** — Kokoro TTS narrates the story in English or Chinese (alternating each post)
5. **Background Music** — Ambient soundtrack generated and mixed under the voiceover
6. **Auto Post** — Final video + narrative text posted to X every 30 minutes

The entire pipeline runs autonomously on a cloud server. No human touches any part of the process after the agent is started.

## Features

🎨 Multiple visual styles (anime, wuxia, watercolor, noir, fantasy, and more)

🗣️ Bilingual narration alternating between English and Chinese

🎵 Auto generated ambient background music

🔄 New unique story every 30 minutes, never repeats

🛡️ Crash recovery with timestamp tracking

📐 16:9 cinematic format

## Tech Stack

- **Story Writing** — Claude API (Anthropic)
- **Video Generation** — Seedance 1.5 Pro via fal.ai
- **Voiceover** — Kokoro TTS via fal.ai
- **Audio** — FFmpeg for stitching, BGM, and voiceover mixing
- **Posting** — X/Twitter API v2
- **Hosting** — Ubuntu cloud server

## Setup

1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/seedancers.git
cd seedancers
```

2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install anthropic fal-client tweepy requests
```

3. Configure your API keys in `config.env`
```env
ANTHROPIC_API_KEY=your_key
FAL_KEY=your_key
X_API_KEY=your_key
X_API_SECRET=your_key
X_ACCESS_TOKEN=your_key
X_ACCESS_SECRET=your_key
```

4. Run in test mode
```bash
python main.py --test
```

5. Run in production (posts every 30 minutes)
```bash
python main.py
```

## Cost Per Video

| Component | Cost |
|-----------|------|
| Seedance (4 clips) | ~$0.80 |
| Claude API | ~$0.01 |
| Kokoro TTS | ~$0.004 |
| **Total** | **~$0.82** |

## Follow Us

🐦 X/Twitter: [@SeeDancers](https://x.com/SeeDancers)

## License

MIT License
