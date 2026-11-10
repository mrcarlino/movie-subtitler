# Movie Subtitler

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A minimalist, Linux-native Python tool that:
1. **Downloads** online videos (YouTube, Facebook, Bilibili, Vimeo, etc.) or processes local video files using `yt-dlp`.
2. **Obtains English Subtitles**:
   - First checks if the video already provides English captions.
   - If none exist, automatically uses **local Whisper AI (`faster-whisper`)** to transcribe speech into timestamped `.srt` subtitles (accelerated by NVIDIA CUDA GPU or CPU).
   - If the video contains non-English audio (Japanese, Hindi, Spanish, etc.), Whisper directly **translates** the speech into English subtitles.
3. **Applies Subtitles**:
   - Burns in (hardcodes) crisp, readable subtitles with outline and drop shadow into the video frames so they are visible on any platform, phone, or media player.
   - Or optionally muxes them as soft selectable tracks.
   - Features real-time progress bars for both transcription and subtitle burning.

Designed for **Linux (Ubuntu / Debian / WSL2)**.

---

## 🛠️ Prerequisites

Install **FFmpeg**, **Node.js** (for `yt-dlp` YouTube cipher extraction), and **`uv`**:

```bash
# 1. Install FFmpeg and Node.js
sudo apt update && sudo apt install -y ffmpeg nodejs

# 2. Install uv (Astral Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

---

## 🚀 Setup

In your project directory (`~/Coding/movie-subtitler`), create your virtual environment and install the dependencies:

```bash
# Create the virtual environment (.venv)
uv venv

# Install requirements
uv pip install -r requirements.txt
```

*(Alternatively, with standard Python: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`)*

---

## 💻 Usage

### 1. Basic Run (Prompts for URL or File)
```bash
uv run movie_subtitler.py
```

### 2. Passing URL or Local File Directly
```bash
# Download and subtitle an online video:
uv run movie_subtitler.py "https://www.youtube.com/watch?v=..."

# Process a local video file:
uv run movie_subtitler.py "my_movie.mp4"
```

### 3. Translate Foreign Audio (e.g. Japanese) to English Subtitles

Choose the best model profile for your desired quality and speed:

#### A. Sweet Spot: High Accuracy & Natural Phrasing (Recommended)
Best balance of conversational nuance, character names, and full-sentence phrasing. Fits comfortably within 4 GB VRAM:
```bash
uv run movie_subtitler.py "<URL_OR_FILE>" -m medium -l ja --task translate --boost-vocals --nvenc
```

#### B. Maximum Translation Quality (Large-v3 quantized to fit 4 GB VRAM)
Whisper's flagship translation model. Using `--compute-type int8_float16` reduces memory usage to ~2.5 GB VRAM so it runs reliably on mobile GPUs (like RTX 3050/4050):
```bash
uv run movie_subtitler.py "<URL_OR_FILE>" -m large-v3 --compute-type int8_float16 -l ja --task translate --boost-vocals --nvenc
```

#### C. Fast Run
Lightweight and fast (~1.0 GB VRAM), good for clear speech:
```bash
uv run movie_subtitler.py "<URL_OR_FILE>" -m small -l ja --task translate --boost-vocals --nvenc
```

> **Model Selection & VRAM Guide:**
> | Model | VRAM Used | Speed (90 min film) | Translation & Phrasing Quality |
> | :--- | :--- | :--- | :--- |
> | **`small`** | ~1.0 GB | ~2–3 mins | Fast; translations can occasionally be overly literal and phrasing may break into shorter cuts. |
> | **`medium`** *(Recommended)* | ~2.2 GB | ~4–6 mins | **Significantly better conversational flow**, understands idioms, character names, and sentence structure. |
> | **`large-v3`** (`int8_float16`) | ~2.5 GB | ~6–8 mins | **Top-tier accuracy**; captures subtle context under heavy soundtracks. |

> **NVIDIA GPU in WSL2 / Linux:**  
> If you are on WSL2 or Linux with an NVIDIA GPU, Whisper and CUDA will detect your GPU automatically for fast processing without needing any manual display driver configuration.

---

## 🎵 Handling Speech Under Loud Background Music & Soundtracks

Movies, anime, and cinematic videos frequently have quiet dialogue mixed under dramatic orchestral scores, sound effects, or heavy bass. Follow these best practices for optimal results:

1. **Use `--model medium` (or `large-v3` with `int8_float16`)**:
   Larger models have substantially better acoustic attention decoders that separate and track speech phonemes even when loud instruments or action effects overlap dialogue.
2. **Enable `--boost-vocals`**:
   Applies an audio vocal bandpass filter (`120Hz–3800Hz`) and dynamic audio normalizer (`dynaudnorm`) via FFmpeg to suppress sub-bass rumble and high-frequency instrumentals while elevating human voice frequencies.
3. **Specify the Source Language (e.g. `-l ja`, `-l es`, `-l en`)**:
   Loud orchestral intros, choir chants, or ambient sound effects can confuse automatic language detection at the start of a film. Specifying `-l` locks the decoder to the correct language from the very first frame.
4. **Keep Speech VAD Disabled (Default)**:
   Voice Activity Detection (`--vad`) can misclassify speech over background music as "background noise" and discard entire lines. Leaving VAD disabled ensures speech under music is captured.
5. **Re-generating Subtitles for an Existing Video (`--force-whisper`)**:
   If a video was already downloaded or transcribed previously, use `--force-whisper` to generate new, higher-quality subtitles without re-downloading the video:
   ```bash
   # Re-transcribe with medium model and vocal isolation:
   uv run movie_subtitler.py "my_movie.mp4" -m medium -l ja --task translate --boost-vocals --force-whisper --nvenc
   ```

---

## ⚡ Fast GPU Burning (`--nvenc`)

Burn subtitles in minutes using your NVIDIA GPU instead of the CPU:
```bash
uv run movie_subtitler.py --nvenc
```

---

## ⚙️ Options Reference

| Flag | Description | Default |
|------|-------------|---------|
| `url` | Video URL or path to a local video file (prompts if omitted) | None |
| `-o`, `--output-dir` | Folder to save output videos | `./downloads` |
| `--nvenc` | Ultra-fast subtitle burning using NVIDIA GPU (NVENC) | Auto-detected |
| `-m`, `--model` | Whisper model (`tiny`, `base`, `small`, `medium`, `large-v3`) | `base` |
| `--boost-vocals` | Attenuate background soundtrack and highlight vocal speech | False |
| `--force-whisper` | Re-generate subtitles with Whisper even if `.srt` exists | False |
| `--vad` | Enable aggressive voice detection (disabled by default so dialogue under music isn't dropped) | False |
| `--device` | Processing hardware (`auto`, `cuda`, `cpu`) | `auto` |
| `--compute-type` | Precision/quantization (`float16`, `int8_float16`, `int8`, `auto`) | `auto` |
| `--task` | `auto` (English -> transcribe; foreign -> translate), `transcribe`, or `translate` | `auto` |
| `-l`, `--language` | Source audio language hint (`auto`, `en`, `ja`, etc.) | `auto` |
| `--soft` | Embed soft subtitles into MP4 instead of burning onto video | False |
| `--font-size` | Subtitle font size | `18` |
| `--font-name` | Subtitle font name | `Arial` |
| `--cookies-from-browser` | Extract cookies from browser (`chrome`, `firefox`, etc.) for private/restricted videos | None |

---

## 🌐 Common Language Codes Reference (`-l`, `--language`)

Whisper supports 99 languages. Specifying the language with `-l` (e.g. `-l ja`, `-l zh`) prevents Whisper from guessing incorrectly during musical intros, sound effects, or background noise.

Here is a quick reference of the most common film & video languages:

| Language | Code (`-l`) | Language | Code (`-l`) |
| :--- | :--- | :--- | :--- |
| **Automatic Detection** | `auto` | **Italian** | `it` |
| **Chinese (Mandarin / Cantonese)** | `zh` | **Portuguese** | `pt` |
| **Japanese** | `ja` | **Russian** | `ru` |
| **Korean** | `ko` | **Dutch** | `nl` |
| **English** | `en` | **Polish** | `pl` |
| **Spanish** | `es` | **Turkish** | `tr` |
| **French** | `fr` | **Swedish** | `sv` |
| **German** | `de` | **Ukrainian** | `uk` |
| **Hindi** | `hi` | **Vietnamese** | `vi` |
| **Arabic** | `ar` | **Thai** | `th` |
| **Indonesian** | `id` | **Filipino / Tagalog** | `tl` |
| **Malay** | `ms` | **Hebrew** | `he` |
| **Bengali** | `bn` | **Greek** | `el` |
| **Tamil** | `ta` | **Persian (Farsi)** | `fa` |
| **Telugu** | `te` | **Urdu** | `ur` |

*(Whisper also accepts the full lowercase English name, e.g. `-l japanese` or `-l chinese`).*

---

## 📦 Git Repository Notes

A `.gitignore` file is included that prevents committing large video files (`downloads/`, `*.mp4`, `*.mkv`, `*.srt`) and local virtual environments (`.venv/`) into your Git history.

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
