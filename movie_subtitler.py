#!/usr/bin/env python3
"""
Movie & Video Subtitler Pipeline
Downloads web videos (YouTube, Facebook, Bilibili, etc.) or processes local video files,
extracts or generates English subtitles via Whisper AI, and burns them into the video frames
(or embeds as soft subtitles).
"""

import os
import sys

# Suppress HuggingFace Hub Windows symlink warning
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Ensure UTF-8 output on Windows consoles to prevent UnicodeEncodeError with video titles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# --- Patch PyAV 14+ compatibility for faster-whisper ---
# Newer PyAV versions removed the 'metadata_errors' parameter from av.open()
try:
    import av
    _orig_av_open = av.open

    def _compat_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)

    av.open = _compat_av_open
except ImportError:
    pass
# --------------------------------------------------------

# --- Preload Linux CUDA runtime libraries (cublas, cudnn) if installed via pip ---
def _setup_linux_cuda_libraries():
    if sys.platform != "linux":
        return
    import ctypes
    from pathlib import Path
    for pkg in ["nvidia.cublas.lib", "nvidia.cudnn.lib"]:
        try:
            mod = __import__(pkg, fromlist=["__path__"])
            pkg_path = Path(mod.__path__[0])
            for so_path in sorted(pkg_path.glob("*.so*")):
                try:
                    ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        except Exception:
            pass

_setup_linux_cuda_libraries()
# ---------------------------------------------------------------------------------

import re
import html
import shutil
import argparse
import logging
import threading
import subprocess
from itertools import chain
from pathlib import Path
from typing import Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("movie_subtitler")


def find_ffmpeg() -> Optional[str]:
    """Find FFmpeg executable on the system or in known local locations."""
    # 1. System PATH
    ffmpeg_cmd = shutil.which("ffmpeg")
    if ffmpeg_cmd:
        return ffmpeg_cmd

    # 2. Common Linux binary locations
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path("/usr/bin/ffmpeg"),
        Path("/usr/local/bin/ffmpeg"),
        Path("/snap/bin/ffmpeg"),
        script_dir / "ffmpeg",
        Path.cwd() / "ffmpeg",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    # 3. Fallback to imageio_ffmpeg if installed
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    return None


def get_video_duration(video_path: Path, ffmpeg_exe: str) -> float:
    """Extract total video duration in seconds using ffmpeg."""
    try:
        cmd = [ffmpeg_exe, "-i", str(video_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", res.stderr)
        if match:
            h, m, s = match.groups()
            return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        pass
    return 0.0


def check_nvenc_available(ffmpeg_exe: str) -> bool:
    """Check if NVIDIA NVENC hardware encoder is supported by FFmpeg."""
    try:
        cmd = [ffmpeg_exe, "-encoders"]
        res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        return "h264_nvenc" in res.stdout
    except Exception:
        return False


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp string: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def clean_video_title(title: str) -> str:
    """
    Cleans raw video titles from social media / YouTube:
    - Removes view / reaction / like counts (e.g. '87M views · 1.9M reactions ｜ ')
    - Removes relative timestamps ('• 2 weeks ago')
    - Removes hashtags ('#afv #fail #funny')
    - Removes emojis and non-standard pictographs
    - Removes trailing ID numbers
    - Normalizes punctuation and replaces filesystem illegal characters
    """
    if not title:
        return "video"

    title = title.replace("｜", " - ").replace("？", "").replace("！", "!")
    title = title.replace("：", " - ").replace("／", "-")

    # Remove view / reaction / like counts (e.g. "87M views · 1.9M reactions ｜ ")
    title = re.sub(
        r"^\s*(?:[\d.,]+\s*[KMkmbB]?\s*(?:views?|reactions?|likes?|shares?|comments?)\s*[·•|\-–—]?\s*)+[|·•\-–—]?\s*",
        "",
        title,
        flags=re.IGNORECASE
    )

    # Remove relative timestamps if prefixed (e.g. "• 2 weeks ago")
    title = re.sub(
        r"^\s*(?:[·•|\-–—]?\s*[\d.,]+\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s*ago\s*[·•|\-–—]?\s*)+",
        "",
        title,
        flags=re.IGNORECASE
    )

    # Remove views/reactions anywhere if formatted like "· 87M views"
    title = re.sub(
        r"[·•|]\s*[\d.,]+\s*[KMkmbB]?\s*(?:views?|reactions?|likes?)\b",
        "",
        title,
        flags=re.IGNORECASE
    )

    # Remove hashtags (e.g. "#afv #fail #funny #shorts #viral")
    title = re.sub(r"#\w+", "", title)

    # Remove emojis and pictographs (avoid matching CJK ideographs \u4e00-\u9fff)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE
    )
    title = emoji_pattern.sub("", title)

    # Remove trailing ID patterns like "-1117085440226819" or "-9pWpK0jf0Es"
    title = re.sub(r"-\w{8,25}$", "", title)

    # Replace filesystem invalid characters: / \ : * ? " < > |
    title = re.sub(r'[/\\:*?"<>|]', " - ", title)

    # Clean up duplicated separators and spaces
    title = re.sub(r"[\s\-_–—]*\s*[-–—]\s*[\s\-_–—]*", " - ", title)
    title = re.sub(r"\s+", " ", title)
    title = title.strip(" -_")

    return title or "video"


def sanitize_subtitle_file(input_path: Path, output_srt: Path) -> bool:
    """
    Cleans raw subtitle files (VTT or SRT), handling:
    - Inline word timestamps (<00:00:29.080>, <00:00:29,080>)
    - WebVTT / HTML tags (<c>, </c>, <v Speaker>, <b>, <i>, etc.)
    - VTT cue positioning metadata (align:start position:0%, etc.)
    - Micro-duration glitch cues (< 150ms)
    - YouTube 2-line rolling teleprompter duplication (deduplicating repeated top line)
    - Missing closing timestamps (closing subtitle after speech stops rather than hanging indefinitely)
    """
    try:
        content = input_path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        raw_cues = []
        current_cue_ts = None
        current_lines = []
        current_inline_ts = []

        def parse_seconds(h_str, m_str, s_str, ms_str):
            return int(h_str) * 3600 + int(m_str) * 60 + int(s_str) + int(ms_str) / 1000.0

        for line in lines:
            line_s = line.strip()
            if not line_s:
                if current_cue_ts and current_lines:
                    raw_cues.append((current_cue_ts, current_lines, current_inline_ts))
                    current_cue_ts = None
                    current_lines = []
                    current_inline_ts = []
                continue

            # Match timestamp line (e.g. 00:00:05.960 --> 00:00:08.670 align:start position:0%)
            match = re.search(
                r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})",
                line_s
            )
            if match:
                if current_cue_ts and current_lines:
                    raw_cues.append((current_cue_ts, current_lines, current_inline_ts))
                    current_lines = []
                    current_inline_ts = []
                h1, m1, s1, ms1, h2, m2, s2, ms2 = match.groups()
                t1 = parse_seconds(h1, m1, s1, ms1)
                t2 = parse_seconds(h2, m2, s2, ms2)
                current_cue_ts = (t1, t2)
                continue

            if current_cue_ts is None or re.match(r"^\d+$", line_s):
                continue

            # Extract inline word timestamps if present before stripping
            for tm in re.finditer(r"<(\d{2}):(\d{2}):(\d{2})[.,](\d{3})>", line):
                current_inline_ts.append(parse_seconds(*tm.groups()))

            # Strip all inline tags (<00:00:00.000>, <c>, </c>, etc.)
            cleaned = re.sub(r"<[^>]+>", "", line)
            cleaned = html.unescape(cleaned).strip()
            cleaned = re.sub(r"\s+", " ", cleaned)
            if cleaned:
                if not current_lines or current_lines[-1] != cleaned:
                    current_lines.append(cleaned)

        if current_cue_ts and current_lines:
            raw_cues.append((current_cue_ts, current_lines, current_inline_ts))

        processed_cues = []
        prev_line = ""

        for (t1, t2), text_lines, inline_ts in raw_cues:
            if (t2 - t1) < 0.15:
                continue

            # YouTube rolling captions: line 0 is often the previous cue's line 1
            if len(text_lines) == 2 and prev_line and (text_lines[0] == prev_line or prev_line.endswith(text_lines[0])):
                actual_lines = [text_lines[1]]
            elif len(text_lines) == 2 and text_lines[0] == text_lines[1]:
                actual_lines = [text_lines[0]]
            else:
                actual_lines = text_lines

            prev_line = text_lines[-1]

            combined_text = " ".join(actual_lines)
            word_count = len(combined_text.split())

            # Smart closing timestamp calculation:
            if inline_ts:
                # Close 1.5s after the last spoken word
                end_sec = min(t2, max(inline_ts) + 1.5)
            elif (t2 - t1) > 7.0:
                # If duration is abnormally long (>7s, e.g. long silence), cap based on reading speed
                reading_duration = max(2.5, min(7.0, word_count * 0.45 + 1.5))
                end_sec = min(t2, t1 + reading_duration)
            else:
                end_sec = t2

            if end_sec < t1 + 1.2:
                end_sec = min(t2, t1 + 1.2)

            processed_cues.append({
                "start": t1,
                "end": end_sec,
                "text": "\n".join(actual_lines)
            })

        # Ensure no cue overlaps the start of the next cue
        for i in range(len(processed_cues) - 1):
            if processed_cues[i]["end"] > processed_cues[i+1]["start"]:
                processed_cues[i]["end"] = max(processed_cues[i]["start"] + 0.8, processed_cues[i+1]["start"] - 0.05)

        out_blocks = []
        counter = 1
        last_text = ""
        for c in processed_cues:
            text = c["text"]
            if text == last_text:
                continue
            last_text = text

            start_str = format_srt_timestamp(c["start"])
            end_str = format_srt_timestamp(c["end"])
            out_blocks.append(f"{counter}\n{start_str} --> {end_str}\n{text}")
            counter += 1

        output_srt.write_text("\n\n".join(out_blocks) + "\n", encoding="utf-8")
        return True
    except Exception as e:
        logger.warning(f"Could not sanitize subtitle file: {e}")
        return False


def download_or_find_video(
    source: str,
    output_dir: Path,
    cookies_from_browser: Optional[str] = None
) -> Tuple[Path, Optional[Path]]:
    """
    Downloads online video using yt-dlp, or uses an existing local video file.
    Returns: (video_path, existing_subtitles_srt_path)
    """
    local_path = Path(source)
    if local_path.is_file():
        logger.info(f"Using local video file: {local_path.name}")
        srt_sub = local_path.with_suffix(".srt")
        if srt_sub.exists():
            sanitize_subtitle_file(srt_sub, srt_sub)
            return local_path, srt_sub
        return local_path, None

    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp is not installed. Please run: pip install yt-dlp")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(output_dir / "%(title).120s.%(ext)s")

    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en.*"],
        "quiet": False,
        "no_warnings": False,
        "noplaylist": True,
    }

    # Enable available JavaScript runtimes for YouTube / video signature challenge solving
    js_runtimes = {}
    for rt in ["deno", "node", "nodejs", "quickjs"]:
        if shutil.which(rt):
            name = "node" if rt == "nodejs" else rt
            js_runtimes[name] = {}
    if js_runtimes:
        ydl_opts["js_runtimes"] = js_runtimes

    if cookies_from_browser:
        logger.info(f"Using cookies from browser: {cookies_from_browser}")
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser, )

    ffmpeg_exe = find_ffmpeg()
    if ffmpeg_exe:
        ydl_opts["ffmpeg_location"] = str(Path(ffmpeg_exe).parent)

    logger.info(f"Checking video source: {source}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=True)
        video_path = None
        if "requested_downloads" in info and info["requested_downloads"]:
            video_path = Path(info["requested_downloads"][0]["filepath"])
        
        if not video_path or not video_path.exists():
            base_cand = Path(ydl.prepare_filename(info))
            if base_cand.exists():
                video_path = base_cand
            elif base_cand.with_suffix(".mp4").exists():
                video_path = base_cand.with_suffix(".mp4")
            elif base_cand.with_suffix(".mkv").exists():
                video_path = base_cand.with_suffix(".mkv")

    if not video_path or not video_path.exists():
        video_id = info.get("id", "")
        if video_id:
            matches = list(output_dir.glob(f"*{video_id}*.mp4"))
            if matches:
                video_path = matches[0]

    if not video_path or not video_path.exists():
        mp4_candidates = [p for p in output_dir.glob("*.mp4") if not p.name.endswith(("_burned.mp4", "_subbed.mp4"))]
        if mp4_candidates:
            video_path = max(mp4_candidates, key=lambda p: p.stat().st_mtime)
        else:
            raise FileNotFoundError(f"Failed to find downloaded video in {output_dir}")

    # Clean the video filename (strip views, reactions, hashtags, emojis, ID numbers)
    raw_title = info.get("title") or video_path.stem
    clean_stem = clean_video_title(raw_title)
    if clean_stem and clean_stem != video_path.stem:
        target_video = output_dir / f"{clean_stem}{video_path.suffix}"
        if not target_video.exists():
            # Rename companion files (like .en.vtt, .en.srt, .srt, etc.)
            for comp in list(output_dir.glob(f"{video_path.stem}*")):
                if comp != video_path and comp.is_file():
                    ext_tail = comp.name[len(video_path.stem):]
                    comp_new = output_dir / f"{clean_stem}{ext_tail}"
                    if not comp_new.exists():
                        try:
                            comp.rename(comp_new)
                        except OSError:
                            pass
            try:
                video_path.rename(target_video)
                video_path = target_video
            except OSError:
                pass
        else:
            video_path = target_video

    logger.info(f"Video available: {video_path.name}")

    # Check for existing subtitles (SRT or VTT) and sanitize them
    srt_candidates = [
        video_path.with_suffix(".en.srt"),
        video_path.with_suffix(".en.vtt"),
    ] + [p for p in output_dir.glob(f"{video_path.stem}*.srt")] + [p for p in output_dir.glob(f"{video_path.stem}*.vtt")]

    for sub in srt_candidates:
        if sub.exists():
            clean_srt = video_path.with_suffix(".srt")
            if sanitize_subtitle_file(sub, clean_srt):
                logger.info(f"Loaded and sanitized existing subtitles: {clean_srt.name}")
                return video_path, clean_srt
            return video_path, sub

    logger.info("No pre-existing English subtitles found with video.")
    return video_path, None


def extract_vocals_filtered_audio(video_path: Path, ffmpeg_exe: str) -> Path:
    """
    Extracts 16kHz mono audio with voice-band filtering to boost speech clarity
    and attenuate heavy bass and background music.
    """
    filtered_audio = video_path.with_name(f"{video_path.stem}_audio_boosted.wav")
    logger.info("Applying speech isolation filter (attenuating background soundtrack music)...")
    # Bandpass filter for human vocal frequencies (120Hz to 3800Hz) + Dynamic Audio Normalizer
    af_filter = "highpass=f=120,lowpass=f=3800,dynaudnorm=f=150:g=15"
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-af", af_filter,
        "-ar", "16000",
        "-ac", "1",
        str(filtered_audio)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if res.returncode == 0 and filtered_audio.exists():
        logger.info(f"Created speech-isolated audio track: {filtered_audio.name}")
        return filtered_audio
    logger.warning("Vocal filter failed, using original video audio.")
    return video_path


def generate_subtitles_whisper(
    video_path: Path,
    srt_output_path: Path,
    ffmpeg_exe: Optional[str] = None,
    model_size: str = "base",
    device: str = "auto",
    compute_type: str = "auto",
    task: str = "auto",
    source_language: Optional[str] = "auto",
    use_vad: bool = False,
    boost_vocals: bool = False,
    no_speech_threshold: float = 0.4
) -> Path:
    """
    Transcribes audio from the video and generates timestamped English subtitles (.srt)
    using faster-whisper.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.error("faster-whisper is not installed. Please run: pip install faster-whisper")
        sys.exit(1)

    if device == "auto":
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = shutil.which("nvidia-smi") is not None
        device = "cuda" if has_cuda else "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    logger.info(
        f"Loading Whisper model '{model_size}' (device={device}, compute_type={compute_type})..."
    )
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        if device == "cuda":
            logger.warning(f"CUDA initialization failed ({e}). Falling back to CPU...")
            device = "cpu"
            compute_type = "int8"
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        else:
            raise e

    audio_source = video_path
    if boost_vocals and ffmpeg_exe:
        audio_source = extract_vocals_filtered_audio(video_path, ffmpeg_exe)

    lang_param = None if (not source_language or source_language == "auto") else source_language
    effective_task = task if task != "auto" else "transcribe"

    transcribe_kwargs = {
        "task": effective_task,
        "language": lang_param,
        "beam_size": 5,
        "no_speech_threshold": no_speech_threshold,
        "vad_filter": use_vad,
    }
    if use_vad:
        transcribe_kwargs["vad_parameters"] = dict(threshold=0.2, min_silence_duration_ms=500)

    logger.info(
        f"Transcribing audio with Whisper (VAD={'enabled' if use_vad else 'disabled (detects speech under music)'}, "
        f"no_speech_threshold={no_speech_threshold})..."
    )

    def _safe_transcribe(m, kwargs):
        try:
            segs, inf = m.transcribe(str(audio_source), **kwargs)
            it = iter(segs)
            first_item = next(it, None)
            combined = chain([first_item], it) if first_item is not None else it
            return combined, inf
        except Exception as e:
            if device == "cuda" and ("libcublas" in str(e) or "CUDA" in str(e) or "cuda" in str(e)):
                logger.warning(
                    f"CUDA execution failed ({e}).\n"
                    "Tip: Install Linux CUDA libraries with: uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12\n"
                    "Falling back to CPU transcription..."
                )
                cpu_model = WhisperModel(model_size, device="cpu", compute_type="int8")
                segs, inf = cpu_model.transcribe(str(audio_source), **kwargs)
                return segs, inf
            raise e

    segments, info = _safe_transcribe(model, transcribe_kwargs)

    logger.info(f"Detected speech language: '{info.language}' (confidence: {info.language_probability:.2f})")

    # If target is English, source is non-English, and task is auto, translate into English
    if task == "auto" and info.language != "en":
        logger.info(f"Source speech is '{info.language}'. Translating into English subtitles...")
        transcribe_kwargs["task"] = "translate"
        transcribe_kwargs["language"] = info.language
        segments, info = _safe_transcribe(model, transcribe_kwargs)

    duration = getattr(info, "duration", 0.0)
    if duration > 0:
        logger.info(f"Total audio duration: {duration:.1f}s (~{duration/60:.1f} min)")

    srt_lines = []
    counter = 1
    last_end = 0.0

    pbar = tqdm(total=round(duration), unit="s", desc="Transcribing (Whisper)", dynamic_ncols=True) if (tqdm and duration > 0) else None

    try:
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            start_str = format_srt_timestamp(seg.start)
            end_str = format_srt_timestamp(seg.end)
            srt_lines.append(f"{counter}\n{start_str} --> {end_str}\n{text}\n")
            counter += 1

            if pbar and seg.end > last_end:
                pbar.update(round(seg.end - last_end))
                last_end = seg.end
    finally:
        if pbar:
            if duration > last_end:
                pbar.update(round(duration - last_end))
            pbar.close()

    # Clean up temporary vocal track if created
    if audio_source != video_path and audio_source.exists():
        try:
            audio_source.unlink()
        except OSError:
            pass

    srt_output_path.write_text("\n".join(srt_lines), encoding="utf-8")
    logger.info(f"Subtitles written successfully to: {srt_output_path.name} ({counter - 1} segments)")
    return srt_output_path


def burn_subtitles_ffmpeg(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    ffmpeg_exe: str,
    font_name: str = "Arial",
    font_size: int = 18,
    use_nvenc: bool = False
) -> Path:
    """
    Burns (hardcodes) subtitles into video frames using FFmpeg with live progress tracking.
    """
    work_dir = srt_path.parent
    rel_video = video_path.resolve()
    rel_output = output_path.resolve()

    # Use a clean, safe temporary subtitle file without spaces, apostrophes, or special characters
    temp_srt = work_dir / f"_burn_{os.getpid()}.srt"
    shutil.copy2(srt_path, temp_srt)

    duration = get_video_duration(rel_video, ffmpeg_exe)

    style = (
        f"FontName={font_name},"
        f"FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"      # White text
        f"OutlineColour=&H00000000,"      # Black outline
        f"BackColour=&H80000000,"         # Semi-transparent shadow
        f"BorderStyle=1,"                 # Outline with shadow
        f"Outline=2.0,"                   # Outline thickness
        f"Shadow=1.0,"                    # Shadow depth
        f"Alignment=2,"                   # Centered bottom
        f"MarginV=25"                     # Distance from bottom margin
    )

    subtitles_filter = f"subtitles='{temp_srt.name}':force_style='{style}'"

    # Select encoder
    if use_nvenc:
        logger.info("Using NVIDIA NVENC hardware acceleration for fast subtitle burning...")
        vcodec = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22"]
    else:
        logger.info("Burning subtitles with CPU (libx264)...")
        vcodec = ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]

    cmd = [
        ffmpeg_exe,
        "-y",
        "-nostats",
        "-progress", "pipe:1",
        "-i", str(rel_video),
        "-vf", subtitles_filter,
        *vcodec,
        "-c:a", "copy",
        str(rel_output)
    ]

    pbar = tqdm(total=round(duration), unit="s", desc="Burning Subtitles (FFmpeg)", dynamic_ncols=True) if (tqdm and duration > 0) else None
    last_sec = 0.0

    process = subprocess.Popen(
        cmd,
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    stderr_buffer = []

    def _drain_stderr(stream):
        for s_line in stream:
            stderr_buffer.append(s_line)
            if len(stderr_buffer) > 100:
                stderr_buffer.pop(0)

    stderr_thread = threading.Thread(target=_drain_stderr, args=(process.stderr,), daemon=True)
    stderr_thread.start()

    try:
        for line in process.stdout:
            line = line.strip()
            if line.startswith("out_time_us="):
                try:
                    us_val = int(line.split("=")[1])
                    cur_sec = us_val / 1_000_000.0
                    if pbar and cur_sec > last_sec:
                        pbar.update(round(cur_sec - last_sec))
                        last_sec = cur_sec
                except ValueError:
                    pass
            elif line.startswith("speed="):
                spd = line.split("=")[1].strip()
                if pbar and spd:
                    pbar.set_postfix({"speed": spd})
    finally:
        if pbar:
            if duration > last_sec:
                pbar.update(round(duration - last_sec))
            pbar.close()
        temp_srt.unlink(missing_ok=True)

    process.wait()
    stderr_thread.join(timeout=1.0)
    if process.returncode != 0:
        err_msg = "".join(stderr_buffer)
        logger.error(f"FFmpeg burning failed:\n{err_msg}")
        raise RuntimeError("FFmpeg subtitle burn-in failed.")

    logger.info(f"Subtitles burned successfully! Final video: {output_path.name}")
    return output_path


def mux_soft_subtitles_ffmpeg(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    ffmpeg_exe: str
) -> Path:
    """
    Embeds (soft-subs) subtitles as an internal toggleable track in an MP4 container.
    """
    logger.info(f"Embedding soft subtitles into MP4 container...")
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(video_path),
        "-i", str(srt_path),
        "-c:v", "copy",
        "-c:a", "copy",
        "-c:s", "mov_text",
        "-metadata:s:s:0", "language=eng",
        "-metadata:s:s:0", "title=English",
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"FFmpeg soft-sub muxing failed:\n{result.stderr}")
        raise RuntimeError("FFmpeg soft subtitle embedding failed.")

    logger.info(f"Soft subtitles embedded! Final video: {output_path.name}")
    return output_path


def process_video(
    source: str,
    output_dir: str = "./downloads",
    burn: bool = True,
    use_nvenc: bool = False,
    model_size: str = "base",
    device: str = "auto",
    compute_type: str = "auto",
    task: str = "auto",
    source_language: Optional[str] = "auto",
    use_vad: bool = False,
    boost_vocals: bool = False,
    font_name: str = "Arial",
    font_size: int = 18,
    cookies_from_browser: Optional[str] = None,
    keep_temp: bool = True,
    force_subtitles: bool = False
) -> Path:
    """Main pipeline to download, transcribe, and apply subtitles."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = find_ffmpeg()
    if not ffmpeg_exe:
        raise RuntimeError(
            "FFmpeg not found! Please install it with: sudo apt update && sudo apt install -y ffmpeg"
        )

    # Auto-detect NVENC if user has NVIDIA GPU
    if use_nvenc or (device in ("auto", "cuda") and check_nvenc_available(ffmpeg_exe)):
        use_nvenc = True

    # 1. Download or find video
    video_path, existing_srt = download_or_find_video(
        source=source,
        output_dir=out_dir,
        cookies_from_browser=cookies_from_browser
    )

    # 2. Obtain subtitles
    srt_path = existing_srt if not force_subtitles else None
    if not srt_path:
        srt_path = video_path.with_suffix(".srt")
        generate_subtitles_whisper(
            video_path=video_path,
            srt_output_path=srt_path,
            ffmpeg_exe=ffmpeg_exe,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            task=task,
            source_language=source_language,
            use_vad=use_vad,
            boost_vocals=boost_vocals
        )

    # 3. Apply subtitles
    suffix = "_burned.mp4" if burn else "_subbed.mp4"
    final_output_path = out_dir / f"{video_path.stem}{suffix}"

    if burn:
        burn_subtitles_ffmpeg(
            video_path=video_path,
            srt_path=srt_path,
            output_path=final_output_path,
            ffmpeg_exe=ffmpeg_exe,
            font_name=font_name,
            font_size=font_size,
            use_nvenc=use_nvenc
        )
    else:
        mux_soft_subtitles_ffmpeg(
            video_path=video_path,
            srt_path=srt_path,
            output_path=final_output_path,
            ffmpeg_exe=ffmpeg_exe
        )

    if not keep_temp:
        if video_path.exists() and video_path != final_output_path:
            video_path.unlink()

    return final_output_path


def main():
    parser = argparse.ArgumentParser(
        description="Download a video/movie or load a local file, generate English subtitles with Whisper AI, and burn them into the video."
    )
    parser.add_argument("url", nargs="?", help="Video URL or path to an already downloaded video file")
    parser.add_argument("-u", "--url-flag", dest="url_opt", help="Alternative way to provide the video URL/file")
    parser.add_argument("-o", "--output-dir", default="./downloads", help="Directory where downloaded files will be saved (default: ./downloads)")
    parser.add_argument("--soft", action="store_true", help="Embed as soft (toggleable) subtitles instead of burning them into video frames")
    parser.add_argument("--nvenc", action="store_true", help="Use NVIDIA GPU (NVENC) for ultra-fast subtitle burning (3-5 mins instead of 40 mins)")
    parser.add_argument("-m", "--model", default="base", choices=["tiny", "base", "small", "medium", "large-v3"], help="Whisper model size (default: base. Use 'small' or 'medium' for much better speech recognition over music)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Processing device for Whisper (default: auto)")
    parser.add_argument("--compute-type", default="auto", help="Quantization / precision (e.g. float16, int8, auto)")
    parser.add_argument("--task", default="auto", choices=["auto", "transcribe", "translate"], help="'auto' (default: transcribes English, translates foreign speech to English), 'transcribe', or 'translate'")
    parser.add_argument("-l", "--language", default="auto", help="Source language code ('auto', 'en', 'ja', 'es', etc.)")
    parser.add_argument("--vad", action="store_true", help="Enable aggressive Voice Activity Detection (disabled by default so dialogue under background music isn't dropped)")
    parser.add_argument("--boost-vocals", action="store_true", help="Pre-filter audio to attenuate background music/bass before sending to Whisper")
    parser.add_argument("--force-whisper", action="store_true", help="Re-generate subtitles with Whisper even if a subtitle file already exists")
    parser.add_argument("--font-name", default="Arial", help="Subtitle font name (default: Arial)")
    parser.add_argument("--font-size", type=int, default=18, help="Subtitle font size (default: 18)")
    parser.add_argument("--cookies-from-browser", default=None, choices=["chrome", "firefox", "edge", "brave", "opera", "safari"], help="Extract cookies from local browser for private/age-restricted videos")
    parser.add_argument("--no-keep-temp", action="store_true", help="Delete the raw un-subtitled video after burning")

    args = parser.parse_args()
    video_source = args.url or args.url_opt

    if not video_source:
        downloads_dir = Path(args.output_dir)
        candidates = [p for p in downloads_dir.glob("*.mp4") if not p.name.endswith(("_burned.mp4", "_subbed.mp4"))] if downloads_dir.exists() else []
        if candidates:
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
            print(f"\nFound existing downloaded video: {latest.name}")
            use_existing = input("Do you want to process this video? [Y/n]: ").strip().lower()
            if use_existing in ("", "y", "yes"):
                video_source = str(latest)

    if not video_source:
        video_source = input("Enter video URL or local file path: ").strip()

    if not video_source:
        parser.print_help()
        sys.exit(1)

    print("=" * 60)
    print(" Movie & Video Subtitler")
    print("=" * 60)
    print(f" Source:       {video_source}")
    print(f" Output Dir:   {args.output_dir}")
    print(f" Mode:         {'Soft subtitles' if args.soft else 'Burn-in (hardcoded)'}")
    print(f" AI Model:     Whisper ({args.model})")
    print(f" Speech VAD:   {'Enabled' if args.vad else 'Disabled (detects speech under music)'}")
    if args.boost_vocals:
        print(f" Vocal Filter: Enabled (attenuating background soundtrack)")
    print("=" * 60)

    try:
        final_video = process_video(
            source=video_source,
            output_dir=args.output_dir,
            burn=not args.soft,
            use_nvenc=args.nvenc,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            task=args.task,
            source_language=args.language,
            use_vad=args.vad,
            boost_vocals=args.boost_vocals,
            font_name=args.font_name,
            font_size=args.font_size,
            cookies_from_browser=args.cookies_from_browser,
            keep_temp=not args.no_keep_temp,
            force_subtitles=args.force_whisper
        )
        print("\n" + "=" * 60)
        print(f"SUCCESS! Subtitled video ready at:")
        print(f" -> {final_video}")
        print("=" * 60)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
