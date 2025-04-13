# AI YouTube Shorts Generator

An automated tool that creates engaging YouTube Shorts using AI models.

## Features

- Script Generation: Uses Mistral 7B to create engaging short scripts
- Text-to-Speech: Converts scripts to natural-sounding voiceovers using Coqui TTS
- Image Generation: Creates relevant visuals with Stable Diffusion
- Video Compilation: Merges audio and visuals into vertical format videos
- Optional Subtitles: Adds burned-in captions using Whisper

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- TTS (Coqui)
- Diffusers
- MoviePy
- (Optional) Whisper

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
python shorts_generator.py --prompt "Write a viral YouTube short script about an unbelievable fact"
```

### Simple UI (Tkinter)

For a lightweight graphical interface:

```bash
python simple_ui.py
```

### Web Interface (Streamlit)

For a more feature-rich web interface:

```bash
streamlit run app.py
```

## Output

The final video will be saved as `shorts_output.mp4` in the output directory.
