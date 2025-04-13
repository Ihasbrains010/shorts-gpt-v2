import os
import argparse
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
import tempfile

# Text generation with Mistral
from transformers import AutoModelForCausalLM, AutoTokenizer

# Text-to-Speech
import TTS
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

# Image generation
from diffusers import StableDiffusionPipeline

# Video creation
from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip
import moviepy.config as moviepy_config

# Optional: Whisper for transcription
import whisper

class ShortsGenerator:
    def __init__(self, output_dir="./output"):
        """Initialize the YouTube Shorts generator with all necessary models."""
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Placeholder attributes for models (loaded on-demand to save memory)
        self.text_model = None
        self.text_tokenizer = None
        self.tts_model = None
        self.image_model = None
        self.whisper_model = None
    
    def load_text_model(self):
        """Load the Mistral 7B model for text generation."""
        print("Loading Mistral 7B model...")
        model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.text_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        print("Text model loaded.")
    
    def load_tts_model(self):
        """Load the TTS model."""
        print("Loading TTS model...")
        # Using Coqui TTS with a pre-trained model
        model_name = "tts_models/en/ljspeech/tacotron2-DDC"
        manager = ModelManager()
        model_path, config_path, model_item = manager.download_model(model_name)
        vocoder_name = model_item["default_vocoder"]
        vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)
        
        self.tts_model = Synthesizer(
            model_path, config_path, vocoder_path, vocoder_config_path, use_cuda=self.device=="cuda"
        )
        print("TTS model loaded.")
    
    def load_image_model(self):
        """Load the Stable Diffusion model for image generation."""
        print("Loading Stable Diffusion model...")
        model_id = "CompVis/stable-diffusion-v1-4"
        self.image_model = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        print("Image model loaded.")
    
    def load_whisper_model(self):
        """Load the Whisper model for transcription."""
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        print("Whisper model loaded.")
    
    def generate_script(self, prompt):
        """Generate a script using the Mistral model."""
        if self.text_model is None:
            self.load_text_model()
        
        # Format the prompt according to Mistral's instruction format
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        print("Generating script...")
        inputs = self.text_tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        outputs = self.text_model.generate(
            inputs["input_ids"],
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        response = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        script = response.split("[/INST]")[-1].strip()
        
        # Count words and ensure it's under 50 words
        words = script.split()
        if len(words) > 50:
            script = " ".join(words[:50])
        
        print(f"Generated script ({len(script.split())} words): {script}")
        return script
    
    def generate_speech(self, text):
        """Convert text to speech using TTS."""
        if self.tts_model is None:
            self.load_tts_model()
        
        print("Generating speech...")
        wav_path = os.path.join(self.output_dir, "voiceover.wav")
        
        # Generate speech
        wav = self.tts_model.tts(text)
        self.tts_model.save_wav(wav, wav_path)
        
        print(f"Speech saved to {wav_path}")
        return wav_path
    
    def generate_image(self, text):
        """Generate an image based on the text using Stable Diffusion."""
        if self.image_model is None:
            self.load_image_model()
        
        print("Generating image...")
        # Create a more descriptive prompt for better image generation
        prompt = f"Cinematic, high quality image depicting: {text}"
        
        # Generate the image
        image = self.image_model(prompt, height=1024, width=1024).images[0]
        
        image_path = os.path.join(self.output_dir, "image.png")
        image.save(image_path)
        
        print(f"Image saved to {image_path}")
        return image_path
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper for subtitles."""
        if self.whisper_model is None:
            self.load_whisper_model()
        
        print("Transcribing audio...")
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]
    
    def create_subtitles(self, text, video_width, video_height):
        """Create subtitle clips from text."""
        # Split long text into chunks for better readability
        words = text.split()
        chunks = []
        chunk_size = 5  # words per line
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i+chunk_size]))
        
        font_size = int(video_height * 0.05)  # 5% of video height
        
        # Create a TextClip for each chunk
        subtitle_clips = []
        for i, chunk in enumerate(chunks):
            txt_clip = TextClip(
                chunk, 
                fontsize=font_size, 
                color='white',
                font='Arial-Bold',
                stroke_color='black',
                stroke_width=2,
                method='caption',
                size=(video_width * 0.9, None)  # 90% of video width
            )
            
            # Position at bottom, staggered based on chunk index
            txt_clip = txt_clip.set_position(('center', video_height * 0.7 + i * font_size * 1.5))
            
            # Set duration according to chunk length (approx. 0.3s per word)
            duration = len(chunk.split()) * 0.3
            txt_clip = txt_clip.set_duration(duration)
            
            # Set start time based on position in text
            start_time = i * duration
            subtitle_clips.append(txt_clip.set_start(start_time))
        
        return subtitle_clips
    
    def create_video(self, image_path, audio_path, add_subtitles=True):
        """Create a YouTube short video from image and audio."""
        print("Creating video...")
        
        # Get audio duration
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        
        # Create video with vertical format (9:16 ratio)
        width = 1080
        height = 1920
        
        # Load and resize image to fill the frame (center crop)
        image = Image.open(image_path)
        img_ratio = image.width / image.height
        video_ratio = width / height
        
        if img_ratio > video_ratio:  # Image is wider than video
            new_height = image.height
            new_width = int(new_height * video_ratio)
            left = (image.width - new_width) // 2
            image = image.crop((left, 0, left + new_width, new_height))
        else:  # Image is taller than video
            new_width = image.width
            new_height = int(new_width / video_ratio)
            top = (image.height - new_height) // 2
            image = image.crop((0, top, new_width, top + new_height))
        
        # Resize to target dimensions
        image = image.resize((width, height), Image.LANCZOS)
        
        # Save the processed image
        temp_img_path = os.path.join(self.output_dir, "processed_image.png")
        image.save(temp_img_path)
        
        # Create video clip
        image_clip = ImageClip(temp_img_path).set_duration(duration)
        
        # Add subtitles if requested
        if add_subtitles:
            transcript = self.transcribe_audio(audio_path)
            subtitle_clips = self.create_subtitles(transcript, width, height)
            video = CompositeVideoClip([image_clip] + subtitle_clips)
        else:
            video = image_clip
        
        # Add audio
        video = video.set_audio(audio_clip)
        
        # Save video
        output_path = os.path.join(self.output_dir, "shorts_output.mp4")
        video.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(self.output_dir, "temp_audio.m4a"),
            remove_temp=True
        )
        
        print(f"Video saved to {output_path}")
        return output_path
    
    def generate_short(self, prompt, add_subtitles=True):
        """Generate a complete YouTube short from a prompt."""
        start_time = time.time()
        
        # 1. Generate script
        script = self.generate_script(prompt)
        
        # 2. Generate speech
        audio_path = self.generate_speech(script)
        
        # 3. Generate image
        image_path = self.generate_image(script)
        
        # 4. Create video
        video_path = self.create_video(image_path, audio_path, add_subtitles)
        
        elapsed_time = time.time() - start_time
        print(f"Short created in {elapsed_time:.2f} seconds")
        
        return {
            "script": script,
            "audio_path": audio_path,
            "image_path": image_path,
            "video_path": video_path,
            "duration": elapsed_time
        }


def main():
    parser = argparse.ArgumentParser(description="Generate AI YouTube Shorts")
    parser.add_argument("--prompt", type=str, default="Write a viral YouTube short script about an unbelievable fact in under 50 words.",
                      help="Prompt for generating the script")
    parser.add_argument("--output", type=str, default="./output",
                      help="Output directory for generated files")
    parser.add_argument("--no-subtitles", action="store_true",
                      help="Disable subtitles in the output video")
    
    args = parser.parse_args()
    
    generator = ShortsGenerator(output_dir=args.output)
    result = generator.generate_short(args.prompt, add_subtitles=not args.no_subtitles)
    
    print("\nGeneration Complete!")
    print(f"Script: {result['script']}")
    print(f"Video saved to: {result['video_path']}")
    print(f"Total duration: {result['duration']:.2f} seconds")


if __name__ == "__main__":
    main()
