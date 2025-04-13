import streamlit as st
import os
from shorts_generator import ShortsGenerator
from pathlib import Path

st.set_page_config(page_title="AI YouTube Shorts Generator", layout="wide")

# Create output directory
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Initialize session state
if "generator" not in st.session_state:
    st.session_state.generator = ShortsGenerator(output_dir=output_dir)
if "generated" not in st.session_state:
    st.session_state.generated = False
if "result" not in st.session_state:
    st.session_state.result = None

def generate_short():
    with st.spinner("Generating YouTube Short..."):
        st.session_state.result = st.session_state.generator.generate_short(
            st.session_state.prompt,
            add_subtitles=st.session_state.add_subtitles
        )
        st.session_state.generated = True

st.title("AI YouTube Shorts Generator")

st.markdown("""
This app creates AI-generated YouTube Shorts by:
1. Generating a script with Mistral 7B
2. Converting the script to speech with TTS
3. Creating a relevant image with Stable Diffusion
4. Compiling everything into a vertical video
""")

# Input form
with st.form("input_form"):
    st.session_state.prompt = st.text_area(
        "Enter a prompt for generating the script:",
        value="Write a viral YouTube short script about an unbelievable fact in under 50 words.",
        height=100
    )
    
    st.session_state.add_subtitles = st.checkbox("Add burned-in subtitles", value=True)
    
    submit_button = st.form_submit_button("Generate Short")
    if submit_button:
        generate_short()

# Display results
if st.session_state.generated and st.session_state.result:
    st.success("YouTube Short generated successfully!")
    
    st.subheader("Generated Script")
    st.write(st.session_state.result["script"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Generated Image")
        st.image(st.session_state.result["image_path"])
    
    with col2:
        st.subheader("Generated Audio")
        st.audio(st.session_state.result["audio_path"])
    
    st.subheader("Final Video")
    video_file = open(st.session_state.result["video_path"], 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    
    st.download_button(
        label="Download Video",
        data=video_bytes,
        file_name="shorts_output.mp4",
        mime="video/mp4"
    )
    
    st.info(f"Total generation time: {st.session_state.result['duration']:.2f} seconds")

st.sidebar.title("About")
st.sidebar.info(
    "This application uses AI models to create YouTube Shorts from a simple prompt. "
    "It leverages Mistral 7B for script generation, TTS for voiceover, "
    "Stable Diffusion for image creation, and MoviePy for video assembly."
)
