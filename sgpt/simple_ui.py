import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import os
import sys
import subprocess
from pathlib import Path
import importlib.util

# Function to check if a package is installed
def is_package_installed(package_name):
    """Check if a package is installed using importlib."""
    return importlib.util.find_spec(package_name) is not None

# List of required packages
REQUIRED_PACKAGES = [
    'torch',
    'transformers',
    'accelerate',
    'diffusers',
    'TTS',
    'moviepy',
    'numpy',
    'Pillow',
    'scipy',
    'sentencepiece'
]

# Check and install required packages
def check_and_install_packages():
    """Check if required packages are installed and offer to install them."""
    missing_packages = [pkg for pkg in REQUIRED_PACKAGES if not is_package_installed(pkg)]
    
    if missing_packages:
        # Create a simple root window
        temp_root = tk.Tk()
        temp_root.withdraw()  # Hide the root window
            
        message = (
            f"The following required packages are missing:\n"
            f"{', '.join(missing_packages)}\n\n"
            f"Would you like to install them now?"
        )
        
        response = messagebox.askyesno("Missing Dependencies", message)
        
        if response:  # User clicked Yes
            # Create a progress window
            progress_win = tk.Toplevel(temp_root)
            progress_win.title("Installing Dependencies")
            progress_win.geometry("500x400")
            
            # Progress text widget
            progress_text = scrolledtext.ScrolledText(progress_win, wrap=tk.WORD, width=60, height=20)
            progress_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            
            def update_progress(message):
                progress_text.insert(tk.END, message + "\n")
                progress_text.see(tk.END)
                progress_win.update()
            
            # Install packages
            update_progress("Installing missing packages. This may take a while...")
            
            try:
                # Install from requirements.txt if it exists
                if os.path.exists("requirements.txt"):
                    update_progress("Found requirements.txt file. Installing from it...")
                    process = subprocess.Popen(
                        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                else:
                    # Install missing packages directly
                    update_progress(f"Installing packages: {', '.join(missing_packages)}")
                    process = subprocess.Popen(
                        [sys.executable, "-m", "pip", "install"] + missing_packages,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                
                # Stream output
                for line in process.stdout:
                    update_progress(line.strip())
                
                process.wait()
                
                if process.returncode == 0:
                    update_progress("\nInstallation complete! You may need to restart the application.")
                    messagebox.showinfo("Success", "Required packages installed successfully! Please restart the application.")
                    temp_root.destroy()
                    return False  # Don't continue execution, need restart
                else:
                    update_progress("\nInstallation failed. Please try manual installation:")
                    update_progress("pip install -r requirements.txt")
                    messagebox.showerror("Error", "Failed to install some packages. Please try manual installation.")
            
            except Exception as e:
                update_progress(f"Error during installation: {str(e)}")
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
            
            finally:
                # Add a close button
                ttk.Button(progress_win, text="Close", command=progress_win.destroy).pack(pady=10)
        
        temp_root.destroy()
            
        if not response:  # User clicked No
            return False
    
    return True


class ShortsGeneratorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube Shorts Generator")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Set output directory
        self.output_dir = "./output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create UI elements
        self.create_widgets()
        
        # Try to import ShortsGenerator
        try:
            from shorts_generator import ShortsGenerator
            self.generator = ShortsGenerator(output_dir=self.output_dir)
            self.imports_successful = True
            self.status_var.set("Ready")
            self.generate_button.config(state=tk.NORMAL)
        except ImportError as e:
            self.imports_successful = False
            self.status_var.set("Dependencies missing")
            self.generate_button.config(state=tk.DISABLED)
            self.update_log(f"ERROR: Failed to import required modules: {str(e)}")
            self.update_log("Please restart the application after installing dependencies.")
        
        # Status variables
        self.is_generating = False
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="AI YouTube Shorts Generator", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Description
        desc_text = "Generate YouTube Shorts with AI: script, voiceover, image, and video."
        desc_label = ttk.Label(main_frame, text=desc_text, wraplength=700)
        desc_label.pack(pady=5)
        
        # Input section frame
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=False, pady=10)
        
        # Prompt label and text area
        prompt_label = ttk.Label(
            input_frame, 
            text="Enter a prompt for generating the script:"
        )
        prompt_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.prompt_text = scrolledtext.ScrolledText(
            input_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=5
        )
        self.prompt_text.insert(
            tk.INSERT, 
            "Write a viral YouTube short script about an unbelievable fact in under 50 words."
        )
        self.prompt_text.pack(fill=tk.BOTH, expand=True)
        
        # Options frame
        options_frame = ttk.Frame(input_frame)
        options_frame.pack(fill=tk.X, pady=10)
        
        # Subtitles checkbox
        self.add_subtitles_var = tk.BooleanVar(value=True)
        subtitles_check = ttk.Checkbutton(
            options_frame, 
            text="Add burned-in subtitles", 
            variable=self.add_subtitles_var
        )
        subtitles_check.pack(side=tk.LEFT, padx=5)
        
        # Generate button
        self.generate_button = ttk.Button(
            input_frame, 
            text="Generate Short", 
            command=self.generate_short,
            state=tk.DISABLED  # Start disabled until we confirm imports work
        )
        self.generate_button.pack(pady=10)
        
        # Output section frame
        self.output_frame = ttk.LabelFrame(main_frame, text="Output", padding=10)
        self.output_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Log area
        log_label = ttk.Label(self.output_frame, text="Generation Log:")
        log_label.pack(anchor=tk.W)
        
        self.log_text = scrolledtext.ScrolledText(
            self.output_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=10, 
            state=tk.DISABLED
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Result actions frame
        self.result_frame = ttk.Frame(self.output_frame)
        self.result_frame.pack(fill=tk.X, pady=5)
        
        self.play_button = ttk.Button(
            self.result_frame, 
            text="Play Video", 
            command=self.play_video,
            state=tk.DISABLED
        )
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(
            self.result_frame, 
            text="Save Video", 
            command=self.save_video,
            state=tk.DISABLED
        )
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Initializing...")
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_log(self, message):
        """Update the log area with a message."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self.root.update_idletasks()
    
    def generate_short(self):
        """Start the generation process in a separate thread."""
        if self.is_generating:
            return
        
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            self.update_log("Error: Prompt cannot be empty!")
            return
        
        self.is_generating = True
        self.generate_button.configure(state=tk.DISABLED)
        self.status_var.set("Generating...")
        self.update_log("Starting generation process...")
        
        # Redirect print to log
        self.original_print = print
        def custom_print(*args, **kwargs):
            message = " ".join(str(arg) for arg in args)
            self.update_log(message)
            self.original_print(*args, **kwargs)
        import builtins
        builtins.print = custom_print
        
        # Start generation in a thread
        threading.Thread(target=self._generate_thread, args=(prompt,)).start()
    
    def _generate_thread(self, prompt):
        """Thread function to handle the generation process."""
        try:
            add_subtitles = self.add_subtitles_var.get()
            
            self.update_log(f"Generating script from prompt: {prompt}")
            self.result = self.generator.generate_short(
                prompt, 
                add_subtitles=add_subtitles
            )
            
            self.root.after(0, self._generation_complete)
            
        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
            self.update_log(error_msg)
            self.root.after(0, self._generation_failed, error_msg)
        finally:
            # Restore print
            import builtins
            builtins.print = self.original_print
    
    def _generation_complete(self):
        """Called when generation is complete."""
        self.is_generating = False
        self.generate_button.configure(state=tk.NORMAL)
        self.play_button.configure(state=tk.NORMAL)
        self.save_button.configure(state=tk.NORMAL)
        
        self.status_var.set("Generation complete!")
        self.update_log("\n--- GENERATION COMPLETE ---")
        self.update_log(f"Script: {self.result['script']}")
        self.update_log(f"Video saved to: {self.result['video_path']}")
        self.update_log(f"Total generation time: {self.result['duration']:.2f} seconds")
    
    def _generation_failed(self, error_msg):
        """Called when generation fails."""
        self.is_generating = False
        self.generate_button.configure(state=tk.NORMAL)
        self.status_var.set("Generation failed!")
    
    def play_video(self):
        """Play the generated video."""
        if hasattr(self, 'result') and os.path.exists(self.result['video_path']):
            video_path = self.result['video_path']
            self.update_log(f"Opening video: {video_path}")
            
            # Use the default video player
            import subprocess
            import platform
            
            system = platform.system()
            if system == "Windows":
                os.startfile(video_path)
            elif system == "Darwin":  # macOS
                subprocess.call(["open", video_path])
            else:  # Linux
                subprocess.call(["xdg-open", video_path])
        else:
            self.update_log("No video available to play")
    
    def save_video(self):
        """Save the generated video to a user-selected location."""
        if hasattr(self, 'result') and os.path.exists(self.result['video_path']):
            save_path = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
                title="Save YouTube Short"
            )
            
            if save_path:
                import shutil
                shutil.copy2(self.result['video_path'], save_path)
                self.update_log(f"Video saved to: {save_path}")
        else:
            self.update_log("No video available to save")


def main():
    """Main entry point for the application."""
    # First check if required packages are installed
    if not check_and_install_packages():
        # If check returns False, exit (either user declined or install failed)
        sys.exit(1)
        
    # Now it's safe to create the main application
    root = tk.Tk()
    app = ShortsGeneratorUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
