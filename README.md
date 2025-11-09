# 🎧 Whisper Live Transcriber

A multi-threaded Python application for real-time voice-to-text transcription using OpenAI's Whisper model and a Tkinter graphical interface. This project is specifically hardened to handle common Windows audio driver conflicts and low-gain microphone environments.

## ✨ Features

* **Real-Time Transcription:** Utilizes two separate threads for continuous audio recording and background transcription processing.
* **Dynamic Audio Resampling:** Captures microphone audio at the system's native **48000 Hz** and automatically **resamples and converts it to 16000 Hz (mono)** for efficient processing by the Whisper model.
* **Low-Gain Compatibility:** Includes a low-sensitivity Voice Activity Detection (VAD) threshold to ensure transcription works even when microphone gain is suppressed by "Smart Audio" drivers.
* **Performance Metrics:** Displays **real-time latency** (time from audio chunk capture to transcription completion) in the GUI.
* **GUI Controls:** Simple interface with Start/Stop, Pause/Resume, Clear, and a Save-to-File option.

## ⚙️ Prerequisites

You must have Python 3.8+ installed. This project requires the following packages, including the PyTorch backend for Whisper and the scientific libraries for audio processing.

### 1. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment:

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
Install all required Python packages using pip:
pip install torch openai-whisper sounddevice soundfile numpy scipy

🚀 Getting Started
1. Save the Code
Save the main Python script as app.py.

2. Check Microphone Access (Windows)Due to conflicts with Intel/Realtek "Smart Audio" drivers, ensure the following steps are performed:
   1. Disable Enhancements: Open Sound Settings $\rightarrow$ Recording $\rightarrow$ Right-click your selected mic $\rightarrow$ Properties $\rightarrow$ Advanced tab $\rightarrow$ UNCHECK "Enable audio
   enhancements.
   2. Check Privacy: Ensure Windows Privacy & security settings allow desktop apps to access the microphone.
   3. Use the Correct Device: If transcription fails, ensure no external applications are locking the device (like Discord or Zoom).

### 3. Run the Application
Execute the script from your activated virtual environment:

Bash

python app.py

### 4. Technical Details
The application uses the following configuration:

Setting,Value,Purpose
Whisper Model,tiny.en (CPU Optimized),Smallest English-only model for speed and low latency.
Input Sample Rate,48000 Hz (2 Channels),Matches common Windows hardware default for stream reliability.
Processing Sample Rate,16000 Hz (Mono),Required input format for the Whisper model.
Chunk Duration,5 seconds,Provides a balance between accuracy and real-time latency.
VAD Threshold,10 (RMS),Low sensitivity to accommodate muted/low-gain microphones.
