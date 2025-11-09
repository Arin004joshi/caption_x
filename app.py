import tkinter as tk
from tkinter import scrolledtext
import threading
import sounddevice as sd
import numpy as np
import whisper
import queue
import time
import sys
import os 
from scipy.signal import resample 

# --- Configuration ---
MODEL_NAME = "tiny.en" 
SAMPLERATE = 48000       
CHANNELS = 2             
WHISPER_TARGET_SR = 16000
CHUNK_DURATION = 5 
VOLUME_THRESHOLD = 10    # *** CHANGED: Lowered threshold for transcription ***
WHISPER_QUEUE_TIMEOUT = 5 

# Prioritizing Index 1 as it was the last one confirmed to be open and receiving *some* signal
DEVICE_INDICES_TO_TRY = [1, 5, 9, 13, 0, 4, 14, 18, 19, 20] 

# --- Whisper Model and Global Flags ---
try:
    if not os.path.exists(os.path.expanduser(f"~/.cache/whisper/{MODEL_NAME}.pt")):
        print(f"Downloading model {MODEL_NAME}. This might take a moment...")
    
    model = whisper.load_model(MODEL_NAME)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Fatal Error loading Whisper model: {e}", file=sys.stderr)
    sys.exit(1)

is_recording = False
is_paused = False
audio_queue = queue.Queue()
default_device_index = None 


# --- Function to find the FIRST working device ---
def find_working_device():
    """Iterates through known input devices and returns the index of the first one that can open a stream."""
    global default_device_index
    devices = sd.query_devices()
    device_map = {i: d['name'] for i, d in enumerate(devices) if d['max_input_channels'] > 0}
    
    print("\n--- Testing Audio Devices for Access ---")
    
    for index in DEVICE_INDICES_TO_TRY:
        if index in device_map:
            device_name = device_map[index]
            print(f"   [TESTING] Trying Index {index}: {device_name[:30]}...")
            
            try:
                test_stream = sd.InputStream(
                    samplerate=SAMPLERATE, 
                    channels=CHANNELS, 
                    dtype='int16', 
                    device=index
                )
                test_stream.start()
                test_stream.stop()
                test_stream.close()
                
                default_device_index = index
                print(f"   ✅ SUCCESS: Device Index {index} is usable with {SAMPLERATE} Hz!")
                return index

            except Exception as e:
                print(f"   ❌ FAILED: Index {index} failed to open stream. Error: {e.args[0]}")
                continue
    
    print("\n❌ FATAL: Could not find any working microphone input device.")
    return None

# --- Device Check Function ---
def check_audio_devices():
    global default_device_index
    
    working_index = find_working_device()
    default_device_index = working_index
    
    devices = sd.query_devices()
    print("\n--- Available Audio Devices (Full List) ---")
    print("Index | Name | Input Channels | Status")
    print("---------------------------------------")
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            status = "SELECTED" if i == default_device_index else ""
            print(f"{i:<5} | {d['name']:<20} | {d['max_input_channels']:<14} | {status}")
    
    if default_device_index is not None:
        print("\n*** EXPECTED OUTCOME ***")
        print("RMS should now exceed the threshold (10) when speaking. Transcription and latency updates should proceed.")


# --- Recording Thread ---
def record_audio():
    global is_recording, is_paused
    
    if default_device_index is None:
        print("FATAL: Cannot start recording, no working device found.", file=sys.stderr)
        root.after(0, lambda: toggle_recording(force_stop=True))
        return

    print(f"\nRecording started using SELECTED device {default_device_index} at {SAMPLERATE} Hz...")
    
    while is_recording:
        if is_paused:
            time.sleep(0.1) 
            continue
        try:
            start_time = time.time() 
            
            audio = sd.rec(
                int(SAMPLERATE * CHUNK_DURATION), 
                samplerate=SAMPLERATE, 
                channels=CHANNELS, 
                dtype='int16',
                device=default_device_index 
            )
            
            sd.wait() 
            
            # --- VAD Proxy & DEBUGGING ---
            mono_audio = audio[:, 0] 
            rms = np.sqrt(np.mean(mono_audio**2))
            
            # The RMS value is checked against the new, lower threshold
            if rms < VOLUME_THRESHOLD:
                print(f"DEBUG: Chunk recorded (Size: {audio.size}). RMS: {rms:.1f}. Detected SILENCE. Skipping transcription.")
                
            else:
                audio_queue.put((SAMPLERATE, audio, start_time)) 
                print(f"DEBUG: Chunk recorded (Size: {audio.size}). RMS: {rms:.1f}. Detected VOICE! Queueing for transcription.")

        except Exception as e:
            print(f"RUNTIME Recording error (Stream Closed): {e}", file=sys.stderr)
            root.after(0, lambda: toggle_recording(force_stop=True))
            break
        
    print("Recording thread gracefully stopped.")


# --- GUI Update Function ---
def update_gui(text, latency):
    caption_box.insert(tk.END, text + " ", "caption")
    caption_box.see(tk.END)
    # This will now display the actual calculated latency!
    latency_label.config(text=f"Latency: {latency:.2f}s") 

    if save_transcript.get():
        try:
            with open("transcription.txt", "a", encoding="utf-8") as f:
                f.write(text + " ")
        except Exception as e:
             print(f"Error saving file: {e}", file=sys.stderr)


# --- Transcription Thread ---
def transcribe_audio():
    global is_recording
    
    while is_recording or not audio_queue.empty(): 
        try:
            # This is where transcription and latency calculation occurs
            samplerate, audio, start_time = audio_queue.get(timeout=WHISPER_QUEUE_TIMEOUT) 
            print(f"DEBUG: Starting transcription of chunk (Duration: {audio.size/samplerate:.2f}s)...")

            # --- PRE-PROCESSING FOR WHISPER ---
            audio_mono = audio.mean(axis=1)
            num_samples = int(audio_mono.size * WHISPER_TARGET_SR / samplerate)
            audio_resampled = resample(audio_mono, num_samples)
            audio_float = audio_resampled.astype(np.float32).flatten() / 32768.0
            
            # Transcribe
            result = model.transcribe(audio_float, fp16=False, language="en")
            text = result["text"].strip()
            
            end_time = time.time() 
            latency = end_time - start_time 

            if text:
                print(f"DEBUG: Transcription successful. Text: '{text}'. Latency: {latency:.2f}s")
                root.after(0, lambda t=text, l=latency: update_gui(t, l))
            else:
                 print(f"DEBUG: Transcription returned empty text (Likely silence or very low confidence). Latency: {latency:.2f}s")
        
        except queue.Empty:
            if not is_recording:
                break 
            continue
        except Exception as e:
            print(f"Transcription runtime error: {e}", file=sys.stderr)
            
    print("Transcription thread gracefully stopped.")


# --- Button Functions (No change) ---
def toggle_recording(force_stop=False):
    global is_recording, is_paused
    if not is_recording or force_stop:
        if default_device_index is None:
             print("Cannot start: No working microphone found. Please check system settings.", file=sys.stderr)
             return
        
        is_recording = True
        is_paused = False
        record_btn.config(text="⏹ Stop", bg="#e74c3c")
        pause_btn.config(text="⏸ Pause", bg="#f1c40f", state=tk.NORMAL)
        latency_label.config(text="Latency: 0.00s") 
        with audio_queue.mutex:
             audio_queue.queue.clear()
        
        threading.Thread(target=record_audio, daemon=True).start()
        threading.Thread(target=transcribe_audio, daemon=True).start()
        
    else:
        is_recording = False
        record_btn.config(text="🎙 Start", bg="#40CE79")
        pause_btn.config(text="⏸ Pause", state=tk.DISABLED) 

def toggle_pause():
    global is_paused
    if is_recording: 
        if is_paused:
            is_paused = False
            pause_btn.config(text="⏸ Pause", bg="#f1c40f")
        else:
            is_paused = True
            pause_btn.config(text="▶ Resume", bg="#9b59b6")

def clear_text():
    caption_box.delete(1.0, tk.END)
    latency_label.config(text="Latency: 0.00s") 

# --- GUI Setup (No change) ---
root = tk.Tk()
root.title("🎧 Real-Time Voice-to-Text Transcriber (Whisper)")
root.geometry("800x550") 
root.config(bg="#1e1e1e")

check_audio_devices() 

save_transcript = tk.BooleanVar(value=False)

tk.Label(
    root, 
    text="🎙 Whisper Live Transcriber", 
    font=("Helvetica", 16, "bold"), 
    fg="white", 
    bg="#1e1e1e"
).pack(pady=10)

latency_label = tk.Label(
    root,
    text="Latency: 0.00s", 
    font=("Consolas", 12, "italic"),
    fg="#00ffcc", 
    bg="#1e1e1e"
)
latency_label.pack(pady=5)

caption_box = scrolledtext.ScrolledText(
    root, wrap=tk.WORD, width=80, height=20, font=("Consolas", 12)
)
caption_box.pack(padx=20, pady=10)
caption_box.tag_configure("caption", foreground="#00ffcc")

frame = tk.Frame(root, bg="#1e1e1e")
frame.pack(pady=10)

record_btn = tk.Button(frame, text="🎙 Start", font=("Arial", 12, "bold"), bg="#40CE79", fg="white", width=10, command=toggle_recording)
record_btn.grid(row=0, column=0, padx=10)

if default_device_index is None:
    record_btn.config(state=tk.DISABLED, text="❌ No Mic")
    
pause_btn = tk.Button(frame, text="⏸ Pause", font=("Arial", 12, "bold"), bg="#f1c40f", fg="white", width=10, command=toggle_pause)
pause_btn.grid(row=0, column=1, padx=10)
pause_btn.config(state=tk.DISABLED) 

clear_btn = tk.Button(frame, text="🧹 Clear", font=("Arial", 12, "bold"), bg="#3498db", fg="white", width=10, command=clear_text)
clear_btn.grid(row=0, column=2, padx=10)

save_check = tk.Checkbutton(
    root, 
    text="💾 Save Transcription", 
    variable=save_transcript, 
    font=("Arial", 11), 
    fg="white", 
    bg="#1e1e1e", 
    selectcolor="#1e1e1e", 
    activebackground="#1e1e1e"
)
save_check.pack(pady=5)

root.mainloop()