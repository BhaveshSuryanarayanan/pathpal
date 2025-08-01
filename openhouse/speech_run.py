import speech_recognition as sr
import time
# Initialize Recognizer
r = sr.Recognizer()
# r = sr.Recognizer() 
start = time.time()

import pyaudio
import wave
import time

DEVICE_INDEX = 17  
FORMAT = pyaudio.paInt16  
CHANNELS = 2  
RATE = 48000  
CHUNK = 512  
RECORD_SECONDS = 10
OUTPUT_FILE = "arducam_audio_fixed.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=DEVICE_INDEX,
                frames_per_buffer=CHUNK)

print("Recording...")

frames = []

for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    time.sleep(0.01) 
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)

print("Recording finished.")

stream.stop_stream()
stream.close()
p.terminate()

# Save to a file
with wave.open(OUTPUT_FILE, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Audio saved as {OUTPUT_FILE}")

# Open the WAV file
with sr.AudioFile("arducam_audio_fixed.wav") as source:
    audio_data = r.record(source)  # Convert to AudioData

# Recognize Speech
try:
    text = r.recognize_google(audio_data)
    print("Recognized Text:", text)
except sr.UnknownValueError:
    print("Speech could not be understood.")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech-to-Text API: {e}")
end = time.time()
print(end-start)
