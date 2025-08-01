import speech_recognition as sr
import time
import pyaudio
import wave

# Audio recording parameters
# DEVICE_INDEX = 11  
# global state
state = False

DEVICE_INDEX = None
FORMAT = pyaudio.paInt16  
CHANNELS = 2  
RATE = 48000  
CHUNK = 512  
RECORD_SECONDS = 10
OUTPUT_FILE = "arducam_audio_fixed.wav"

def record_audio():
    """ Records audio and saves it as a WAV file. """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                    input=True, input_device_index=DEVICE_INDEX, 
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

    with wave.open(OUTPUT_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    print(f"Audio saved as {OUTPUT_FILE}")

def recognize_speech():
    """ Converts recorded audio to text and extracts grocery items. """
    r = sr.Recognizer()
    try:
        with sr.AudioFile(OUTPUT_FILE) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            print("Recognized Text:", text)
            return text.lower().split()  # Convert text to list of words
    except sr.UnknownValueError:
        print("Speech could not be understood.")
        return []
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech-to-Text API: {e}")
        return []

# def get_grocery_list():
#     """ Combines recording and recognition to return a grocery list. """
#     global state
#     if state == False:
#         record_audio()
#         state = True
#     return recognize_speech()
def get_grocery_list():
    """ Combines recording and recognition to return a grocery list. """
    # global state
    # if state == False:
    record_audio()
    # state = True
    return recognize_speech()
