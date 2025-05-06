import speech_recognition as sr
import pyaudio
import wave
import os
import threading
import sys

# Recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WAVE_OUTPUT_FILENAME = "output.wav"

# Flag to control recording
is_recording = True
frames = []

def record_audio(stream, p):
    """Records audio chunks into the global frames list."""
    global frames, is_recording
    print("Recording... Press Enter to stop.")
    while is_recording:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False) # Add exception handling for overflow
            frames.append(data)
        except IOError as e:
            # Handle potential input overflow or other stream errors
            print(f"Stream read error: {e}")
            # Optionally decide if this error should stop recording
            # is_recording = False
    print("Finished recording.")

# Initialize PyAudio
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Start recording in a separate thread
record_thread = threading.Thread(target=record_audio, args=(stream, p))
record_thread.start()

# Wait for user to press Enter in the main thread
input("Press Enter to stop recording...\n")
is_recording = False # Signal the recording thread to stop

# Wait for the recording thread to finish
record_thread.join()

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded data as a WAV file
if not frames:
    print("No audio recorded.")
    sys.exit() # Exit if no frames were captured

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
# Use get_sample_size from the PyAudio instance 'p' that was used to open the stream
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"Saved recording as {WAVE_OUTPUT_FILENAME}")

# Initialize the recognizer
r = sr.Recognizer()

# Process the saved audio file for speech-to-text
audio_file_to_process = WAVE_OUTPUT_FILENAME # Directly use the WAV file

if not os.path.exists(audio_file_to_process):
    print(f"Error: Audio file {audio_file_to_process} not found for processing.")
    sys.exit()

try:
    # use the audio file as the audio source
    with sr.AudioFile(audio_file_to_process) as source:
        print(f"Processing {audio_file_to_process} for speech-to-text...")
        # reads the audio file.
        audio_data = r.record(source) # Reads the entire file

        # Using google to recognize audio
        print("Sending audio to Google Speech Recognition...")
        my_text = r.recognize_google(audio_data)
        my_text = my_text.lower()

        print(f"Write the Following Memory w/ Punctuation & Typo Adjustments. Try to use existing memories to guess the names of the people in the memory. Make sure you review any name guesses with me before writing the memory and feel free to ask me about any people that you would want a name for. Memory: {my_text}")

except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")

except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")

except FileNotFoundError:
    print(f"Error: Audio file not found at {audio_file_to_process}")

except Exception as e: # Catch other potential exceptions during recognition
    print(f"An unexpected error occurred during speech recognition: {e}")

finally:
    # Clean up the audio file
    if os.path.exists(audio_file_to_process):
        try:
            os.remove(audio_file_to_process)
            print(f"Removed temporary file {audio_file_to_process}")
        except Exception as e:
            print(f"Could not remove temporary file {audio_file_to_process}: {e}")
