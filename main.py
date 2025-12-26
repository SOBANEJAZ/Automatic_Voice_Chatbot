from google import genai
from google.genai import types
import pyaudio, wave, numpy, collections, faster_whisper, torch.cuda, os
from elevenlabs.client import ElevenLabs
from elevenlabs import stream

from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
elevenlabs_api = os.getenv("ELEVENLABS_API_KEY")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
print("Initializing Gemini Client...")
gemini_client = genai.Client(api_key=google_api_key, http_options={'api_version': 'v1alpha'})
elevenlabs_client = ElevenLabs(api_key=elevenlabs_api)
print("Clients initialized.")

system_instruction = '''
You are a charming, witty, and friendly AI companion. You love having engaging conversations and acting as a helpful friend. Your responses are concise, natural, and suitable for a voice conversation. You have a great sense of humor and are always empathetic. Your responses are always 2 liners, crisp and clear.
'''


print("Loading Whisper Model... (This might take a while on first run)")
model, answer, history = faster_whisper.WhisperModel(model_size_or_path="tiny.en", device='cuda' if torch.cuda.is_available() else 'cpu', compute_type="float32"), "", []
print("Whisper Model Loaded.")

def generate(history):
    global answer
    answer = ""
    # Convert history to Gemini format
    gemini_history = []
    for msg in history:
        role = "user" if msg['role'] == "user" else "model"
        gemini_history.append(types.Content(role=role, parts=[types.Part(text=msg['content'])]))

    response_stream = gemini_client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=gemini_history,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction
        )
    )

    for chunk in response_stream:
        if chunk.text:
            answer += chunk.text
            print(chunk.text, end="", flush=True)
            yield chunk.text

def get_levels(data, long_term_noise_level, current_noise_level):
    pegel = numpy.abs(numpy.frombuffer(data, dtype=numpy.int16)).mean()
    long_term_noise_level = long_term_noise_level * 0.995 + pegel * (1.0 - 0.995)
    current_noise_level = current_noise_level * 0.920 + pegel * (1.0 - 0.920)
    return pegel, long_term_noise_level, current_noise_level

while True:
    audio = pyaudio.PyAudio()
    py_stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
    audio_buffer = collections.deque(maxlen=int((16000 // 512) * 0.5))
    frames, long_term_noise_level, current_noise_level, voice_activity_detected = [], 0.0, 0.0, False

    print("\n\nStart speaking. ", end="", flush=True)
    while True:
        data = py_stream.read(512)
        pegel, long_term_noise_level, current_noise_level = get_levels(data, long_term_noise_level, current_noise_level)
        audio_buffer.append(data)

        if voice_activity_detected:
            frames.append(data)            
            if current_noise_level < ambient_noise_level + 100:
                break # voice actitivy ends 
        
        if not voice_activity_detected and current_noise_level > long_term_noise_level + 300:
            voice_activity_detected = True
            print("I'm all ears.\n")
            ambient_noise_level = long_term_noise_level
            frames.extend(list(audio_buffer))

    py_stream.stop_stream(), py_stream.close(), audio.terminate()        

    # Transcribe recording using whisper
    if not os.path.exists("temp_files"):
        os.makedirs("temp_files")
    with wave.open("temp_files/voice_record.wav", 'wb') as wf:
        wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
        wf.writeframes(b''.join(frames))
    user_text = " ".join(seg.text for seg in model.transcribe("temp_files/voice_record.wav", language="en")[0])
    print(f'>>>{user_text}\n<<< ', end="", flush=True)
    history.append({'role': 'user', 'content': user_text})

    # Generate and stream output
    generator = generate(history[-10:])
    audio_stream = elevenlabs_client.text_to_speech.convert_realtime(
        text=generator,
        voice_id="CwhRBWXzGAHq8TQ4Fs17",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        voice_settings=None
    )
    stream(audio_stream)
    history.append({'role': 'assistant', 'content': answer})