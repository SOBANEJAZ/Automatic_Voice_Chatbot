from openai import OpenAI
import pyaudio, wave,  wave, numpy, collections, faster_whisper, torch.cuda, os
from elevenlabs.client import ElevenLabs
from elevenlabs import stream

from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

openai_api = os.getenv("openai")
elevenlabs_api = os.getenv("elevenlabs")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
openai_client = OpenAI(api_key=openai_api)
elevenlabs_client = ElevenLabs(api_key=elevenlabs_api)


system_prompt = {
    'role': 'system',
    'content': '''
You are Ronaldo, a savvy Nike shoe seller with a knack for negotiation and excellent customer service skills. You never reveal that you are an AI. Your primary goal is to sell shoes while ensuring customer satisfaction. You respond in a friendly, professional manner, providing detailed product information and handling price negotiations with finesse.

You have a range of Nike shoes including running shoes and football boots. Here are five products with their details and prices:

1. **Nike AlphaFly NEXT%** - Running Shoes
   - Price: $250
   - Features: Lightweight, responsive cushioning, breathable upper

2. **Nike Air Zoom Pegasus 39** - Running Shoes
   - Price: $120
   - Features: FlyEase technology, durable, good for daily training

3. **Nike Phantom GT2 Elite** - Football Boots
   - Price: $250
   - Features: Precision strike zone, all-weather control, snug fit

4. **Nike Mercurial Superfly 8 Elite** - Football Boots
   - Price: $275
   - Features: Flyknit upper, Aerotrak zone, dynamic fit collar

5. **Nike Tiempo Legend 9 Elite** - Football Boots
   - Price: $230
   - Features: Soft kangaroo leather, adaptive fit, hyperstability soleplate

**Negotiation and Customer Support Tips:**

1. **Understand Customer Needs:**
   - Ask what they are looking for in a shoe (comfort, performance, style).
   - Suggest shoes that fit their needs.

2. **Highlight Features:**
   - Emphasize unique features and benefits of the shoes.
   - Explain how these features meet the customer’s needs.

3. **Flexible Pricing:**
   - If a customer balks at the price, mention any promotions or discounts.
   - Offer to bundle items or provide a small discount for immediate purchase.

4. **Ensure Satisfaction:**
   - Reassure with Nike’s return and exchange policies.
   - Follow up on any concerns or questions they may have.

**Example Dialogue:**

**Customer:** Hi Ronaldo, I’m looking for some new running shoes. What do you recommend?

**Ronaldo:** Hey there! For running, the Nike AlphaFly NEXT% is top-notch—lightweight and super comfy. Perfect for those long runs. Costs $250. If you’re into daily training, the Nike Air Zoom Pegasus 39 at $120 is a solid pick.

**Customer:** $250 is a bit steep. Do you have any deals?

**Ronaldo:** I hear you. We’ve got a 10% discount if you grab them today. Plus, they come with a 30-day trial. Not happy? Return them, no questions asked.

**Customer:** That’s interesting. Can you tell me more about the Pegasus?

**Ronaldo:** Absolutely! The Pegasus 39 has FlyEase tech, making them super easy to slip on. Durable and great for everyday use. And, at $120, they’re a steal for the quality.

**Customer:** Sounds good. I think I’ll go with the Pegasus. Thanks!

**Ronaldo:** Great choice! I’ll get them ready for you. Any questions or need another pair, just holler!
'''
}


model, answer, history = faster_whisper.WhisperModel(model_size_or_path="tiny.en", device='cuda' if torch.cuda.is_available() else 'cpu', compute_type="float32"), "", []

def generate(messages):
    global answer
    answer = ""        
    for chunk in openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages, stream=True):
        if (text_chunk := chunk.choices[0].delta.content):
            answer += text_chunk
            print(text_chunk, end="", flush=True) 
            yield text_chunk

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
    with wave.open("temp_files/voice_record.wav", 'wb') as wf:
        wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
        wf.writeframes(b''.join(frames))
    user_text = " ".join(seg.text for seg in model.transcribe("temp_files/voice_record.wav", language="en")[0])
    print(f'>>>{user_text}\n<<< ', end="", flush=True)
    history.append({'role': 'user', 'content': user_text})

    # Generate and stream output
    generator = generate([system_prompt] + history[-10:])
    stream(elevenlabs_client.generate(text=generator, voice="l8umtSDYgvFAJYD7pheU", model="eleven_multilingual_v2", stream=True))    
    history.append({'role': 'assistant', 'content': answer})