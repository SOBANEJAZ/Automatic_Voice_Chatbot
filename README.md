# 🎙️🤖 Seamless and Real-Time Voice Interaction with AI Voice Chatbot 🗣️💬

🚀 Uses faster_whisper and elevenlabs input streaming for low latency responses to spoken input.

## 🛠️ Setup

### 1. 🔑 API Keys

Rename the `.env_example` to `.env` and enter your OpenAI and ElevenLabs API key values in the .env file

### 2. 📦 Dependencies 

#### 🎵 mpv player installation

Make sure you have installed mpv player.

> 🖥️ Windows Users:

1. 📥 Download the latest mpv archive file from https://mpv.io/
2. 📁 Extract the folder and rename it to mpv
3. 📌 Copy the folder to "C:\Program Files"
4. 🔒 Give user permissions (right-click "mpv" folder, properties, security, Users --> Full Control checkbox)
5. 🏃‍♂️ Go to installer folder and run the installer as Administrator
6. 📋 Copy the directory "C:\Program Files\mpv"
7. 🔍 Open "System Environment Variables" from the Windows search bar and click "Environment Variables"
8. 📝 In System environment variables, find "Path", click "Edit"
9. ➕ Add "C:\Program Files\mpv" to the list and click Ok on all windows
10. 🔄 Restart the system

> 🍎 Mac Users:

Run: `brew install mpv`

> 🐧 Linux Users:

Install it yourself.  You're not like Windows users and You are the chad and chads don't need help.. 😉

#### 📚 Python Libraries:

Install the required Python libraries:

```bash
pip install openai elevenlabs pyaudio wave keyboard faster_whisper numpy torch
```
### 3. 🏃‍♀️ Run the Script:

Execute the main script:

```bash
python main.py
```
🗣️ Talk into your microphone and 
👂 Listen to the reply

## 🤝 Contribute

🌟 Feel free to fork, improve, and submit pull requests. If you're considering significant changes or additions, please start by opening an issue.