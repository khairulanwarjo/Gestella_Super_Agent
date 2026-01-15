import os
import logging
import asyncio
from dotenv import load_dotenv

# Telegram Libraries
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

# OpenAI (for Whisper Voice Transcription)
from openai import OpenAI

# Import our Brain
from graph import app
from langchain_core.messages import HumanMessage

# Load keys
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup Logging (so you can see errors in terminal)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Initialize OpenAI Client for Transcription
client = OpenAI(api_key=OPENAI_API_KEY)

async def transcribe_voice(voice_file_path):
    """
    Sends the voice file to OpenAI Whisper to get text.
    We FORCE the language to English ('en') to prevent it from guessing Malay.
    """
    print("🎤 Transcribing voice (Forcing English)...")
    with open(voice_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            language="en"  # <--- THIS IS THE MAGIC FIX
        )
    return transcription.text

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id # <--- NEW: Get Chat ID
    user_text = update.message.text
    
    print(f"📩 Message from {user_id}: {user_text}")
    
    # 1. Define the Config (The "Thread")
    # This tells LangGraph: "This is conversation ID 123"
    config = {"configurable": {"thread_id": str(chat_id)}} # <--- NEW
    
    inputs = {"messages": [HumanMessage(content=user_text)]}
    
    final_response = ""
    # 2. Pass 'config' to the stream
    for event in app.stream(inputs, config=config): # <--- NEW: Pass config
        for value in event.values():
            final_response = value["messages"][-1].content
            
    await context.bot.send_message(chat_id=chat_id, text=final_response)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles Voice Notes.
    """
    user_id = update.effective_user.id
    print(f"🎤 Voice note received from {user_id}")
    
    # 1. Download the voice file
    voice_file = await context.bot.get_file(update.message.voice.file_id)
    file_path = "voice_note.ogg"
    await voice_file.download_to_drive(file_path)
    
    # 2. Transcribe
    transcribed_text = await transcribe_voice(file_path)
    print(f"📝 Transcribed: {transcribed_text}")
    
    # ... When calling the app:
    chat_id = update.effective_chat.id
    config = {"configurable": {"thread_id": str(chat_id)}} # <--- NEW
    
    inputs = {"messages": [HumanMessage(content=transcribed_text)]}
    
    final_response = ""
    for event in app.stream(inputs, config=config): # <--- NEW: Pass config
        for value in event.values():
            final_response = value["messages"][-1].content
            
    await context.bot.send_message(chat_id=chat_id, text=final_response)
    
    # Cleanup
    if os.path.exists(file_path):
        os.remove(file_path)

if __name__ == '__main__':
    print("🚀 Gestella is waking up...")
    
    # Build the Telegram App
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    # Add Handlers (Ears)
    # 1. Text Messages
    text_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    application.add_handler(text_handler)
    
    # 2. Voice Notes
    voice_handler = MessageHandler(filters.VOICE, handle_voice)
    application.add_handler(voice_handler)
    
    # Run
    application.run_polling()