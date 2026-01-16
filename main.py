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
    Handles Voice Notes AND Audio Files (MP3s).
    """
    user_id = update.effective_user.id
    
    # 1. Detect if it's a Voice Note or an Audio File
    if update.message.voice:
        file_obj = update.message.voice
        file_type = "Voice Note"
    elif update.message.audio:
        file_obj = update.message.audio
        file_type = "Audio File"
    else:
        return

    print(f"🎤 {file_type} received from {user_id}")
    
    # 2. CHECK FILE SIZE (Limit is 20MB = ~20,000,000 bytes)
    file_size = file_obj.file_size
    if file_size > 20 * 1024 * 1024:
        await context.bot.send_message(
            chat_id=update.effective_chat.id, 
            text=f"⚠️ This file is too large ({file_size/1024/1024:.1f}MB). Telegram bots can only process files under 20MB. Please compress it or split it."
        )
        return

    # 3. Download and Process
    try:
        status_msg = await context.bot.send_message(chat_id=update.effective_chat.id, text="⏳ Downloading & Transcribing...")
        
        voice_file = await context.bot.get_file(file_obj.file_id)
        file_path = "voice_note.ogg" # Whisper handles most formats fine
        await voice_file.download_to_drive(file_path)
        
        # Transcribe
        transcribed_text = await transcribe_voice(file_path)
        print(f"📝 Transcribed: {transcribed_text[:50]}...")
        
        # 4. Send to Agent Brain
        # IMPORTANT: We explicitly tell the brain this is a MEETING if it's long
        if len(transcribed_text) > 500:
             input_text = f"Analyze this meeting recording: {transcribed_text}"
        else:
             input_text = transcribed_text

        chat_id = update.effective_chat.id
        config = {"configurable": {"thread_id": str(chat_id)}}
        
        inputs = {"messages": [HumanMessage(content=input_text)]}
        
        final_response = ""
        for event in app.stream(inputs, config=config):
            for value in event.values():
                final_response = value["messages"][-1].content
                
        # Delete the "Downloading" status message
        await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
        
        await context.bot.send_message(chat_id=chat_id, text=final_response)
        
    except Exception as e:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"❌ Error: {str(e)}")
    
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
    voice_handler = MessageHandler(filters.VOICE | filters.AUDIO, handle_voice)
    application.add_handler(voice_handler)
    
    # Run
    application.run_polling()