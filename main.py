import os
import logging
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Telegram Libraries
from telegram import Update
from telegram.constants import ChatAction
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

async def send_smart_response(context, chat_id, text):
    """
    SMART DISPATCHER:
    1. If text is short (< 3000 chars), send as chat message.
    2. If text is LONG (Meeting Minutes), save as .md file and send document.
    """
    if not text:
        await context.bot.send_message(chat_id=chat_id, text="⚠️ Error: Empty response from Agent.")
        return

    # THRESHOLD: If longer than this, make a file.
    if len(text) > 3000:
        # Generate a filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        filename = f"Meeting_Minutes_{timestamp}.md"
        
        # Write to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
            
        # Send the file
        await context.bot.send_message(chat_id=chat_id, text="📝 The report is long, so I've packaged it into a file for you:")
        with open(filename, "rb") as f:
            await context.bot.send_document(
                chat_id=chat_id, 
                document=f, 
                caption="Here are your structured meeting notes."
            )
        
        # Cleanup (delete file from server)
        os.remove(filename)
    else:
        # Send as normal text (split if slightly over 4096, but unlikely given threshold)
        if len(text) > 4096:
            for x in range(0, len(text), 4096):
                await context.bot.send_message(chat_id=chat_id, text=text[x:x+4096])
        else:
            await context.bot.send_message(chat_id=chat_id, text=text)


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
            language="en"
        )
    return transcription.text

async def run_agent(chat_id, user_text, context):
    # ... (Keep existing code) ...
    config = {"configurable": {"thread_id": str(chat_id)}}
    inputs = {"messages": [HumanMessage(content=user_text)]}
    
    final_response = ""
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    
    async for event in app.astream(inputs, config=config):
        for value in event.values():
            if value["messages"][-1].content:
                final_response = value["messages"][-1].content
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    return final_response

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    user_text = update.message.text
    print(f"📩 Message from {user_id}: {user_text}")
    
    try:
        response_text = await run_agent(chat_id, user_text, context)
        await send_smart_response(context, chat_id, response_text) # <--- USE NEW FUNCTION
    except Exception as e:
        print(f"❌ Error: {e}")
        await context.bot.send_message(chat_id=chat_id, text="Sorry, I encountered an error.")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ... (Keep logic, just update the sender at the end) ...
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    if update.message.voice:
        file_obj = update.message.voice
    elif update.message.audio:
        file_obj = update.message.audio
    else:
        return

    # Check Size Limit (20MB)
    if file_obj.file_size > 20 * 1024 * 1024:
        await context.bot.send_message(chat_id=chat_id, text="⚠️ File too large (>20MB).")
        return

    try:
        status_msg = await context.bot.send_message(chat_id=chat_id, text="⏳ Downloading & Transcribing...")
        
        file_ref = await context.bot.get_file(file_obj.file_id)
        file_path = "temp_audio.ogg"
        await file_ref.download_to_drive(file_path)
        
        transcript = await transcribe_voice(file_path)
        
        if len(transcript) > 500:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text="🧠 Analyzing meeting content... (This can take 1-2 mins)")
            input_text = f"Analyze this meeting recording and generate minutes: {transcript}"
        else:
            await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
            input_text = transcript

        response_text = await run_agent(chat_id, input_text, context)
        
        # USE THE NEW SMART SENDER
        await send_smart_response(context, chat_id, response_text)
        
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Error: {str(e)}")
    
    if os.path.exists(file_path):
        os.remove(file_path)

if __name__ == '__main__':
    print("🚀 Gestella is waking up...")
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    
    application.run_polling()