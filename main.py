import os
import json
import logging
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Telegram Libraries
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

# OpenAI
from openai import OpenAI

# LangGraph Brain
from graph import app
from langchain_core.messages import HumanMessage

# Load keys
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

client = OpenAI(api_key=OPENAI_API_KEY)

# --- NEW: AUTO-LOGIN FUNCTION ---
def setup_google_credentials():
    """
    Checks for Google Credentials in Environment Variables and creates the files.
    This allows 'One-Click' deployment without manually uploading sensitive files.
    """
    print("🔐 Checking for Google Credentials in Environment...")
    
    # 1. The Token (Login Session)
    # In Railway, you will create a variable: GOOGLE_TOKEN_JSON
    # and paste the entire content of token.json into it.
    token_data = os.getenv("GOOGLE_TOKEN_JSON")
    if token_data:
        print("   ✅ Found GOOGLE_TOKEN_JSON. Creating token.json...")
        with open("token.json", "w") as f:
            f.write(token_data)
    else:
        print("   ⚠️ No GOOGLE_TOKEN_JSON found. (If running locally, ensure token.json exists).")

    # 2. The Credentials (App ID)
    # In Railway, you will create a variable: GOOGLE_CREDENTIALS_JSON
    cred_data = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if cred_data:
        print("   ✅ Found GOOGLE_CREDENTIALS_JSON. Creating credentials.json...")
        with open("credentials.json", "w") as f:
            f.write(cred_data)

async def send_smart_response(context, chat_id, text):
    """
    SMART DISPATCHER:
    1. If it looks like Meeting Minutes (contains '# '), ALWAYS send as a file.
    2. If text is huge (> 2000 chars), send as a file.
    3. Otherwise, send as chat message.
    """
    if not text:
        await context.bot.send_message(chat_id=chat_id, text="⚠️ Error: I processed the audio, but the final text response was empty. Please try again.")
        return

    # LOGIC UPDATE: Force file if it has Markdown Headers OR is long
    is_meeting_notes = "# Executive Summary" in text or "###" in text
    is_long = len(text) > 2000

    if is_meeting_notes or is_long:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        filename = f"Meeting_Minutes_{timestamp}.md"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
            
        await context.bot.send_message(chat_id=chat_id, text="📝 Here is your structured report:")
        with open(filename, "rb") as f:
            await context.bot.send_document(
                chat_id=chat_id, 
                document=f, 
                caption="Meeting_Minutes.md"
            )
        os.remove(filename)
    else:
        if len(text) > 4096:
            for x in range(0, len(text), 4096):
                await context.bot.send_message(chat_id=chat_id, text=text[x:x+4096])
        else:
            await context.bot.send_message(chat_id=chat_id, text=text)

async def transcribe_voice(voice_file_path):
    print("🎤 Transcribing voice (Forcing English)...")
    with open(voice_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            language="en"
        )
    return transcription.text

async def run_agent(chat_id, user_text, context):
    """
    Runs the LangGraph Agent using 'ainvoke' (Native Async).
    """
    config = {"configurable": {"thread_id": str(chat_id)}}
    inputs = {"messages": [HumanMessage(content=user_text)]}
    
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    print(f"🤖 Agent started for chat {chat_id} (ASYNC INVOKE)...")
    
    try:
        final_state = await app.ainvoke(inputs, config)
        messages = final_state.get("messages", [])
        
        # DEBUG LOGS
        for m in messages:
            print(f" - {type(m).__name__}: {len(m.content)} chars")

        if not messages or isinstance(messages[-1], HumanMessage):
            print("❌ Agent did not run! Returning error.")
            return "Error: The Agent failed to execute the logic. Please try again."

        # VACUUM LOGIC (Find the longest message)
        final_response = messages[-1].content
        
        if len(final_response) < 500:
            print("⚠️ Final response is short. Searching history for the long report...")
            for msg in reversed(messages):
                if not isinstance(msg, HumanMessage) and len(msg.content) > 500:
                    final_response = msg.content
                    print(f"✅ Found the long report in history ({len(final_response)} chars)")
                    break
        
        return final_response

    except Exception as e:
        print(f"❌ Critical Agent Error: {e}")
        return f"Error running agent: {e}"

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    user_text = update.message.text
    print(f"📩 Message from {user_id}: {user_text}")
    
    try:
        response_text = await run_agent(chat_id, user_text, context)
        await send_smart_response(context, chat_id, response_text)
    except Exception as e:
        print(f"❌ Error: {e}")
        await context.bot.send_message(chat_id=chat_id, text="Sorry, I encountered an error.")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    if update.message.voice:
        file_obj = update.message.voice
    elif update.message.audio:
        file_obj = update.message.audio
    else:
        return

    if file_obj.file_size > 20 * 1024 * 1024:
        await context.bot.send_message(chat_id=chat_id, text="⚠️ File too large (>20MB). Please compress it.")
        return

    try:
        status_msg = await context.bot.send_message(chat_id=chat_id, text="⏳ Downloading & Transcribing...")
        
        file_ref = await context.bot.get_file(file_obj.file_id)
        file_path = "temp_audio.ogg"
        await file_ref.download_to_drive(file_path)
        
        transcript = await transcribe_voice(file_path)
        print(f"📝 Transcript length: {len(transcript)} chars")
        
        if len(transcript) > 500:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text="🧠 Analyzing meeting content... (This can take 1-2 mins)")
            input_text = f"Analyze this meeting recording and generate minutes: {transcript}"
        else:
            await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
            input_text = transcript

        response_text = await run_agent(chat_id, input_text, context)
        await send_smart_response(context, chat_id, response_text)
        
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Error: {str(e)}")
    
    if os.path.exists(file_path):
        os.remove(file_path)

if __name__ == '__main__':
    # 1. Run the credential setup FIRST
    setup_google_credentials()
    
    # 2. Get Bot Name from settings (Defaults to Gestella)
    bot_name = os.getenv("BOT_NAME", "Gestella")
    print(f"🚀 {bot_name} is waking up...")
    
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    
    application.run_polling()
