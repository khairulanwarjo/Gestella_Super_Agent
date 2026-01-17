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

# Google Auth Libraries
from google_auth_oauthlib.flow import InstalledAppFlow

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

# Global State to track user login attempts
# Format: { user_id: "WAITING_FOR_CODE" }
AUTH_STATE = {}

# --- 1. SETUP MASTER CREDENTIALS ---
def setup_google_app():
    """
    Creates credentials.json from the Environment Variable.
    This is YOUR Master App ID that you put in Railway.
    """
    cred_data = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if cred_data:
        print("🔐 Loading Master Google App Credentials...")
        with open("credentials.json", "w") as f:
            f.write(cred_data)
    else:
        print("⚠️ Warning: No GOOGLE_CREDENTIALS_JSON found. Auth will fail.")

# --- 2. THE NEW AUTH GATEKEEPER ---
async def check_auth_status(update, context):
    """
    Checks if the bot has a valid token.json.
    If not, it guides the user through the login flow.
    """
    user_id = update.effective_user.id
    
    # A. If we already have the file, we are safe!
    if os.path.exists("token.json"):
        return True
    
    # B. Check if we are waiting for the user to paste the code
    if user_id in AUTH_STATE and AUTH_STATE[user_id] == "WAITING_FOR_CODE":
        code = update.message.text.strip()
        
        # Simple check to see if it looks like a code (prevent analyzing random text)
        if len(code) < 10: 
             await context.bot.send_message(chat_id=update.effective_chat.id, text="⚠️ That doesn't look like a Google code. Please copy the code from the link.")
             return False

        try:
            status_msg = await context.bot.send_message(chat_id=update.effective_chat.id, text="🔄 Verifying code...")
            
            # Load the flow using YOUR Master Credentials
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json',
                scopes=['https://www.googleapis.com/auth/calendar']
            )
            # Special redirect for copy-paste method
            flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
            
            # Exchange the code for the actual token
            flow.fetch_token(code=code)
            
            # Save the token locally!
            with open('token.json', 'w') as token:
                token.write(flow.credentials.to_json())
            
            # Cleanup state
            del AUTH_STATE[user_id]
            
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=status_msg.message_id)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="✅ **Success!** I am now connected to your Calendar.\n\nYou can ask me to schedule things now!")
            return True
            
        except Exception as e:
             await context.bot.send_message(chat_id=update.effective_chat.id, text=f"❌ Authorization failed. The code might be expired.\nPlease click the link and try again.")
             # We don't return False here immediately to let them try again, but usually we restart flow
             return False

    # C. Start the Flow (First time user sees this)
    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json',
            scopes=['https://www.googleapis.com/auth/calendar']
        )
        flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
        
        auth_url, _ = flow.authorization_url(prompt='consent')
        
        AUTH_STATE[user_id] = "WAITING_FOR_CODE"
        
        msg = f"""
🛑 **Action Required**

To manage your calendar, I need your permission.

1. Click this Google Link:
{auth_url}

2. Log in and copy the code.
3. **Paste the code here.**
        """
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    except FileNotFoundError:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="❌ System Error: Master Credentials missing. Please contact Admin.")
    
    return False

# --- STANDARD FUNCTIONS ---
async def send_smart_response(context, chat_id, text):
    if not text: return
    
    is_meeting_notes = "# Executive Summary" in text or "###" in text
    is_long = len(text) > 2000

    if is_meeting_notes or is_long:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        filename = f"Meeting_Minutes_{timestamp}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        await context.bot.send_message(chat_id=chat_id, text="📝 Here is your structured report:")
        with open(filename, "rb") as f:
            await context.bot.send_document(chat_id=chat_id, document=f, caption="Meeting_Minutes.md")
        os.remove(filename)
    else:
        if len(text) > 4096:
            for x in range(0, len(text), 4096):
                await context.bot.send_message(chat_id=chat_id, text=text[x:x+4096])
        else:
            await context.bot.send_message(chat_id=chat_id, text=text)

async def transcribe_voice(voice_file_path):
    print("🎤 Transcribing...")
    with open(voice_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file, language="en")
    return transcription.text

async def run_agent(chat_id, user_text, context):
    config = {"configurable": {"thread_id": str(chat_id)}}
    inputs = {"messages": [HumanMessage(content=user_text)]}
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    
    try:
        final_state = await app.ainvoke(inputs, config)
        messages = final_state.get("messages", [])
        
        if not messages or isinstance(messages[-1], HumanMessage):
            return "Error: Agent failed to respond."

        final_response = messages[-1].content
        if len(final_response) < 500:
            for msg in reversed(messages):
                if not isinstance(msg, HumanMessage) and len(msg.content) > 500:
                    final_response = msg.content
                    break
        return final_response
    except Exception as e:
        return f"Error: {e}"

# --- UPDATED HANDLERS ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    user_text = update.message.text
    print(f"📩 Message from {user_id}: {user_text}")
    
    # 1. GATEKEEPER CHECK
    is_authenticated = await check_auth_status(update, context)
    if not is_authenticated:
        return # Stop here, user needs to login first

    # 2. Run Normal Logic
    try:
        response_text = await run_agent(chat_id, user_text, context)
        await send_smart_response(context, chat_id, response_text)
    except Exception as e:
        print(f"❌ Error: {e}")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    # 1. GATEKEEPER CHECK
    is_authenticated = await check_auth_status(update, context)
    if not is_authenticated:
        return 

    # 2. Run Voice Logic
    if update.message.voice: file_obj = update.message.voice
    elif update.message.audio: file_obj = update.message.audio
    else: return

    if file_obj.file_size > 20 * 1024 * 1024:
        await context.bot.send_message(chat_id=chat_id, text="⚠️ File too large (>20MB).")
        return

    try:
        status_msg = await context.bot.send_message(chat_id=chat_id, text="⏳ Processing...")
        file_ref = await context.bot.get_file(file_obj.file_id)
        file_path = "temp_audio.ogg"
        await file_ref.download_to_drive(file_path)
        
        transcript = await transcribe_voice(file_path)
        
        if len(transcript) > 500:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=status_msg.message_id, text="🧠 Analyzing meeting...")
            input_text = f"Analyze this meeting: {transcript}"
        else:
            await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
            input_text = transcript

        response_text = await run_agent(chat_id, input_text, context)
        await send_smart_response(context, chat_id, response_text)
        
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Error: {str(e)}")
    
    if os.path.exists(file_path): os.remove(file_path)

if __name__ == '__main__':
    # 1. Load the Master Credentials for Auth Flow
    setup_google_app()
    
    bot_name = os.getenv("BOT_NAME", "Gestella")
    print(f"🚀 {bot_name} is waking up...")
    
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    
    application.run_polling()
