import os
import json
import logging
import asyncio
from datetime import datetime
from dotenv import load_dotenv

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from google_auth_oauthlib.flow import InstalledAppFlow
from openai import OpenAI
from langchain_core.messages import HumanMessage

# Import our updated Database logic
from database import check_user_subscription, save_user_google_token, get_user_google_token
from graph import app

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

client = OpenAI(api_key=OPENAI_API_KEY)

# Global State for Auth Flow: { user_id: "WAITING" }
AUTH_STATE = {}

# --- 1. SETUP MASTER CREDENTIALS (YOUR APP ID) ---
def setup_master_credentials():
    """
    Creates credentials.json from the Railway Variable GOOGLE_CREDENTIALS_JSON.
    This file is SHARED by all users to authenticate against Google.
    """
    cred_data = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if cred_data:
        print("🔐 Loading Master Google App Credentials...")
        with open("credentials.json", "w") as f:
            f.write(cred_data)
    else:
        print("⚠️ Warning: GOOGLE_CREDENTIALS_JSON missing. Users cannot log in.")

# --- 2. AUTH FLOW & GATEKEEPER ---
async def check_access_and_auth(update, context):
    user_id = str(update.effective_user.id)
    
    # A. GATEKEEPER: Check Subscription
    if not check_user_subscription(user_id):
        await context.bot.send_message(
            chat_id=update.effective_chat.id, 
            text="⛔ **Access Denied**\n\nIt seems you don't have an active subscription."
        )
        return False

    # B. AUTH CHECK: Do we have their Token in Supabase?
    user_token = get_user_google_token(user_id)
    
    if user_token:
        # ✅ FIX 1: Restore the file from the Database!
        # This ensures the Calendar Tool always finds the credentials it needs.
        print(f"🔄 Restoring Google Token file for user {user_id}...")
        with open("token.json", "w") as f:
            json.dump(user_token, f)
        return True

    # C. LOGIN FLOW: If no token, ask for it.
    if user_id in AUTH_STATE and AUTH_STATE[user_id] == "WAITING":
        code = update.message.text.strip()
        
        # Basic validation
        if " " in code or len(code) < 10:
             await context.bot.send_message(chat_id=update.effective_chat.id, text="⚠️ Invalid code. Please copy the exact code from the Google page.")
             return False

        try:
            status_msg = await context.bot.send_message(chat_id=update.effective_chat.id, text="🔄 Verifying...")
            
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json',
                scopes=['https://www.googleapis.com/auth/calendar']
            )
            flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
            
            flow.fetch_token(code=code)
            
            # Save to Supabase (Cloud)
            token_json = json.loads(flow.credentials.to_json())
            save_user_google_token(user_id, token_json)
            
            # Save to Local File (For immediate use)
            with open("token.json", "w") as f:
                json.dump(token_json, f)
            
            del AUTH_STATE[user_id]
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=status_msg.message_id)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="✅ **Connected!** I am now synced with your Calendar.")
            
            # ✅ FIX 2: Stop processing here!
            # Prevents the bot from trying to "chat" with your password code.
            return False 
            
        except Exception as e:
             await context.bot.send_message(chat_id=update.effective_chat.id, text=f"❌ Login failed. Please try the link again.\nError: {e}")
             return False

    # D. SEND LOGIN LINK
    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json',
            scopes=['https://www.googleapis.com/auth/calendar']
        )
        flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
        auth_url, _ = flow.authorization_url(prompt='consent')
        
        AUTH_STATE[user_id] = "WAITING"
        
        msg = f"""
👋 **Welcome to Gestella Pro!**

To manage your calendar, I need permission.

1. Click here: [Authorize Google Calendar]({auth_url})
2. Log in and copy the code.
3. **Paste the code here.**
        """
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg, parse_mode="Markdown")
    except FileNotFoundError:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="❌ System Error: Master Credentials missing. Contact Admin.")
    
    return False

# --- STANDARD FUNCTIONS ---
async def send_smart_response(context, chat_id, text):
    if not text: return
    is_meeting = "# Executive Summary" in text or "###" in text
    is_long = len(text) > 2000

    if is_meeting or is_long:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        filename = f"Meeting_Minutes_{timestamp}.md"
        with open(filename, "w", encoding="utf-8") as f: f.write(text)
        await context.bot.send_message(chat_id=chat_id, text="📝 Here is your structured report:")
        with open(filename, "rb") as f:
            await context.bot.send_document(chat_id=chat_id, document=f, caption="Minutes.md")
        os.remove(filename)
    else:
        if len(text) > 4096:
            for x in range(0, len(text), 4096):
                await context.bot.send_message(chat_id=chat_id, text=text[x:x+4096])
        else:
            await context.bot.send_message(chat_id=chat_id, text=text)

async def transcribe_voice(voice_file_path):
    print("🎤 Transcribing...")
    with open(voice_file_path, "rb") as f:
        return client.audio.transcriptions.create(model="whisper-1", file=f, language="en").text

async def run_agent(chat_id, user_text, context):
    """
    Runs the LangGraph Agent using 'ainvoke' (Native Async).
    """
    config = {"configurable": {"thread_id": str(chat_id)}}
    
    # ✅ THE FIX: Inject the ID here!
    secure_input = f"User ID: {chat_id}\n\n{user_text}" 
    inputs = {"messages": [HumanMessage(content=secure_input)]}
    
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    print(f"🤖 Agent started for chat {chat_id}...")
    
    try:
        final_state = await app.ainvoke(inputs, config)
        messages = final_state.get("messages", [])
        
        if not messages or isinstance(messages[-1], HumanMessage):
            return "Error: Agent failed."

        final_response = messages[-1].content
        
        # Cleanup: If response is short, check history (Vacuum Logic)
        if len(final_response) < 500:
            for msg in reversed(messages):
                if not isinstance(msg, HumanMessage) and len(msg.content) > 500:
                    final_response = msg.content
                    break
        
        return final_response

    except Exception as e:
        print(f"❌ Critical Agent Error: {e}")
        return f"Error running agent: {e}"

# --- HANDLERS ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 1. Check Subscription & Auth
    if not await check_access_and_auth(update, context):
        return 

    # 2. Run Logic
    try:
        response_text = await run_agent(update.effective_chat.id, update.message.text, context)
        await send_smart_response(context, update.effective_chat.id, response_text)
    except Exception as e:
        print(f"❌ Error: {e}")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_access_and_auth(update, context): return 

    if update.message.voice: file_obj = update.message.voice
    elif update.message.audio: file_obj = update.message.audio
    else: return

    if file_obj.file_size > 20 * 1024 * 1024:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="⚠️ File too large.")
        return

    try:
        status_msg = await context.bot.send_message(chat_id=update.effective_chat.id, text="⏳ Processing...")
        file_ref = await context.bot.get_file(file_obj.file_id)
        file_path = "temp_audio.ogg"
        await file_ref.download_to_drive(file_path)
        
        transcript = await transcribe_voice(file_path)
        
        if len(transcript) > 500:
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text="🧠 Analyzing meeting...")
            input_text = f"Analyze this meeting: {transcript}"
        else:
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=status_msg.message_id)
            input_text = transcript

        response_text = await run_agent(update.effective_chat.id, input_text, context)
        await send_smart_response(context, update.effective_chat.id, response_text)
    except Exception as e:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"❌ Error: {str(e)}")
    
    if os.path.exists(file_path): os.remove(file_path)

if __name__ == '__main__':
    # 1. Setup Admin Credentials
    setup_master_credentials()
    
    print("🚀 Gestella (SaaS Mode) is waking up...")
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    application.run_polling()
