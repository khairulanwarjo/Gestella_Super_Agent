import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

# Lazy load Supabase to prevent crash if keys are missing during build
supabase: Client = None
if url and key:
    supabase = create_client(url, key)

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

def get_embedding(text: str):
    return embeddings_model.embed_query(text)

# --- 1. SUBSCRIPTION GATEKEEPER ---
def check_user_subscription(telegram_id: str) -> bool:
    """Checks if the user has an 'active' subscription in Supabase."""
    if not supabase: return True # Dev mode: Allow if DB missing
    try:
        response = supabase.table("users").select("subscription_status").eq("telegram_id", str(telegram_id)).execute()
        # If user exists and status is active
        if response.data and response.data[0]['subscription_status'] == 'active':
            return True
        return False
    except Exception as e:
        print(f"⚠️ Subscription check failed: {e}")
        return False

# --- 2. MULTI-USER TOKEN MANAGEMENT ---
def save_user_google_token(telegram_id: str, token_data: dict):
    """Saves the user's personal Google Token to Supabase."""
    if not supabase: return
    try:
        # Upsert: Update if exists, Insert if new
        data = {
            "telegram_id": str(telegram_id),
            "google_token": token_data,
            # We assume they are active if they are logging in, or you manage this manually
            "subscription_status": "active" 
        }
        supabase.table("users").upsert(data).execute()
        print(f"✅ Saved Google Token for user {telegram_id}")
    except Exception as e:
        print(f"❌ Error saving token: {e}")

def get_user_google_token(telegram_id: str):
    """Retrieves the user's personal Google Token."""
    if not supabase: return None
    try:
        response = supabase.table("users").select("google_token").eq("telegram_id", str(telegram_id)).execute()
        if response.data and response.data[0]['google_token']:
            return response.data[0]['google_token']
        return None
    except Exception:
        return None

# --- 3. MEMORY FUNCTIONS (Unchanged) ---
def save_memory(user_id: str, text: str, memory_type: str = "general"):
    if not supabase: return "Error: No DB"
    vector = get_embedding(text)
    data = {
        "user_id": str(user_id),
        "content": text,
        "metadata": {"type": memory_type},
        "embedding": vector
    }
    try:
        supabase.table("memories").insert(data).execute()
        return f"Success: {text[:20]}..."
    except Exception as e:
        return f"Error saving memory: {str(e)}"

def search_memory(query: str, match_threshold: float = 0.1):
    if not supabase: return "Error: No DB"
    query_vector = get_embedding(query)
    try:
        response = supabase.rpc(
            "match_memories",
            {"query_embedding": query_vector, "match_threshold": match_threshold, "match_count": 5}
        ).execute()
        results = [item['content'] for item in response.data]
        return "\n".join(results) if results else "No relevant memories found."
    except Exception as e:
        return f"Error searching memory: {str(e)}"
