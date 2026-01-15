import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# We use OpenAI for Embeddings (Converting text to numbers)
# This is independent of whether you use Claude or Gemini for the "Brain"
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

def get_embedding(text: str):
    """Generates a vector embedding using OpenAI."""
    return embeddings_model.embed_query(text)

def save_memory(user_id: str, text: str, memory_type: str = "general"):
    """
    Saves a new memory to Supabase with a vector embedding.
    """
    print(f"💾 Saving memory for {user_id}...")
    
    # 1. Generate Vector
    vector = get_embedding(text)
    
    # 2. Prepare Data Payload
    data = {
        "user_id": str(user_id),
        "content": text,
        "metadata": {"type": memory_type},
        "embedding": vector
    }
    
    # 3. Insert into Supabase
    try:
        response = supabase.table("memories").insert(data).execute()
        return f"Success: {text[:20]}..."
    except Exception as e:
        return f"Error saving memory: {str(e)}"

def search_memory(query: str, match_threshold: float = 0.1):
    """
    Semantic Search: Finds memories relevant to the user's query.
    """
    print(f"🔍 Searching memory for: {query}")
    
    # 1. Generate Vector for the Query
    query_vector = get_embedding(query)
    
    # 2. Call the Supabase function (rpc)
    try:
        response = supabase.rpc(
            "match_memories",
            {
                "query_embedding": query_vector,
                "match_threshold": match_threshold,
                "match_count": 5
            }
        ).execute()
        
        # Format results for the AI to read easily
        results = [item['content'] for item in response.data]
        return "\n".join(results) if results else "No relevant memories found."
        
    except Exception as e:
        return f"Error searching memory: {str(e)}"

# --- Quick Test Block ---
if __name__ == "__main__":
    # Run this file directly to test your database connection
    print("Testing connection...")
    
    # Uncomment these lines to test saving
    # print(save_memory("test_user_1", "I prefer my coffee black with no sugar."))
    
    # Test searching
    print(search_memory("How do I like my coffee?"))