import re
from langchain_core.tools import tool
from database import save_memory as db_save, search_memory as db_search

def clean_user_id(raw_id: str) -> str:
    """
    Safety Filter: Removes any text (like 'User ID:') and keeps only the numbers.
    Example: 'User ID: 12345' -> '12345'
    Example: 'telegram_user' -> 'telegram_user' (fallback if no numbers)
    """
    # 1. Try to extract just the digits
    clean_id = re.sub(r"\D", "", str(raw_id))
    
    # 2. If we found numbers, return them. 
    # If the result is empty (e.g. input was just "me"), fallback to the raw input or a default.
    if clean_id:
        return clean_id
    return str(raw_id)

@tool
def save_memory(text: str, user_id: str):
    """
    Saves important information.
    Args:
        text: The content to save.
        user_id: The numeric ID provided in the context (e.g., '123456789').
    """
    # ✨ SANITIZE HERE BEFORE SAVING
    safe_id = clean_user_id(user_id)
    return db_save(safe_id, text)

@tool
def search_memory(query: str, user_id: str):
    """
    Searches past notes.
    Args:
        query: What to look for.
        user_id: The numeric ID provided in the context.
    """
    # ✨ SANITIZE HERE BEFORE SEARCHING
    safe_id = clean_user_id(user_id)
    return db_search(safe_id, query)
