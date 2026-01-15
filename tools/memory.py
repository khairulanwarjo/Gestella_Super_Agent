from langchain_core.tools import tool
# Import the functions we wrote in database.py
# Note: We go 'up' one level to import database hence the simple import might fail 
# if not running as a module. 
# FIX: When running from main.py, 'import database' works if database.py is in root.
from database import save_memory as db_save, search_memory as db_search

@tool
def save_memory(text: str, user_id: str = "telegram_user"):
    """
    Saves important information, facts, tasks, or debriefs to the user's brain.
    Useful when the user asks you to 'remember' something or 'note this down'.
    """
    # In a real app, we would inject the user_id automatically. 
    # For now, we default to 'telegram_user' or let the LLM guess.
    return db_save(user_id, text)

@tool
def search_memory(query: str):
    """
    Searches past notes, debriefs, and facts in the database.
    Useful when you need to answer a question based on the user's past context.
    """
    return db_search(query)