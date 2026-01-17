from langchain_core.tools import tool
# Import the functions we wrote in database.py
from database import save_memory as db_save, search_memory as db_search

@tool
def save_memory(text: str, user_id: str):
    """
    Saves important information, facts, tasks, or debriefs.
    CRITICAL: You MUST provide the 'user_id' from the context.
    """
    return db_save(user_id, text)

@tool
def search_memory(query: str, user_id: str):
    """
    Searches past notes and memories.
    CRITICAL: You MUST provide the 'user_id' from the context.
    """
    # Now we correctly pass the user_id to the database function
    return db_search(user_id, query)
