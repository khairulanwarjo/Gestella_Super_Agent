import os
import datetime
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver # <--- NEW: Import Memory

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage # <--- NEW: For the persona

# Import Tools (Ensure your tools folder structure is correct)
# If using the 'flat' structure on GitHub, remove 'tools.' prefix
try:
    from tools.memory import save_memory, search_memory
    from tools.calculator import calculator
    from tools.calendar import list_calendar_events, add_calendar_event
    from tools.meeting import analyze_meeting
except ImportError:
    # Fallback for flat structure
    from memory import save_memory, search_memory
    from calculator import calculator
    from calendar import list_calendar_events, add_calendar_event
    from meeting import analyze_meeting

load_dotenv()

def init_llm(provider: str = "openai"):
    if provider == "openai":
        return ChatOpenAI(model="gpt-4o", temperature=0)
    elif provider == "claude":
        return ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    else:
        raise ValueError(f"Unknown provider: {provider}")

llm = init_llm("openai") 

# --- CONNECT TOOLS ---
tools_list = [save_memory, search_memory, calculator, list_calendar_events, add_calendar_event, analyze_meeting] # <--- Added here
llm_with_tools = llm.bind_tools(tools_list)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

async def chatbot_node(state: AgentState):
    # --- DYNAMIC CONFIGURATION ---
    user_name = os.getenv("USER_NAME", "Sir")
    bot_name = os.getenv("BOT_NAME", "Gestella")
    bot_personality = os.getenv("BOT_PERSONALITY","an elite executive assistant.")
    user_location = os.getenv("USER_LOCATION", "Singapore (GMT+8)")
    
    now = datetime.datetime.now()
    current_time_str = now.strftime("%A, %d %B %Y, %I:%M %p")
    
    # --- UPDATED PERSONA ---
    persona_text = f"""
    You are {bot_name}, {bot_personality} You assist {user_name}.
    
    CURRENT CONTEXT:
    - Today is: {current_time_str}
    - User Location: {user_location}
    
    CRITICAL RULES:
    1. **SYSTEM INJECTION:** The user's message will start with "User ID: <ID>". 
       - You MUST extract this <ID> and use it as the 'user_id' argument for the 'save_memory' and 'search_memory' tools.
       - **DO NOT** ask the user for their ID. You already have it.
       - **DO NOT** mention the User ID in your final response.
       
    2. If the user provides enough info for a calendar event, just DO IT.
    3. Speak English/Singlish.
    4. If the user sends a LONG voice note, use 'analyze_meeting'.
    """
    
    persona = SystemMessage(content=persona_text)
    
    if isinstance(state["messages"][0], SystemMessage):
        state["messages"][0] = persona
        messages = state["messages"]
    else:
        messages = [persona] + state["messages"]

    # Async invoke
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools_list) 

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"

workflow = StateGraph(AgentState)
workflow.add_node("agent", chatbot_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# --- MEMORY SETUP ---
memory = MemorySaver() # <--- Using RAM to fix Async Error
app = workflow.compile(checkpointer=memory)
