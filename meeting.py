from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# We use a dedicated LLM instance for this
llm_analyst = ChatOpenAI(model="gpt-4o", temperature=0)

@tool
async def analyze_meeting(transcript: str) -> str:
    """
    Analyzes a long meeting transcript and produces a structured 'Notion-style' minute report.
    Use this when the user asks to 'summarize meeting', 'debrief', or 'create notes' from a long text/voice.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert Meeting Analyst and Minute Taker.
        Your goal is to convert raw, messy meeting transcripts into structured, professional notes.
        
        Follow this EXACT format structure (Use Markdown):
        
        # 📝 [Meeting Title/Topic based on content]
        
        ## ⚡ Executive Summary
        (A 3-5 sentence high-level overview of the entire discussion)
        
        ## 🏗️ Key Discussion Points
        (Group the points by category/topic. Use bullet points.)
        - **[Category Name]**: [Detail]
        
        ## 💰 Financials & Logistics (If applicable)
        (Extract any budget numbers, dates, locations, or specific vendor details)
        
        ## ❓ Decisions Made & Questions Raised
        - ✅ **Decision:** [What was agreed?]
        - ❓ **Open Question:** [What is still unsolved?]
        
        ## 🚀 Action Items (Crucial)
        (List every single task mentioned with the person responsible if known)
        - [ ] Task 1 (Owner)
        - [ ] Task 2 (Owner)
        """),
        ("user", "{transcript}")
    ])
    
    chain = prompt | llm_analyst
    
    try:
        # ASYNC CHANGE: We use 'ainvoke' instead of 'invoke'
        # This prevents the bot from freezing during long analysis
        result = await chain.ainvoke({"transcript": transcript})
        return result.content
    except Exception as e:
        return f"Error analyzing meeting: {str(e)}"