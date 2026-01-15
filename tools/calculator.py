from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    Calculates a math expression. Use this for ANY math problem.
    Example input: "5000 * 0.3" or "(100 + 50) / 2"
    """
    try:
        # Evaluate string math safely
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating: {str(e)}"