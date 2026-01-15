# test.py
from graph import app
from langchain_core.messages import HumanMessage

# 1. Define a test input
input_text = "Save a note that I have a budget of $5000 for the event."

print(f"User: {input_text}")
print("🤖 Agent is running...")

# 2. Run the Agent
inputs = {"messages": [HumanMessage(content=input_text)]}

for event in app.stream(inputs):
    for value in event.values():
        print("Response:", value["messages"][-1].content)

print("\n--- Test 2: Checking Memory ---")
# 3. Ask it to recall
input_text_2 = "How much is my budget?"
inputs_2 = {"messages": [HumanMessage(content=input_text_2)]}

for event in app.stream(inputs_2):
    for value in event.values():
        print("Response:", value["messages"][-1].content)