from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class State(TypedDict):
    # Messages are appended to the list using add_messages
    messages: Annotated[list, add_messages]

def chatbot_node(state: State):
    # Initialize the LLM with the API key from environment
    # langchain-openai looks for OPENAI_API_KEY by default
    llm = ChatOpenAI(model="gpt-4o-mini")
    return {"messages": [llm.invoke(state["messages"])]}

# Create the graph
workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# Compile the graph
graph = workflow.compile()

def chatbot(message: str):
    # Helper function to interface with main.py
    input_state = {"messages": [("user", message)]}
    result = graph.invoke(input_state)
    return result["messages"][-1].content
