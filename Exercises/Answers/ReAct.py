from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain import hub
import os

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int):
    """Function for addition."""
    return a + b

@tool
def subtract(a: int, b: int):
    "Function for subtraction."
    return a - b

@tool
def multiply(a: int, b: int):
    "Function for multiplication."
    return a * b

tools = [add, subtract, multiply] 

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_VERSION"),
    api_version=os.getenv("AZURE_OPENAI_DEPLOYMENT_VERSION")
).bind_tools(tools)

def model_call(state: AgentState):
    system_prompt = SystemMessage(content="You are a helpful AI assistant")
    response = llm.invoke([system_prompt, *state['messages']])
    return {"messages": [response]}

def decider(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
        
    



graph = StateGraph(AgentState)

graph.add_node("agent", model_call)
tool_node = ToolNode(tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")


graph.add_conditional_edges(
    "agent",
    decider,
    {
        "continue": "tools",
        "end": END,
    }
    
)

graph.add_edge("tools", "agent")

app = graph.compile()









def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40689 + 12 and then multiply the result by 6. Also tell me a joke about UCL please.")]}
print_stream(app.stream(inputs, stream_mode="values"))
    




