from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import requests
from pydantic import ValidationError
import json

from langchain_core.runnables import Runnable


from pydantic import BaseModel

class WeatherData(BaseModel):
    temperature: str
    wind_speed: str


# ------------------------------
# Step 1: Define State
# ------------------------------
class AgentState(TypedDict):
    messages: List  # List of LangChain message objects

# ------------------------------
# Step 2: Define Tool
# ------------------------------
@tool
def get_weather(latitude, longitude):
    """This is a publically available API that returns the weather for a given location based on latitude and longitude."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]

# ------------------------------
# Step 3: Define Model
# ------------------------------
llm = ChatOllama(model="llama3.2").bind_tools([get_weather])
llm_notools = ChatOllama(model="llama3.2")

# ------------------------------
# Step 4: Define Node Functions
# ------------------------------
def call_llm(state: AgentState) -> AgentState:
    print("calling llm")
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages + [response]}

def call_llm_notools(state: AgentState) -> AgentState:
    print("calling llm no tools")
    messages = state["messages"]
    response = llm_notools.invoke(messages)
    return {"messages": messages + [response]}

def call_tool(state: AgentState) -> AgentState:
    print("calling tool")
    messages = state["messages"]
    # print(messages)
    last_message = messages[-1]
    # Check if tool call was actually made
    if not last_message.tool_calls:
        return {"messages": messages}

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name == "get_weather":
        result = get_weather.invoke(tool_args)
        messages.append(ToolMessage(tool_call_id=tool_call["id"], content=result))

    return {"messages": messages}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        print("routing to end")
        return "end"  # need to call the tool
    print("routing to tool")
    return "tool"  # no tool call, we're done

# ------------------------------
# Step 5: Build LangGraph
# ------------------------------
workflow = StateGraph(AgentState)  # ✅ Pass the state schema
workflow.add_node("llm", call_llm)
workflow.add_node("tool", call_tool)
workflow.set_entry_point("llm")
workflow.add_conditional_edges("llm", should_continue, {
    "tool": "tool",
    "end": END
})
workflow.add_edge("tool", "llm")
graph = workflow.compile()

structure_workflow = StateGraph(AgentState)  # ✅ Pass the state schema
structure_workflow.add_node("llm", call_llm_notools)
structure_workflow.set_entry_point("llm")
structure_graph = structure_workflow.compile()

def prompt_structured(message):
    structured_output = f"""
        Extract the following structured data from the message:
        - temperature: Temperature of the place
        - wind_speed: Speed of the win

        Respond ONLY in JSON format.

        Message: "{message}"
    """
    return structured_output
# ------------------------------
# Step 6: Run the agent
# ------------------------------
if __name__ == "__main__":
    prompt = """
        What's the weather in Banglore? Get latitute and logitude of bangalore
        """
    initial_messages = [HumanMessage(content=prompt)]
    unstructed_state = graph.invoke({"messages": initial_messages})
    
    unstructed_messages = unstructed_state["messages"]
    structured_prompt = [(HumanMessage(content=prompt_structured(unstructed_messages[-1].content)))]
    print("\nstructured prompt", structured_prompt)
    final_state = structure_graph.invoke({"messages": structured_prompt})
    print("\nstructured output", final_state["messages"][-1].content)


    try:
        data_dict = json.loads(final_state["messages"][-1].content)
        weather = WeatherData(**data_dict)
        print("\n✅ Validated Weather Data:")
        print(weather)
    except json.JSONDecodeError as e:
        print("\n❌ Invalid JSON format from LLM:")
        print(e)
    except ValidationError as ve:
        print("\n❌ Validation failed:")
        print(ve)