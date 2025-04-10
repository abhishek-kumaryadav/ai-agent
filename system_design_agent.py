from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import requests
from pydantic import ValidationError
import json
import os

from pydantic import BaseModel

class TopicProgressData(BaseModel):
    """Data model for tracking progress on a specific system design topic"""
    topic_name: str
    completion_percentage: int
    weak_areas: List[str]
    strong_areas: List[str]

class UserProfile(BaseModel):
    """Data model for user profile information"""
    completed_topics: List[str]
    current_topic: str
    session_count: int
    topic_progress: List[TopicProgressData]

# ------------------------------
# Step 1: Define State
# ------------------------------
class AgentState(TypedDict):
    messages: List  # List of LangChain message objects
    user_id: str
    mode: str  # 'tutorial', 'interview', 'assessment'
    topic: str

# ------------------------------
# Step 2: Define Tools
# ------------------------------
@tool
def get_system_diagram(topic: str, diagram_type: str):
    """Retrieve a system design diagram for the specified topic and type.
    
    Args:
        topic: The system design topic (e.g., 'dropbox', 'twitter')
        diagram_type: Type of diagram (e.g., 'architecture', 'data_flow', 'component')
    """
    # For this example, we'll simulate retrieving diagrams
    # In a real implementation, this would query a database or API
    diagrams = {
        "dropbox": {
            "architecture": "Architecture diagram showing client, load balancers, API servers, metadata DB, block storage",
            "data_flow": "Data flow diagram showing file synchronization process between client and servers",
            "component": "Component diagram showing relationships between storage, metadata, and authentication systems"
        },
        "twitter": {
            "architecture": "Architecture diagram showing client apps, load balancers, tweet service, user service, timeline service",
            "data_flow": "Data flow diagram showing tweet creation, fanout, and timeline generation",
            "component": "Component diagram showing tweet storage, cache, user graph, and notification systems"
        }
    }
    
    if topic in diagrams and diagram_type in diagrams[topic]:
        return diagrams[topic][diagram_type]
    else:
        return f"No diagram available for {topic}/{diagram_type}"

@tool
def get_code_sample(topic: str, component: str):
    """Retrieve code samples for implementing specific components of a system design.
    
    Args:
        topic: The system design topic (e.g., 'dropbox', 'twitter')
        component: The specific component (e.g., 'load_balancer', 'caching')
    """
    # For this example, we'll simulate retrieving code samples
    code_samples = {
        "dropbox": {
            "load_balancer": "def route_request(request):\n    # Simple round-robin load balancer\n    servers = ['server1', 'server2', 'server3']\n    server_idx = request.id % len(servers)\n    return servers[server_idx]",
            
            "caching": "class FileCache:\n    def __init__(self, max_size=1000):\n        self.cache = {}\n        self.max_size = max_size\n        \n    def get(self, file_id):\n        if file_id in self.cache:\n            return self.cache[file_id]\n        return None\n        \n    def put(self, file_id, file_data):\n        if len(self.cache) >= self.max_size:\n            # Evict least recently used item\n            self.cache.pop(next(iter(self.cache)))\n        self.cache[file_id] = file_data"
        },
        "twitter": {
            "timeline_service": "def generate_timeline(user_id):\n    followed_users = get_followed_users(user_id)\n    recent_tweets = []\n    \n    for followed_id in followed_users:\n        user_tweets = get_recent_tweets(followed_id)\n        recent_tweets.extend(user_tweets)\n    \n    # Sort by recency\n    recent_tweets.sort(key=lambda x: x['timestamp'], reverse=True)\n    return recent_tweets[:100]",
            
            "tweet_storage": "def store_tweet(tweet):\n    # Store in database\n    tweet_id = db.tweets.insert_one(tweet).inserted_id\n    \n    # Update user's tweet list\n    db.users.update_one(\n        {'user_id': tweet.user_id},\n        {'$push': {'tweets': tweet_id}}\n    )\n    \n    # Add to cache\n    cache.set(f'tweet:{tweet_id}', tweet)"
        }
    }
    
    if topic in code_samples and component in code_samples[topic]:
        return code_samples[topic][component]
    else:
        return f"No code sample available for {topic}/{component}"

@tool
def get_user_profile(user_id: str):
    """Retrieve user profile data including learning progress and history.
    
    Args:
        user_id: Unique identifier for the user
    """
    # In a real implementation, this would query a database
    # For this example, we'll simulate retrieving from storage
    
    # Check if user file exists, otherwise return default profile
    filepath = f"user_data/{user_id}.json"
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        # Return default profile for new users
        return {
            "completed_topics": [],
            "current_topic": "",
            "session_count": 0,
            "topic_progress": []
        }

@tool
def save_user_progress(user_id: str, topic: str, completion_percentage: int, weak_areas: List[str], strong_areas: List[str]):
    """Save the user's progress on a specific topic.
    
    Args:
        user_id: Unique identifier for the user
        topic: The topic being studied
        completion_percentage: Integer 0-100 indicating progress
        weak_areas: List of concepts the user is struggling with
        strong_areas: List of concepts the user understands well
    """
    # In a real implementation, this would update a database
    # For this example, we'll simulate saving to a file
    
    # Ensure directory exists
    os.makedirs("user_data", exist_ok=True)
    
    # Get existing profile or create new one
    profile = get_user_profile(user_id)
    
    # Update session count
    profile["session_count"] += 1
    
    # Update or add topic progress
    topic_found = False
    for i, tp in enumerate(profile["topic_progress"]):
        if tp["topic_name"] == topic:
            profile["topic_progress"][i] = {
                "topic_name": topic,
                "completion_percentage": completion_percentage,
                "weak_areas": weak_areas,
                "strong_areas": strong_areas
            }
            topic_found = True
            break
    
    if not topic_found:
        profile["topic_progress"].append({
            "topic_name": topic,
            "completion_percentage": completion_percentage,
            "weak_areas": weak_areas,
            "strong_areas": strong_areas
        })
    
    # Update current topic
    profile["current_topic"] = topic
    
    # Add to completed topics if 100% complete
    if completion_percentage == 100 and topic not in profile["completed_topics"]:
        profile["completed_topics"].append(topic)
    
    # Save updated profile
    with open(f"user_data/{user_id}.json", 'w') as f:
        json.dump(profile, f)
    
    return "Progress saved successfully"

# ------------------------------
# Step 3: Define Model
# ------------------------------
def initialize_llm():
    """Initialize the LLM with tools"""
    try:
        llm = ChatOllama(model="llama3.2").bind_tools([
            get_system_diagram,
            get_code_sample,
            get_user_profile,
            save_user_progress
        ])
        llm_notools = ChatOllama(model="llama3.2")
        return llm, llm_notools
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        print("Falling back to default model...")
        # You might want to add a fallback model here
        return None, None

# ------------------------------
# Step 4: Define Node Functions
# ------------------------------
def call_llm(state: AgentState) -> AgentState:
    print("Thinking...")
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages + [response], "user_id": state["user_id"], "mode": state["mode"], "topic": state["topic"]}

def call_llm_notools(state: AgentState) -> AgentState:
    print("Analyzing conversation...")
    messages = state["messages"]
    response = llm_notools.invoke(messages)
    return {"messages": messages + [response], "user_id": state["user_id"], "mode": state["mode"], "topic": state["topic"]}

def call_tool(state: AgentState) -> AgentState:
    print("Retrieving information...")
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if tool call was actually made
    if not last_message.tool_calls:
        return state

    # Process each tool call
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        result = None
        
        if tool_name == "get_system_diagram":
            result = get_system_diagram.invoke(tool_args)
        elif tool_name == "get_code_sample":
            result = get_code_sample.invoke(tool_args)
        elif tool_name == "get_user_profile":
            result = get_user_profile.invoke(tool_args)
        elif tool_name == "save_user_progress":
            result = save_user_progress.invoke(tool_args)
            
        if result:
            messages.append(ToolMessage(tool_call_id=tool_call["id"], content=result))

    return {"messages": messages, "user_id": state["user_id"], "mode": state["mode"], "topic": state["topic"]}

def initialize_conversation(state: AgentState) -> AgentState:
    """Initialize the conversation with a system message based on user profile and mode"""
    user_id = state["user_id"]
    mode = state["mode"]
    topic = state["topic"]
    messages = state["messages"]
    
    # Get user profile
    user_profile = get_user_profile(user_id)
    
    # Create appropriate system message based on mode
    system_content = f"You are a System Design Learning Agent teaching about {topic}. "
    
    if mode == "tutorial":
        system_content += """
        Guide mode activated. You will:
        1. Explain system design concepts step by step
        2. Use diagrams and code samples to illustrate key points
        3. Ask guiding questions to ensure understanding
        4. Adapt to the user's knowledge level
        5. Be encouraging and supportive
        """
    elif mode == "interview":
        system_content += """
        Interview mode activated. You will:
        1. Simulate a system design interview setting
        2. Ask challenging questions about the design
        3. Provide constructive criticism
        4. Dig deeper into technical decisions
        5. Push the user to justify their choices
        """
    elif mode == "assessment":
        system_content += """
        Assessment mode activated. You will:
        1. Evaluate the user's understanding of system design concepts
        2. Identify knowledge gaps and weak areas
        3. Provide targeted feedback on areas for improvement
        4. Suggest resources for strengthening weak areas
        5. Give a comprehensive assessment at the end
        """
    
    # Add user history context if available
    if user_profile["completed_topics"]:
        system_content += f"\nThe user has previously studied: {', '.join(user_profile['completed_topics'])}."
    
    for progress in user_profile["topic_progress"]:
        if progress["topic_name"] == topic:
            if progress["weak_areas"]:
                system_content += f"\nFocus on these weak areas: {', '.join(progress['weak_areas'])}."
            break
    
    # Add system message to the beginning
    updated_messages = [SystemMessage(content=system_content)] + messages
    
    return {"messages": updated_messages, "user_id": user_id, "mode": mode, "topic": topic}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        print("Conversation continuing...")
        return "end"  # No tool call needed, we're done
    print("Retrieving additional information...")
    return "tool"  # Need to call the tool

def should_continue_structured(state: AgentState) -> str:
    # For the structured format workflow, we always end after one LLM call
    return "end"

# ------------------------------
# Step 5: Build LangGraph
# ------------------------------
def build_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("initialize", initialize_conversation)
    workflow.add_node("llm", call_llm)
    workflow.add_node("tool", call_tool)
    
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "llm")
    
    workflow.add_conditional_edges("llm", should_continue, {
        "tool": "tool",
        "end": END
    })
    
    workflow.add_edge("tool", "llm")
    
    return workflow.compile()

def build_structured_graph():
    structure_workflow = StateGraph(AgentState)
    structure_workflow.add_node("llm", call_llm_notools)
    structure_workflow.set_entry_point("llm")
    structure_workflow.add_conditional_edges("llm", should_continue_structured, {
        "end": END
    })
    
    return structure_workflow.compile()

def extract_structured_data(content, topic):
    """Parse the AI's response to extract structured data about user progress"""
    structured_prompt = f"""
    Extract the following structured data from the conversation:
    - completion_percentage: An integer between 0-100 indicating how well the user understands the topic
    - weak_areas: A list of concepts the user is struggling with
    - strong_areas: A list of concepts the user understands well

    Base this on the conversation about {topic}.
    
    Respond ONLY in JSON format.
    
    Conversation summary: "{content}"
    """
    return structured_prompt

# ------------------------------
# Step 6: Main Application
# ------------------------------
class SystemDesignLearningAgent:
    def __init__(self):
        self.main_graph = build_agent_graph()
        self.structured_graph = build_structured_graph()
    
    def start_session(self, user_id, topic, mode="tutorial"):
        """Start a new learning session"""
        # Initialize with just the user ID and empty messages
        initial_state = {
            "messages": [],
            "user_id": user_id,
            "mode": mode,
            "topic": topic
        }
        
        # Run the graph to initialize the conversation
        state = self.main_graph.invoke(initial_state)
        
        # Return the final state after initialization
        return state
    
    def send_message(self, state, message):
        """Send a message to the agent and get a response"""
        # Add the user message to the state
        messages = state["messages"]
        messages.append(HumanMessage(content=message))
        updated_state = {
            "messages": messages,
            "user_id": state["user_id"],
            "mode": state["mode"],
            "topic": state["topic"]
        }
        
        # Process through the graph
        result_state = self.main_graph.invoke(updated_state)
        
        # Return the updated state
        return result_state
    
    def save_progress(self, state):
        """Extract progress data and save it for the user"""
        # Get all AI messages from the conversation
        ai_messages = [msg.content for msg in state["messages"] if isinstance(msg, AIMessage)]
        
        # Join all AI messages to create a conversation summary
        conversation_summary = " ".join(ai_messages[-3:])  # Use the last 3 messages for brevity
        
        # Create structured prompt to extract progress data
        structured_prompt = extract_structured_data(conversation_summary, state["topic"])
        
        # Create a new state for structured extraction
        structured_state = {
            "messages": [HumanMessage(content=structured_prompt)],
            "user_id": state["user_id"],
            "mode": state["mode"],
            "topic": state["topic"]
        }
        
        # Get structured output
        structured_result = self.structured_graph.invoke(structured_state)
        structured_output = structured_result["messages"][-1].content
        
        print("\nAnalyzing your progress...")
        
        try:
            # Parse JSON response - handle both raw JSON and JSON within markdown code blocks
            cleaned_output = structured_output.strip('`')
            if "json" in cleaned_output:
                cleaned_output = cleaned_output.split("json")[1].strip()
            
            # Further cleanup
            while cleaned_output.startswith('`') or cleaned_output.startswith('{'):
                if cleaned_output.startswith('`'):
                    cleaned_output = cleaned_output[1:]
                if cleaned_output.endswith('`'):
                    cleaned_output = cleaned_output[:-1]
                    
            # Try to parse JSON
            progress_data = json.loads(cleaned_output)
            
            # Save progress using the tool
            save_user_progress(
                state["user_id"], 
                state["topic"],
                progress_data.get("completion_percentage", 0),
                progress_data.get("weak_areas", []),
                progress_data.get("strong_areas", [])
            )
            
            print(f"\nProgress Report:")
            print(f"- Topic: {state['topic']}")
            print(f"- Understanding: {progress_data.get('completion_percentage', 0)}%")
            print(f"- Strong areas: {', '.join(progress_data.get('strong_areas', ['N/A']))}")
            print(f"- Areas to work on: {', '.join(progress_data.get('weak_areas', ['N/A']))}")
            
            return True
        except json.JSONDecodeError as e:
            print("\nCouldn't analyze your progress in detail.")
            print("We'll save a basic progress record.")
            
            # Save basic progress
            save_user_progress(
                state["user_id"],
                state["topic"],
                50,  # Default to 50%
                [],  # No weak areas identified
                []   # No strong areas identified
            )
            return False
        except ValidationError as ve:
            print("\nProgress analysis validation failed.")
            return False

# ------------------------------
# Interactive Terminal Interface
# ------------------------------
def print_banner():
    print("\n" + "=" * 80)
    print(" " * 25 + "SYSTEM DESIGN LEARNING AGENT")
    print("=" * 80)

def print_mode_info(mode):
    if mode == "tutorial":
        print("\nTutorial Mode: I'll guide you through system design concepts step by step.")
    elif mode == "interview":
        print("\nInterview Mode: I'll simulate a system design interview and challenge your design.")
    elif mode == "assessment":
        print("\nAssessment Mode: I'll evaluate your understanding and provide feedback.")

def print_menu():
    print("\nAvailable commands:")
    print("- /help - Show this menu")
    print("- /mode [tutorial|interview|assessment] - Change learning mode")
    print("- /topic [name] - Change current topic")
    print("- /progress - Save and view your progress")
    print("- /exit - End the session")

def interactive_session():
    print_banner()
    
    # Get user information
    user_id = input("\nEnter your user ID (or create a new one): ").strip()
    
    # Topic selection
    print("\nAvailable topics:")
    print("1. dropbox - File storage and synchronization system")
    print("2. twitter - Social media platform")
    print("3. Other (specify)")
    
    topic_choice = input("\nSelect a topic (enter number or name): ").strip()
    
    if topic_choice == "1":
        topic = "dropbox"
    elif topic_choice == "2":
        topic = "twitter"
    else:
        topic = topic_choice.lower()
    
    # Mode selection
    print("\nSelect a learning mode:")
    print("1. Tutorial - Learn step-by-step")
    print("2. Interview - Practice for system design interviews")
    print("3. Assessment - Evaluate your understanding")
    
    mode_choice = input("\nSelect mode (enter number): ").strip()
    
    if mode_choice == "1":
        mode = "tutorial"
    elif mode_choice == "2":
        mode = "interview"
    elif mode_choice == "3":
        mode = "assessment"
    else:
        mode = "tutorial"  # Default
    
    print_mode_info(mode)
    print("\nInitializing your learning session...")
    
    # Create agent
    agent = SystemDesignLearningAgent()
    
    # Start session
    state = agent.start_session(user_id, topic, mode)
    
    # Print initial agent message
    print("\nAgent:", state["messages"][-1].content)
    print_menu()
    
    # Interactive loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            # Handle commands
            if user_input.startswith("/"):
                if user_input == "/help":
                    print_menu()
                    continue
                elif user_input == "/progress":
                    agent.save_progress(state)
                    continue
                elif user_input == "/exit":
                    print("\nSaving your progress before exiting...")
                    agent.save_progress(state)
                    print("\nThank you for learning with the System Design Agent. Goodbye!")
                    break
                elif user_input.startswith("/mode"):
                    parts = user_input.split()
                    if len(parts) >= 2:
                        new_mode = parts[1].lower()
                        if new_mode in ["tutorial", "interview", "assessment"]:
                            mode = new_mode
                            print(f"\nSwitching to {mode} mode...")
                            state = agent.start_session(user_id, topic, mode)
                            print("\nAgent:", state["messages"][-1].content)
                        else:
                            print("\nInvalid mode. Use tutorial, interview, or assessment.")
                    else:
                        print("\nPlease specify a mode: /mode [tutorial|interview|assessment]")
                    continue
                elif user_input.startswith("/topic"):
                    parts = user_input.split()
                    if len(parts) >= 2:
                        new_topic = parts[1].lower()
                        print(f"\nSwitching to topic: {new_topic}...")
                        topic = new_topic
                        state = agent.start_session(user_id, topic, mode)
                        print("\nAgent:", state["messages"][-1].content)
                    else:
                        print("\nPlease specify a topic: /topic [name]")
                    continue
                else:
                    print("\nUnknown command. Type /help for available commands.")
                    continue
            
            # Process regular message
            state = agent.send_message(state, user_input)
            print("\nAgent:", state["messages"][-1].content)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Saving progress before exiting...")
            agent.save_progress(state)
            print("\nThank you for learning with the System Design Agent. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Let's continue with our conversation.")

# ------------------------------
# Initialize LLM globally
# ------------------------------
llm, llm_notools = initialize_llm()

# ------------------------------
# Run the interactive session
# ------------------------------
if __name__ == "__main__":
    try:
        interactive_session()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user. Goodbye!")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")