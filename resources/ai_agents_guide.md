# AI Agents Comprehensive Guide

Complete guide to building advanced AI agents using CrewAI, AutoGen, Langgraph, and AutoGPT.

## Table of Contents

- [Introduction to AI Agents](#introduction-to-ai-agents)
- [CrewAI](#crewai)
- [AutoGen](#autogen)
- [Langgraph](#langgraph)
- [AutoGPT](#autogpt)
- [MCP (Model Context Protocol)](#mcp-model-context-protocol)
- [Agent-to-Agent (A2A) Communication](#agent-to-agent-a2a-communication)
- [Comparing Frameworks](#comparing-frameworks)
- [Real-World Projects](#real-world-projects)
- [Best Practices](#best-practices)
- [Resources](#resources)

---

## Introduction to AI Agents

### What are AI Agents?

**AI Agents** are autonomous systems that can:
- **Perceive**: Understand environment and inputs
- **Reason**: Make decisions based on goals
- **Act**: Execute actions to achieve objectives
- **Learn**: Improve from experience

### Types of AI Agents

1. **Single Agent**: One agent working alone
2. **Multi-Agent**: Multiple agents collaborating
3. **Hierarchical**: Agents with different roles/levels
4. **Swarm**: Many simple agents working together

### Why Use AI Agents?

- **Automation**: Automate complex workflows
- **Scalability**: Handle multiple tasks simultaneously
- **Specialization**: Different agents for different tasks
- **Robustness**: Distributed, fault-tolerant systems

---

## CrewAI

### What is CrewAI?

**CrewAI** is a framework for orchestrating role-playing, autonomous AI agents. Agents work together in a crew to accomplish tasks.

### Installation

```bash
pip install crewai
```

### Basic Example

```python
from crewai import Agent, Task, Crew

# Define agents
researcher = Agent(
    role="Research Analyst",
    goal="Research and analyze information",
    backstory="You are an expert researcher",
    verbose=True
)

writer = Agent(
    role="Content Writer",
    goal="Write engaging content",
    backstory="You are a skilled writer",
    verbose=True
)

# Define tasks
research_task = Task(
    description="Research the latest trends in AI",
    agent=researcher
)

writing_task = Task(
    description="Write an article based on the research",
    agent=writer
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True
)

# Execute
result = crew.kickoff()
print(result)
```

### Advanced CrewAI

```python
from crewai import Agent, Task, Crew, Process

# Specialized agents
data_scientist = Agent(
    role="Data Scientist",
    goal="Analyze data and extract insights",
    backstory="Expert in data analysis and ML",
    tools=[python_tool, sql_tool],
    verbose=True
)

ml_engineer = Agent(
    role="ML Engineer",
    goal="Build and deploy ML models",
    backstory="Expert in ML model development",
    tools=[mlflow_tool, docker_tool],
    verbose=True
)

# Sequential tasks
analysis_task = Task(
    description="Analyze the dataset and provide insights",
    agent=data_scientist,
    expected_output="Analysis report with key findings"
)

model_task = Task(
    description="Build ML model based on analysis",
    agent=ml_engineer,
    context=[analysis_task],
    expected_output="Trained ML model with evaluation metrics"
)

# Create crew with sequential process
crew = Crew(
    agents=[data_scientist, ml_engineer],
    tasks=[analysis_task, model_task],
    process=Process.sequential,  # Tasks run in order
    verbose=True
)

result = crew.kickoff()
```

### Custom Tools

```python
from crewai_tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information"""
    # Implementation
    return results

@tool
def analyze_data(file_path: str) -> str:
    """Analyze data file and return insights"""
    # Implementation
    return insights

# Use in agent
agent = Agent(
    role="Analyst",
    tools=[search_web, analyze_data],
    verbose=True
)
```

---

## AutoGen

### What is AutoGen?

**AutoGen** is a framework for building multi-agent conversational systems. Agents can have conversations, use tools, and solve problems together.

### Installation

```bash
pip install pyautogen
```

### Basic Example

```python
import autogen

# Configure LLM
config_list = [
    {
        "model": "gpt-4",
        "api_key": "your-api-key",
    }
]

# Create agents
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list}
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding"}
)

# Start conversation
user_proxy.initiate_chat(
    assistant,
    message="Write Python code to sort a list"
)
```

## Multi-Agent Systems

Multi-agent systems use **multiple specialized agents** (e.g., planner, coder, reviewer) that collaborate to solve a task. This is useful when:

- the task is complex and benefits from role separation
- you want automatic review/verification steps
- you want tool-using agents coordinating workflows (RAG, browsing, code execution)

### Multi-Agent Conversation

```python
# Create multiple specialized agents
coder = autogen.AssistantAgent(
    name="coder",
    system_message="You are a Python expert. Write code to solve problems.",
    llm_config={"config_list": config_list}
)

reviewer = autogen.AssistantAgent(
    name="reviewer",
    system_message="You review code for quality and correctness.",
    llm_config={"config_list": config_list}
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding"}
)

# Group chat
groupchat = autogen.GroupChat(
    agents=[user_proxy, coder, reviewer],
    messages=[],
    max_round=12
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config={"config_list": config_list}
)

user_proxy.initiate_chat(
    manager,
    message="Build a web scraper in Python"
)
```

### Function Calling

```python
import autogen

# Define function
def get_weather(city: str) -> str:
    """Get weather for a city"""
    # Implementation
    return f"Weather in {city}: Sunny, 25°C"

# Register function
functions = [
    {
        "name": "get_weather",
        "description": "Get weather information for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["city"]
        }
    }
]

# Create agent with function calling
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list,
        "functions": functions
    },
    function_map={"get_weather": get_weather}
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    function_map={"get_weather": get_weather}
)

user_proxy.initiate_chat(
    assistant,
    message="What's the weather in New York?"
)
```

---

## Langgraph

### What is Langgraph?

**Langgraph** is a library for building stateful, multi-actor applications with LLMs. It's built on Langchain and provides graph-based agent workflows.

### Installation

```bash
pip install langgraph
```

### Basic Example

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define state
class State(TypedDict):
    messages: Annotated[list, operator.add]

# Define nodes
def agent_node(state: State):
    # Agent logic
    return {"messages": [response]}

def tool_node(state: State):
    # Tool execution
    return {"messages": [tool_result]}

# Create graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Add edges
workflow.set_entry_point("agent")
workflow.add_edge("agent", "tools")
workflow.add_conditional_edges(
    "tools",
    should_continue,  # Function to decide next step
    {
        "continue": "agent",
        "end": END
    }
)

# Compile
app = workflow.compile()

# Run
result = app.invoke({"messages": [{"role": "user", "content": "Hello"}]})
```

### Advanced Langgraph

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.tools import tool

# Define tools
@tool
def search_web(query: str) -> str:
    """Search the web"""
    return results

@tool
def calculate(expression: str) -> str:
    """Calculate mathematical expression"""
    return str(eval(expression))

tools = [search_web, calculate]

# Create tool node
tool_node = ToolNode(tools)

# Agent with tool calling
from langchain.agents import create_react_agent

agent = create_react_agent(llm, tools, prompt)

# Create graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)
workflow.add_node("tools", tool_node)

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")
workflow.set_entry_point("agent")

app = workflow.compile()
```

### Multi-Agent Workflow

```python
# Define different agent roles
researcher = create_react_agent(llm, research_tools, research_prompt)
writer = create_react_agent(llm, writing_tools, writing_prompt)
reviewer = create_react_agent(llm, review_tools, review_prompt)

# Create workflow
workflow = StateGraph(State)

workflow.add_node("researcher", researcher)
workflow.add_node("writer", writer)
workflow.add_node("reviewer", reviewer)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")

workflow.add_conditional_edges(
    "reviewer",
    check_quality,
    {
        "approve": END,
        "revise": "writer"
    }
)

app = workflow.compile()
```

---

## AutoGPT

### What is AutoGPT?

**AutoGPT** is an autonomous AI agent that can break down complex goals into sub-tasks and execute them autonomously using GPT and other tools.

### Installation

```bash
git clone https://github.com/Significant-Gravitas/AutoGPT.git
cd AutoGPT
pip install -r requirements.txt
```

### Basic Usage

```python
# AutoGPT is typically run via command line
# python -m autogpt --gpt3only --continuous

# Or programmatically
from autogpt.agent import Agent
from autogpt.config import Config

config = Config()
config.set_continuous_mode(True)

agent = Agent(
    ai_name="Assistant",
    memory=None,
    full_message_history=[],
    next_action_count=0,
    system_prompt="You are a helpful assistant",
    triggering_prompt="Determine which next command to use"
)

# Set goal
agent.start_interaction_loop(
    user_input="Research and write a report on AI trends"
)
```

### Custom AutoGPT

```python
from autogpt.agent.agent import Agent
from autogpt.config import Config
from autogpt.memory import get_memory

config = Config()
memory = get_memory(config)

agent = Agent(
    ai_name="Researcher",
    memory=memory,
    full_message_history=[],
    next_action_count=0,
    system_prompt="You are an expert researcher",
    triggering_prompt="Research the topic and provide insights"
)

# Execute goal
agent.start_interaction_loop(
    user_input="Research machine learning trends in 2024"
)
```

---

## Comparing Frameworks

| Framework | Best For | Strengths | Weaknesses |
|-----------|----------|-----------|------------|
| **CrewAI** | Role-based multi-agent tasks | Easy to use, clear roles | Less flexible |
| **AutoGen** | Conversational multi-agent systems | Great conversations, tool use | Can be verbose |
| **Langgraph** | Complex stateful workflows | Flexible, graph-based | Steeper learning curve |
| **AutoGPT** | Autonomous goal completion | Fully autonomous | Can be unpredictable |

### When to Use What

- **CrewAI**: When you need agents with clear roles working together
- **AutoGen**: When you need conversational agents with tool use
- **Langgraph**: When you need complex, stateful workflows
- **AutoGPT**: When you need fully autonomous goal completion

---

## Advanced Agent Architectures

### ReAct: Reasoning and Acting

**ReAct** (Reasoning + Acting) is a framework that combines reasoning and acting in language models, allowing them to interact with external tools while maintaining a reasoning trace.

**Key Concept:**
- **Reasoning**: LLM thinks step-by-step about the problem
- **Acting**: LLM uses tools to gather information or perform actions
- **Iterative**: Alternates between reasoning and acting

**Architecture:**
```
Thought → Action → Observation → Thought → Action → ...
```

**Implementation:**
```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

# Define tools
tools = [
    Tool(
        name="Search",
        func=search_function,
        description="Search the web for information"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="Perform mathematical calculations"
    )
]

# Create ReAct agent
agent = initialize_agent(
    tools,
    llm,
    agent="react-docstore",  # ReAct agent type
    verbose=True
)

# Use agent
result = agent.run(
    "What is the population of Berlin? Multiply it by 2 and tell me the result."
)
```

**ReAct Prompt Structure:**
```
Question: What is the capital of France?

Thought: I need to find the capital of France. I can use a search tool.
Action: Search[capital of France]
Observation: Paris is the capital of France.
Thought: I have the answer.
Action: Finish[Paris]
```

**Benefits:**
- **Transparency**: Reasoning trace is visible
- **Tool Integration**: Seamlessly uses external tools
- **Error Recovery**: Can reason about failures and retry
- **Interpretability**: Easy to understand agent decisions

### Program-Aided Language Models (PAL)

**PAL** is an approach where LLMs generate code (Python) to solve problems, then execute the code to get answers.

**Key Concept:**
- LLM generates Python code to solve the problem
- Code is executed in a sandboxed environment
- Results are returned to the LLM

**Why PAL?**
- **Precision**: Code execution is exact, not approximate
- **Complex Reasoning**: Handles multi-step calculations
- **Verification**: Code can be inspected and verified
- **Reproducibility**: Same code produces same results

**Implementation:**
```python
from langchain.agents import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)

# Create PAL agent
agent = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True
)

# Use agent for mathematical reasoning
result = agent.run(
    """
    A store has 100 apples. They sell 30% on Monday, 
    25% of remaining on Tuesday. How many are left?
    Write Python code to solve this.
    """
)
```

**PAL Prompt Structure:**
```
Question: If a train travels 120 km in 2 hours, what's its average speed?

# solution using Python:
def solution():
    distance = 120  # km
    time = 2  # hours
    speed = distance / time
    return speed

print(solution())
```

**Use Cases:**
- **Mathematical Problems**: Complex calculations, algebra
- **Data Analysis**: Process and analyze data
- **Algorithmic Problems**: Implement algorithms
- **Scientific Computing**: Physics, chemistry calculations

**Safety Considerations:**
- **Sandboxing**: Execute code in isolated environment
- **Resource Limits**: Limit execution time and memory
- **Input Validation**: Validate code before execution
- **Error Handling**: Catch and handle execution errors

**Comparison:**

| Approach | Strengths | Weaknesses | Best For |
|----------|-----------|------------|----------|
| **Direct Generation** | Fast, simple | May be inaccurate | Simple Q&A |
| **ReAct** | Transparent, tool use | Can be verbose | Complex multi-step tasks |
| **PAL** | Precise, verifiable | Requires code execution | Mathematical, algorithmic problems |

---

## Real-World Projects

### Project 1: Research and Writing Crew

```python
from crewai import Agent, Task, Crew

# Agents
researcher = Agent(
    role="Researcher",
    goal="Research topics thoroughly",
    backstory="Expert researcher",
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Write engaging articles",
    backstory="Skilled content writer",
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Edit and improve content",
    backstory="Experienced editor",
    verbose=True
)

# Tasks
research_task = Task(
    description="Research AI trends",
    agent=researcher
)

writing_task = Task(
    description="Write article based on research",
    agent=writer,
    context=[research_task]
)

editing_task = Task(
    description="Edit and polish the article",
    agent=editor,
    context=[writing_task]
)

# Crew
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    verbose=True
)

result = crew.kickoff()
```

### Project 2: Code Development Team

```python
import autogen

# Agents
architect = autogen.AssistantAgent(
    name="architect",
    system_message="You design software architecture",
    llm_config=config_list
)

developer = autogen.AssistantAgent(
    name="developer",
    system_message="You write code based on architecture",
    llm_config=config_list
)

tester = autogen.AssistantAgent(
    name="tester",
    system_message="You write and run tests",
    llm_config=config_list
)

# Group chat
groupchat = autogen.GroupChat(
    agents=[architect, developer, tester],
    messages=[],
    max_round=20
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=config_list
)

user_proxy.initiate_chat(
    manager,
    message="Build a REST API for a todo app"
)
```

### Project 3: Data Analysis Pipeline

```python
from langgraph.graph import StateGraph

# Define workflow
workflow = StateGraph(State)

# Add nodes for each step
workflow.add_node("load_data", load_data_node)
workflow.add_node("clean_data", clean_data_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("visualize", visualize_node)
workflow.add_node("report", report_node)

# Define flow
workflow.set_entry_point("load_data")
workflow.add_edge("load_data", "clean_data")
workflow.add_edge("clean_data", "analyze")
workflow.add_edge("analyze", "visualize")
workflow.add_edge("visualize", "report")
workflow.add_edge("report", END)

app = workflow.compile()
result = app.invoke({"data_path": "data.csv"})
```

---

## MCP (Model Context Protocol)

### What is MCP?

**Model Context Protocol (MCP)** is an open protocol that standardizes how LLM applications can access context in terms of tools and data resources. It uses a client-server architecture to enable:
- **Context Sharing**: Share information between models and agents
- **State Management**: Maintain conversation state across agents
- **Tool Integration**: Standardize tool/function calling
- **Interoperability**: Work across different AI frameworks
- **Resource Access**: Expose data resources and prompt templates

### MCP Architecture

**Client-Server Model:**
```
MCP Client (LLM Application)
    ↕
MCP Server (Tools & Resources)
```

**Key Components:**
1. **MCP Client**: Hosted inside your LLM application
2. **MCP Server**: Exposes tools, data resources, and prompt templates
3. **Protocol**: Standardized communication format

### Why MCP?

**Before MCP:**
- Each application needed custom integrations
- Duplicate code for common tools
- Difficult to share tools across applications

**With MCP:**
- Reusable MCP servers for common tasks
- Standardized protocol
- Easy integration with multiple applications
- Community-built servers available

### Key Concepts

1. **Context Providers**: Sources of context (databases, APIs, files)
2. **Context Consumers**: Models/agents that use context
3. **Context Format**: Standardized JSON schema for context
4. **Protocol Handlers**: Implementations for different frameworks

### MCP Implementation Example

```python
# MCP Context Provider
class MCPContextProvider:
    def __init__(self):
        self.context_store = {}
    
    def add_context(self, agent_id: str, context: dict):
        """Add context for an agent"""
        if agent_id not in self.context_store:
            self.context_store[agent_id] = []
        self.context_store[agent_id].append(context)
    
    def get_context(self, agent_id: str) -> list:
        """Retrieve context for an agent"""
        return self.context_store.get(agent_id, [])
    
    def share_context(self, from_agent: str, to_agent: str, context_key: str):
        """Share specific context between agents"""
        context = self.get_context(from_agent)
        shared = [c for c in context if context_key in c]
        self.add_context(to_agent, {"shared_from": from_agent, "data": shared})

# Usage
mcp = MCPContextProvider()

# Agent 1 adds context
mcp.add_context("researcher", {
    "topic": "AI trends",
    "findings": ["LLMs are growing", "Multimodal AI is emerging"],
    "timestamp": "2024-01-15"
})

# Agent 2 retrieves context
context = mcp.get_context("researcher")
print(context)  # [{"topic": "AI trends", ...}]

# Share context between agents
mcp.share_context("researcher", "writer", "findings")
```

### MCP with Langchain

```python
from langchain.schema import BaseMessage
from typing import List, Dict

class MCPContextManager:
    def __init__(self):
        self.contexts: Dict[str, List[BaseMessage]] = {}
    
    def store_context(self, agent_id: str, messages: List[BaseMessage]):
        """Store conversation context"""
        self.contexts[agent_id] = messages
    
    def retrieve_context(self, agent_id: str) -> List[BaseMessage]:
        """Retrieve conversation context"""
        return self.contexts.get(agent_id, [])
    
    def merge_contexts(self, agent_ids: List[str]) -> List[BaseMessage]:
        """Merge contexts from multiple agents"""
        merged = []
        for agent_id in agent_ids:
            merged.extend(self.retrieve_context(agent_id))
        return merged

# Usage with Langchain agents
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory

mcp_manager = MCPContextManager()

# Agent 1 conversation
agent1_memory = ConversationBufferMemory()
agent1_memory.save_context({"input": "Research AI trends"}, {"output": "LLMs are growing"})
mcp_manager.store_context("researcher", agent1_memory.chat_memory.messages)

# Agent 2 uses shared context
agent2_memory = ConversationBufferMemory()
shared_context = mcp_manager.retrieve_context("researcher")
for msg in shared_context:
    agent2_memory.chat_memory.add_message(msg)
```

### MCP Best Practices

1. **Context Versioning**: Version your context for tracking changes
2. **Context Filtering**: Only share relevant context to reduce noise
3. **Privacy**: Be careful with sensitive data in shared context
4. **Performance**: Cache frequently accessed context
5. **Standardization**: Use consistent context schemas

---

## Agent-to-Agent (A2A) Communication

### What is A2A Communication?

**Agent-to-Agent (A2A) Communication** enables direct communication between AI agents without human intervention. It includes:
- **Message Passing**: Agents send messages to each other
- **Event Broadcasting**: Agents publish/subscribe to events
- **Shared State**: Agents access shared memory/state
- **Coordination Protocols**: Standardized communication patterns

### A2A Communication Patterns

#### 1. Direct Messaging

```python
class Agent:
    def __init__(self, name: str):
        self.name = name
        self.inbox = []
        self.peers = {}
    
    def register_peer(self, peer_name: str, peer_agent):
        """Register another agent as a peer"""
        self.peers[peer_name] = peer_agent
    
    def send_message(self, recipient: str, message: dict):
        """Send message to another agent"""
        if recipient in self.peers:
            self.peers[recipient].receive_message(self.name, message)
    
    def receive_message(self, sender: str, message: dict):
        """Receive message from another agent"""
        self.inbox.append({
            "from": sender,
            "message": message,
            "timestamp": time.time()
        })
    
    def process_messages(self):
        """Process messages in inbox"""
        for msg in self.inbox:
            print(f"{self.name} received from {msg['from']}: {msg['message']}")
        self.inbox.clear()

# Usage
researcher = Agent("researcher")
writer = Agent("writer")

researcher.register_peer("writer", writer)
writer.register_peer("researcher", researcher)

# Researcher sends findings to writer
researcher.send_message("writer", {
    "type": "research_findings",
    "content": "AI trends: LLMs are growing rapidly"
})

# Writer processes message
writer.process_messages()
```

#### 2. Event-Based Communication

```python
from typing import Callable, Dict, List
import asyncio

class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    def publish(self, event_type: str, data: dict):
        """Publish an event"""
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                handler(data)

# Agent with event support
class EventAgent:
    def __init__(self, name: str, event_bus: EventBus):
        self.name = name
        self.event_bus = event_bus
    
    def publish_event(self, event_type: str, data: dict):
        """Publish an event"""
        self.event_bus.publish(event_type, data)
    
    def handle_event(self, event_type: str, handler: Callable):
        """Subscribe to events"""
        self.event_bus.subscribe(event_type, handler)

# Usage
bus = EventBus()

researcher = EventAgent("researcher", bus)
writer = EventAgent("writer", bus)

# Writer subscribes to research events
def handle_research(data):
    print(f"Writer received research: {data}")

writer.handle_event("research_complete", handle_research)

# Researcher publishes event
researcher.publish_event("research_complete", {
    "topic": "AI trends",
    "findings": ["LLMs growing", "Multimodal emerging"]
})
```

#### 3. Shared Memory Communication

```python
from threading import Lock
from typing import Dict, Any

class SharedMemory:
    def __init__(self):
        self.memory: Dict[str, Any] = {}
        self.lock = Lock()
    
    def write(self, key: str, value: Any):
        """Write to shared memory"""
        with self.lock:
            self.memory[key] = value
    
    def read(self, key: str) -> Any:
        """Read from shared memory"""
        with self.lock:
            return self.memory.get(key)
    
    def read_all(self) -> Dict[str, Any]:
        """Read all memory"""
        with self.lock:
            return self.memory.copy()

class SharedMemoryAgent:
    def __init__(self, name: str, shared_memory: SharedMemory):
        self.name = name
        self.memory = shared_memory
    
    def share_data(self, key: str, value: Any):
        """Share data with other agents"""
        self.memory.write(key, value)
        print(f"{self.name} shared: {key} = {value}")
    
    def read_shared_data(self, key: str) -> Any:
        """Read data shared by other agents"""
        return self.memory.read(key)

# Usage
shared_mem = SharedMemory()

researcher = SharedMemoryAgent("researcher", shared_mem)
writer = SharedMemoryAgent("writer", shared_mem)

# Researcher shares findings
researcher.share_data("research_findings", {
    "topic": "AI trends",
    "findings": ["LLMs growing", "Multimodal emerging"]
})

# Writer reads shared findings
findings = writer.read_shared_data("research_findings")
print(f"Writer read: {findings}")
```

#### 4. Request-Response Pattern

```python
import asyncio
from typing import Optional, Dict

class RequestResponseAgent:
    def __init__(self, name: str):
        self.name = name
        self.request_handlers: Dict[str, Callable] = {}
        self.pending_requests: Dict[str, asyncio.Future] = {}
    
    def register_handler(self, request_type: str, handler: Callable):
        """Register handler for request type"""
        self.request_handlers[request_type] = handler
    
    async def send_request(self, recipient: 'RequestResponseAgent', 
                          request_type: str, data: dict) -> dict:
        """Send request and wait for response"""
        request_id = f"{self.name}_{id(data)}"
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        # Send request
        await recipient.handle_request(self.name, request_type, data, request_id)
        
        # Wait for response
        response = await future
        return response
    
    async def handle_request(self, sender: str, request_type: str, 
                            data: dict, request_id: str):
        """Handle incoming request"""
        if request_type in self.request_handlers:
            handler = self.request_handlers[request_type]
            response = await handler(data)
            # Send response back
            await self.send_response(sender, request_id, response)
    
    async def send_response(self, recipient: str, request_id: str, 
                           response: dict):
        """Send response to request"""
        # In real implementation, this would notify the recipient
        pass

# Usage
async def main():
    researcher = RequestResponseAgent("researcher")
    writer = RequestResponseAgent("writer")
    
    # Writer registers handler
    async def handle_research_request(data):
        return {"status": "research_complete", "findings": ["LLMs growing"]}
    
    writer.register_handler("research_request", handle_research_request)
    
    # Researcher sends request
    response = await researcher.send_request(
        writer, "research_request", {"topic": "AI trends"}
    )
    print(f"Researcher received: {response}")

# asyncio.run(main())
```

### A2A Communication Best Practices

1. **Message Queuing**: Use message queues for reliable delivery
2. **Error Handling**: Handle communication failures gracefully
3. **Timeouts**: Set timeouts for requests
4. **Authentication**: Authenticate agent communications
5. **Monitoring**: Monitor A2A communication patterns
6. **Idempotency**: Make operations idempotent
7. **Rate Limiting**: Prevent message flooding

### A2A with CrewAI

```python
from crewai import Agent, Task, Crew

# Agents can communicate through shared tasks
researcher = Agent(
    role="Researcher",
    goal="Research and share findings",
    backstory="Expert researcher",
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Write based on research",
    backstory="Skilled writer",
    verbose=True
)

# Task that passes data between agents
research_task = Task(
    description="Research AI trends and document findings",
    agent=researcher,
    expected_output="Research findings document"
)

writing_task = Task(
    description="Write article based on research findings",
    agent=writer,
    expected_output="Article about AI trends",
    context=[research_task]  # Writer has access to research output
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task]
)

result = crew.kickoff()
```

---

## Best Practices

1. **Clear Roles**: Define clear roles for each agent
2. **Goal Setting**: Set specific, measurable goals
3. **Error Handling**: Handle errors gracefully
4. **Monitoring**: Monitor agent behavior and outputs
5. **Testing**: Test with diverse scenarios
6. **Cost Management**: Monitor API usage and costs
7. **Security**: Secure API keys and sensitive data

---

## Resources

### Official Documentation

- [CrewAI Documentation](https://docs.crewai.com/)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [Langgraph Documentation](https://langchain-ai.github.io/langgraph/)
- [AutoGPT GitHub](https://github.com/Significant-Gravitas/AutoGPT)

### Tutorials

- [CrewAI Tutorials](https://docs.crewai.com/tutorials/)
- [AutoGen Examples](https://github.com/microsoft/autogen/tree/main/notebook)
- [Langgraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)

### Community

- [CrewAI Discord](https://discord.gg/crewai)
- [AutoGen GitHub Discussions](https://github.com/microsoft/autogen/discussions)

---

**Remember**: AI agents are powerful but require careful design. Start simple, test thoroughly, and iterate!

