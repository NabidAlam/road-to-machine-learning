# AI Agents Comprehensive Guide

Complete guide to building advanced AI agents using CrewAI, AutoGen, Langgraph, and AutoGPT.

## Table of Contents

- [Introduction to AI Agents](#introduction-to-ai-agents)
- [CrewAI](#crewai)
- [AutoGen](#autogen)
- [Langgraph](#langgraph)
- [AutoGPT](#autogpt)
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

