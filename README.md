# LangGraph — From Concepts to Production

> A structured learning repo. Each folder is a concept. Each concept feeds a real project.  
> Stack: LangGraph · LangChain · FastAPI · Groq · Tavily · Python

---

## Why This Repo Exists

LangGraph is not just another LangChain wrapper. It's a fundamentally different way of thinking about agents — you're building **stateful graphs** where nodes are functions, edges are decisions, and cycles are intentional.

Most people learn it the wrong way: they copy a tutorial graph, it works, they don't understand why, and then they can't build anything original. This repo is structured so that never happens to you.

**The rule here:** you don't move to the next concept until you can explain the current one out loud without looking at your notes.

---

---

## Concepts Map

### 01 · State and Nodes
**The foundation. Get this wrong and everything else is shaky.**

LangGraph is built around a shared `State` object — a TypedDict that flows through every node in your graph. Nodes are just Python functions that receive the current state and return an update to it.

```python
from typing import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    messages: list
    next_step: str

def my_node(state: AgentState):
    # do something, return partial state update
    return {"next_step": "done"}
```

**What to build:** A single-node graph that takes a user query, calls Groq, and returns a response. No bells. Just state in → state out.

**What to understand before moving on:**
- Why `TypedDict` and not a regular dict
- What "reducers" are and when you need `add_messages` vs plain assignment
- Why state is immutable between nodes

---

### 02 · Edges and Conditional Edges
**This is where the graph gets smart.**

Normal edges are fixed: node A always goes to node B. Conditional edges are decision points — the graph routes differently based on what's in the state.

```python
from langgraph.graph import END

def route(state: AgentState):
    if state["next_step"] == "search":
        return "search_node"
    return END

graph.add_conditional_edges("decision_node", route)
```

**What to build:** A two-branch graph. If the user's question needs a tool call, route to a tool node. If it doesn't, route to END.

**What to understand before moving on:**
- The difference between `END` and looping back
- How to avoid infinite routing bugs
- What happens when your routing function returns an unexpected value

---

### 03 · Cycles and Loops
**This is what makes LangGraph different from a simple chain.**

Graphs can loop. An agent can call a tool, check the result, decide it needs another tool call, and loop again — up to some limit. This is the "agent loop" that powers ReAct-style agents.

```python
def should_continue(state):
    if len(state["messages"]) > 10:  # safety limit
        return END
    if state["needs_tool"]:
        return "tool_node"
    return END
```

**What to build:** A ReAct agent from scratch — no `create_react_agent` shortcut. Build the loop manually: LLM node → tool node → back to LLM → exit when done.

**What to understand before moving on:**
- How to set a maximum iteration limit to prevent infinite loops
- The difference between a loop that terminates on condition vs. on count
- Why you should never trust the LLM to always return a clean exit signal

---

### 04 · Human-in-the-Loop
**Agents that pause and ask before doing something dangerous.**

LangGraph supports `interrupt_before` and `interrupt_after` — the graph pauses execution, waits for human input, then resumes from exactly where it stopped. This is non-trivial to implement without a framework.

```python
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["dangerous_action_node"]
)
```

**What to build:** An agent that plans a sequence of actions, shows the plan to the user, waits for approval, then executes.

**What to understand before moving on:**
- Why you need a checkpointer for this to work at all
- How `thread_id` connects a paused graph to its resumed execution
- The difference between `interrupt_before` and `interrupt_after`

---

### 05 · Multi-Agent Systems
**Multiple specialized agents, one orchestrator.**

Two patterns exist: **Supervisor** (one LLM routes between sub-agents) and **Swarm** (agents hand off to each other directly). Most production systems use Supervisor.

```
User Query
    ↓
Supervisor Agent
    ↓           ↓           ↓
Research     Analysis    Writer
  Agent        Agent      Agent
    ↓           ↓           ↓
              Supervisor
                  ↓
              Final Output
```

**What to build:** A 3-agent pipeline. Supervisor receives a query, routes to either a Search Agent or an Analysis Agent, collects results, hands to a Writer Agent.

**What to understand before moving on:**
- How sub-agents share state vs. maintain their own local state
- Why a Supervisor pattern is fragile when you have 5+ sub-agents
- When to use `Command` objects for agent handoffs

---

### 06 · Persistence and Memory
**Agents that remember across sessions.**

LangGraph's checkpointers save graph state to a backend (SQLite, Postgres, Redis). A `thread_id` lets you resume a conversation exactly where it left off — even after a server restart.

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")  # swap for real DB in prod
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "user_123"}}
graph.invoke(input, config=config)
```

**What to build:** Take your Competitor Research Agent and add persistence. Each company being researched gets its own `thread_id`. Resume research on a company without starting over.

**What to understand before moving on:**
- The difference between short-term memory (state within a run) and long-term memory (across runs)
- Why `:memory:` SQLite is dev-only
- How to version your state schema when you change it mid-project

---

### 07 · Streaming
**Show progress to users instead of making them stare at a spinner.**

LangGraph supports streaming at two levels: token-by-token from the LLM (`astream_events`) and node-by-node from the graph (`astream`).

```python
async for chunk in graph.astream(input, config):
    print(chunk)  # fires after each node completes

async for event in graph.astream_events(input, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

**What to build:** Wire streaming into your FastAPI backend with Server-Sent Events (SSE). The frontend gets live updates as each node of the graph completes — not just the final answer.

**What to understand before moving on:**
- The difference between `astream` and `astream_events`
- Why streaming breaks if you're not using `async` throughout your stack
- How to filter stream events by node name

---

### 08 · Tool Calling Agents
**Real utility comes from tools. This ties everything together.**

LangGraph agents call tools via `ToolNode` — a built-in node that handles tool dispatch, execution, and result injection back into state automatically.

```python
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=3)]
tool_node = ToolNode(tools)

# bind tools to your LLM
llm_with_tools = llm.bind_tools(tools)
```

**What to build:** This IS the Competitor Research Agent. Everything from concepts 01–07 feeds into this project.

**What to understand before moving on:**
- How `ToolNode` knows which tool to call and with which arguments
- What happens when a tool throws an exception mid-graph
- How to add custom tools (not just pre-built ones)

---

## Projects

### Project 01 · Competitor Research Agent

**The goal:** User inputs a product description. The agent autonomously identifies 3–5 competitors, researches each one, and outputs a structured comparison report.

**Stack:** LangGraph · Tavily Search API · Groq · FastAPI

**Concepts required:** All of 01–08

**What makes this non-trivial:**
- The agent has to decide *when* it has enough information vs. when to search more
- Search results are noisy — the agent needs to filter, not just collect
- The output needs to be structured (JSON or Markdown table), not a blob of text

---

## Environment Setup

```bash
# Clone the repo
git clone https://github.com/ahmadimrannn/langgraph-learning
cd langgraph-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install langgraph langchain langchain-groq langchain-community tavily-python fastapi uvicorn python-dotenv

# Set up environment variables
cp .env.example .env
# Add: GROQ_API_KEY, TAVILY_API_KEY
```

---

## Environment Variables

```env
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
LANGCHAIN_TRACING_V2=true          # optional but highly recommended
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=langgraph-learning
```

> **On LangSmith:** Set it up from day one. It gives you a visual trace of every node execution, what state looked like at each step, and where your graph broke. Debugging a multi-node graph without it is a nightmare.

---

## How to Use This Repo

1. Work through concepts in order — `01` before `02`, not by interest
2. Each concept folder has a `concept.py` (the minimal runnable example) and a `README.md` (explains the *why*, not just the *what*)
3. Don't read the project code before completing all the concept folders it depends on
4. When a concept "clicks," write your own example from scratch in a `scratch.py` file — if you can't do that, you don't actually know it yet

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `langgraph` | Graph orchestration |
| `langchain-groq` | LLM calls via Groq API |
| `langchain-community` | Tools (Tavily, etc.) |
| `tavily-python` | Web search for agents |
| `fastapi` | API layer |
| `uvicorn` | ASGI server |
| `python-dotenv` | Environment management |

---

## Related Projects

| Project | Repo | Live |
|---|---|---|
| Captur (Meeting Intelligence) | [github.com/ahmadimrannn](https://github.com/ahmadimrannn) | [captur-sand.vercel.app](https://captur-sand.vercel.app) |
| AskMyDocs (RAG QA System) | [github.com/ahmadimrannn](https://github.com/ahmadimrannn) | HuggingFace Spaces |

---

*Built while learning. Broken often. Fixed anyway.*
