import json
import os
import getpass
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain.chat_models import init_chat_model
from langchain.agents import initialize_agent, AgentType, Tool
# from langchain.tools import DuckDuckGoSearchResults
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langsmith import traceable



# =======================================================
# 1️⃣ Environment Setup
# =======================================================
load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"

if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")

if not os.environ.get("LANGSMITH_PROJECT"):
    project = getpass.getpass('Enter your LangSmith Project Name (default="default"): ')
    os.environ["LANGSMITH_PROJECT"] = project if project else "default"

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter Groq API key: ")

print("✅ Initialization complete.")

# =======================================================
# 2️⃣ Define Web Search Tool
# =======================================================
search_tool = DuckDuckGoSearchResults(name="web_search", backend="duckduckgo")

tools = [
    Tool(
        name="WebSearch",
        func=search_tool.run,
        description="Useful for fetching recent and factual information from the web."
    )
]

print("🔍 DuckDuckGo search tool is ready.")

# =======================================================
# 3️⃣ Define State for LangGraph
# =======================================================
class PaperState(TypedDict):
    user_query: str
    search_results: List[str]
    paper_title: str
    abstract: str

# =======================================================
# 4️⃣ Initialize LLM (Groq-hosted Llama 3.3)
# =======================================================
llm = init_chat_model(
    "Llama-3.3-70B-Versatile",
    model_provider="groq",
    temperature=0.3
)
print("🤖 Groq-hosted Llama model initialized.")

# =======================================================
# 5️⃣ Define Prompts & LLM Chains
# =======================================================
title_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template=(
        "You are a research assistant.\n"
        "Based on the topic and search results below, suggest a clear and formal research paper title.\n\n"
        "Topic: {query}\n\n"
        "Search Context:\n{context}\n\n"
        "Return only the title:"
    ),
)

abstract_prompt = PromptTemplate(
    input_variables=["title", "context"],
    template=(
        "You are an academic writer.\n"
        "Using the given research paper title and search context, write a well-structured abstract (200–300 words).\n"
        "The abstract should mention motivation, method, and key contributions.\n\n"
        "Title: {title}\n\n"
        "Search Context:\n{context}\n\n"
        "Abstract:"
    ),
)

title_chain = LLMChain(llm=llm, prompt=title_prompt)
abstract_chain = LLMChain(llm=llm, prompt=abstract_prompt)


# =======================================================
# 6️⃣ Define LangGraph Node Functions
# =======================================================

def node_duckduckgo_search(state: PaperState) -> PaperState:
    """Run DuckDuckGo search using the defined tool."""
    query = state["user_query"]
    print(f"\n🔍 Searching web for: {query}")
    raw_results = tools[0].func(query)
    try:
        parsed = json.loads(raw_results)
        summaries = [r.get("snippet") or r.get("body") or str(r) for r in parsed][:5]
    except Exception:
        summaries = [raw_results[:500]]  # fallback to plain text
    print(f"✅ Retrieved {len(summaries)} snippets from DuckDuckGo.")
    return {"search_results": summaries}


def node_generate_title(state: PaperState) -> PaperState:
    context = "\n".join(state["search_results"])
    query = state["user_query"]
    title = title_chain.run(query=query, context=context)
    print(f"\n📘 Generated Title:\n{title.strip()}")
    return {"paper_title": title.strip()}


def node_generate_abstract(state: PaperState) -> PaperState:
    context = "\n".join(state["search_results"])
    title = state["paper_title"]
    abstract = abstract_chain.run(title=title, context=context)
    print("\n📄 Abstract generated successfully.\n")
    return {"abstract": abstract.strip()}

# =======================================================
# 7️⃣ Build LangGraph Flow
# =======================================================
builder = StateGraph(PaperState)

builder.add_node("search_web", node_duckduckgo_search)
builder.add_node("generate_title", node_generate_title)
builder.add_node("generate_abstract", node_generate_abstract)

builder.add_edge(START, "search_web")
builder.add_edge("search_web", "generate_title")
builder.add_edge("generate_title", "generate_abstract")
builder.add_edge("generate_abstract", END)

graph = builder.compile()

# =======================================================
# 8️⃣ Helper Function to Run the Agent
# =======================================================
@traceable(run_type="chain")
def run_title_abstract_agent(topic: str):
    input_state = {
        "user_query": topic,
        "search_results": [],
        "paper_title": "",
        "abstract": "",
    }
    print(f"\n🚀 Starting research agent for topic: {topic}\n")
    result = graph.invoke(input_state)
    print("\n✅ --- Final Output --- ✅")
    print("\n📘 Title:\n", result["paper_title"])
    print("\n📄 Abstract:\n", result["abstract"])
    return result

# =======================================================
# 9️⃣ Example Run
# =======================================================
# Uncomment to test
run_title_abstract_agent("AI-driven prediction of crop yields using satellite imagery")



