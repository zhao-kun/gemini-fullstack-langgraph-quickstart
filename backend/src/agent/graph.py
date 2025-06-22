import os

import os
from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)
from agent.search_tools import create_search_tool

load_dotenv()

# Initialize Google client only if using Google provider
genai_client = None
if os.getenv("GEMINI_API_KEY"):
    genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init LLM client
    llm = configurable.get_llm_client(
        model_name=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using various search tools.

    Executes web search using Google Search API, Firecrawl API, Brave Search API, or falls back to LLM knowledge.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    if configurable.search_tool == "google" and configurable.llm_provider == "google":
        # Use Google's native search with grounding
        if not genai_client:
            raise ValueError("Google client not initialized. GEMINI_API_KEY may be missing.")
            
        response = genai_client.models.generate_content(
            model=configurable.query_generator_model,
            contents=formatted_prompt,
            config={
                "tools": [{"google_search": {}}],
                "temperature": 0,
            },
        )
        # resolve the urls to short urls for saving tokens and time
        resolved_urls = resolve_urls(
            response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
        )
        # Gets the citations and adds them to the generated text
        citations = get_citations(response, resolved_urls)
        modified_text = insert_citation_markers(response.text, citations)
        sources_gathered = [item for citation in citations for item in citation["segments"]]
        
    elif configurable.search_tool == "firecrawl":
        # Use Firecrawl for web search and scraping
        try:
            # Create Firecrawl search tool
            search_tool = create_search_tool(
                "firecrawl",
                api_key=configurable.firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY"),
                base_url=configurable.firecrawl_base_url
            )
            
            if not search_tool:
                raise ValueError("Failed to create Firecrawl search tool")
            
            # Perform search and scraping
            search_results = search_tool.search_and_scrape(
                query=state["search_query"],
                max_results=configurable.max_search_results,
                max_content_length=configurable.max_content_length
            )
            
            if search_results:
                # Format results for LLM
                search_content = search_tool.format_search_results(search_results)
                
                # Use LLM to synthesize research based on scraped content
                llm = configurable.get_llm_client(
                    model_name=configurable.query_generator_model,
                    temperature=0,
                    max_retries=2
                )
                
                research_prompt = f"""
                {formatted_prompt}
                
                Based on the following web search results, provide a comprehensive research response:
                
                {search_content}
                
                Please synthesize the information from these sources and provide insights relevant to the research topic.
                Reference the sources using [1], [2], etc. format where appropriate.
                """
                
                response = llm.invoke(research_prompt)
                modified_text = response.content
                
                # Create citations compatible with existing system
                citations = search_tool.create_citations(search_results)
                sources_gathered = [item for citation in citations for item in citation["segments"]]
            else:
                # Fallback to knowledge-based response if no search results
                llm = configurable.get_llm_client(
                    model_name=configurable.query_generator_model,
                    temperature=0,
                    max_retries=2
                )
                response = llm.invoke(f"{formatted_prompt}\n\nNote: No web search results available. Provide response based on available knowledge.")
                modified_text = response.content
                sources_gathered = []
                citations = []
                
        except Exception as e:
            print(f"Firecrawl search failed: {str(e)}")
            # Fallback to knowledge-based response
            llm = configurable.get_llm_client(
                model_name=configurable.query_generator_model,
                temperature=0,
                max_retries=2
            )
            response = llm.invoke(f"{formatted_prompt}\n\nNote: Web search unavailable. Provide response based on available knowledge.")
            modified_text = response.content
            sources_gathered = []
            citations = []
            
    elif configurable.search_tool == "brave":
        # Use Brave Search API for web search
        try:
            # Create Brave search tool
            search_tool = create_search_tool(
                "brave",
                api_key=configurable.brave_api_key or os.getenv("BRAVE_API_KEY"),
                base_url=configurable.brave_base_url
            )
            
            if not search_tool:
                raise ValueError("Failed to create Brave search tool")
            
            # Perform search
            search_results = search_tool.search_and_scrape(
                query=state["search_query"],
                max_results=configurable.max_search_results,
                max_content_length=configurable.max_content_length
            )
            
            if search_results:
                # Format results for LLM
                search_content = search_tool.format_search_results(search_results)
                
                # Use LLM to synthesize research based on search content
                llm = configurable.get_llm_client(
                    model_name=configurable.query_generator_model,
                    temperature=0,
                    max_retries=2
                )
                
                research_prompt = f"""
                {formatted_prompt}
                
                Based on the following web search results from Brave Search, provide a comprehensive research response:
                
                {search_content}
                
                Please synthesize the information from these sources and provide insights relevant to the research topic.
                Reference the sources using [1], [2], etc. format where appropriate.
                """
                
                response = llm.invoke(research_prompt)
                modified_text = response.content
                
                # Create citations compatible with existing system
                citations = search_tool.create_citations(search_results)
                sources_gathered = [item for citation in citations for item in citation["segments"]]
            else:
                # Fallback to knowledge-based response if no search results
                llm = configurable.get_llm_client(
                    model_name=configurable.query_generator_model,
                    temperature=0,
                    max_retries=2
                )
                response = llm.invoke(f"{formatted_prompt}\n\nNote: No web search results available. Provide response based on available knowledge.")
                modified_text = response.content
                sources_gathered = []
                citations = []
                
        except Exception as e:
            print(f"Brave search failed: {str(e)}")
            # Fallback to knowledge-based response
            llm = configurable.get_llm_client(
                model_name=configurable.query_generator_model,
                temperature=0,
                max_retries=2
            )
            response = llm.invoke(f"{formatted_prompt}\n\nNote: Web search unavailable. Provide response based on available knowledge.")
            modified_text = response.content
            sources_gathered = []
            citations = []
    else:
        # For cases where no search tool is configured or available
        llm = configurable.get_llm_client(
            model_name=configurable.query_generator_model,
            temperature=0,
            max_retries=2
        )
        
        fallback_prompt = f"""
        {formatted_prompt}
        
        Note: Provide a comprehensive research response based on your knowledge. 
        Since web search is not available, use your training data to answer as thoroughly as possible.
        """
        
        response = llm.invoke(fallback_prompt)
        modified_text = response.content
        sources_gathered = []
        citations = []

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = configurable.get_llm_client(
        model_name=reasoning_model,
        temperature=1.0,
        max_retries=2
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model
    llm = configurable.get_llm_client(
        model_name=reasoning_model,
        temperature=0,
        max_retries=2
    )
    result = llm.invoke(formatted_prompt)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
