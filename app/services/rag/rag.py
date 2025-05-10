import os
import logging
import time
import json
import random
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from functools import lru_cache
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# LangChain imports
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document, HumanMessage, SystemMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, FunctionMessage, BaseMessage
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolExecutor, ToolInvocation

# Redis cache (optional, if available)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Telemetry and monitoring
try:
    import prometheus_client as prom
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_agent")

# Load environment variables
load_dotenv()

# API Keys and Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please check your .env file.")

# Optional environment variables with defaults
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")
DOCS_PATH = os.getenv("DOCS_PATH", "./docs")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSIONS_DB_PATH = os.getenv("SESSIONS_DB_PATH", "./sessions.db")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Set up telemetry if available
if TELEMETRY_AVAILABLE:
    # Prometheus metrics
    QUERY_LATENCY = prom.Summary('rag_query_latency', 'Time spent processing RAG queries')
    EMBEDDING_LATENCY = prom.Summary('embedding_latency', 'Time spent generating embeddings')
    RETRIEVAL_LATENCY = prom.Summary('retrieval_latency', 'Time spent retrieving documents')
    LLM_LATENCY = prom.Summary('llm_latency', 'Time spent on LLM calls')
    FAILED_QUERIES = prom.Counter('failed_queries', 'Number of failed queries')
    TOTAL_QUERIES = prom.Counter('total_queries', 'Total number of queries')
    TOOL_CALLS = prom.Counter('tool_calls', 'Number of tool calls', ['tool_name'])
    CACHE_HITS = prom.Counter('cache_hits', 'Number of cache hits')
    CACHE_MISSES = prom.Counter('cache_misses', 'Number of cache misses')
    
    # OpenTelemetry
    resource = Resource(attributes={"service.name": "rag_agent"})
    trace.set_tracer_provider(TracerProvider(resource=resource))
    otlp_exporter = OTLPSpanExporter(endpoint=os.getenv("OTLP_ENDPOINT", "localhost:4317"))
    span_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    tracer = trace.get_tracer(__name__)

# Initialize Redis cache if available
if REDIS_AVAILABLE:
    try:
        redis_client = redis.Redis.from_url(REDIS_URL)
        redis_client.ping()  # Test connection
        logger.info("Redis cache initialized successfully")
    except Exception as e:
        logger.warning(f"Redis cache initialization failed: {e}")
        redis_client = None
else:
    redis_client = None

# Custom exceptions for better error handling
class DocumentLoadError(Exception):
    """Raised when document loading fails"""
    pass

class EmbeddingError(Exception):
    """Raised when embedding generation fails"""
    pass

class RetrievalError(Exception):
    """Raised when document retrieval fails"""
    pass

class LLMError(Exception):
    """Raised when LLM generation fails"""
    pass

# Helper functions
def generate_cache_key(text: str) -> str:
    """Generate a deterministic cache key for a text string."""
    return hashlib.md5(text.encode()).hexdigest()

@lru_cache(maxsize=1000)
def cached_embed(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """In-memory LRU cache for embeddings."""
    # This function is wrapped with lru_cache for memory caching
    # It will be called directly if Redis is not available
    start_time = time.time()
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=model, api_key=GOOGLE_API_KEY)
        result = embeddings.embed_query(text)
        if TELEMETRY_AVAILABLE:
            EMBEDDING_LATENCY.observe(time.time() - start_time)
        return result
    except Exception as e:
        if TELEMETRY_AVAILABLE:
            FAILED_QUERIES.inc()
        logger.error(f"Embedding generation failed: {e}")
        raise EmbeddingError(f"Failed to generate embeddings: {e}")

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """Get embeddings with multi-level caching strategy."""
    cache_key = f"emb:{model}:{generate_cache_key(text)}"
    
    # Try Redis cache first if available
    if redis_client:
        cached = redis_client.get(cache_key)
        if cached:
            if TELEMETRY_AVAILABLE:
                CACHE_HITS.inc()
            return json.loads(cached)
        
        if TELEMETRY_AVAILABLE:
            CACHE_MISSES.inc()
            
        # Generate embedding and store in Redis
        result = cached_embed(text, model)
        redis_client.setex(cache_key, CACHE_TTL, json.dumps(result))
        return result
    else:
        # Fall back to in-memory LRU cache
        return cached_embed(text, model)

# Enhanced document splitting with metadata preservation
def process_documents(file_paths: List[str] = None, directory: str = None) -> List[Document]:
    """Process documents from either a list of files or a directory."""
    start_time = time.time()
    documents = []
    
    try:
        if file_paths:
            for file_path in file_paths:
                if os.path.exists(file_path):
                    loader = TextLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = file_path
                        doc.metadata["filename"] = os.path.basename(file_path)
                        doc.metadata["load_time"] = datetime.now().isoformat()
                    documents.extend(docs)
                else:
                    logger.warning(f"File not found: {file_path}")
        
        if directory:
            if os.path.exists(directory) and os.path.isdir(directory):
                loader = DirectoryLoader(directory, glob="**/*.txt")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = doc.metadata.get("source", "unknown")
                    doc.metadata["filename"] = os.path.basename(doc.metadata.get("source", "unknown"))
                    doc.metadata["load_time"] = datetime.now().isoformat()
                documents.extend(docs)
            else:
                logger.warning(f"Directory not found: {directory}")
        
        if not documents:
            raise DocumentLoadError("No documents were loaded")
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Process in parallel for large document sets
        if len(documents) > 10:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Process documents in batches to avoid memory issues
                batch_size = max(1, len(documents) // MAX_WORKERS)
                batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
                
                # Process each batch in parallel
                results = list(executor.map(
                    lambda batch: text_splitter.split_documents(batch),
                    batches
                ))
                
                # Combine results
                splits = [split for batch_result in results for split in batch_result]
        else:
            splits = text_splitter.split_documents(documents)
        
        logger.info(f"Processed {len(documents)} documents into {len(splits)} chunks in {time.time() - start_time:.2f}s")
        return splits
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise DocumentLoadError(f"Failed to process documents: {e}")

# Cache decorator for vector store queries
def cached_query(func):
    """Decorator to cache vector store query results."""
    def wrapper(query, *args, **kwargs):
        cache_key = f"query:{generate_cache_key(query)}"
        
        # Try Redis cache first if available
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                if TELEMETRY_AVAILABLE:
                    CACHE_HITS.inc()
                return json.loads(cached)
            
            if TELEMETRY_AVAILABLE:
                CACHE_MISSES.inc()
                
            # Execute query and store in Redis
            result = func(query, *args, **kwargs)
            serializable_result = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in result
            ]
            redis_client.setex(cache_key, CACHE_TTL, json.dumps(serializable_result))
            return result
        else:
            # No caching available, just execute
            return func(query, *args, **kwargs)
    
    return wrapper

# Initialize vector store with custom embedding function
def init_vector_store(documents: List[Document]) -> Chroma:
    """Initialize vector store with the provided documents."""
    start_time = time.time()
    
    # Custom embedding function that uses our caching implementation
    def embedding_function(texts: List[str]) -> List[List[float]]:
        """Custom embedding function with caching and parallelization."""
        # For small batches, process serially to avoid overhead
        if len(texts) <= 4:
            return [get_embedding(text) for text in texts]
        
        # For larger batches, use parallel processing
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            embeddings = list(executor.map(get_embedding, texts))
        
        return embeddings
    
    try:
        # Create persistent vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory=VECTOR_DB_PATH
        )
        
        logger.info(f"Vector store initialized with {len(documents)} documents in {time.time() - start_time:.2f}s")
        return vectorstore
    except Exception as e:
        logger.error(f"Vector store initialization failed: {e}")
        raise EmbeddingError(f"Failed to initialize vector store: {e}")

# Advanced retrieval function with prefiltering and re-ranking
@cached_query
def enhanced_retrieval(query: str, vectorstore: Chroma, k: int = 5, filter_metadata: Dict = None, 
                      fetch_k: int = 20) -> List[Document]:
    """
    Enhanced retrieval with prefiltering, contextual re-ranking, and hybrid search.
    
    Args:
        query: The query string
        vectorstore: The vector store to query
        k: Number of results to return
        filter_metadata: Optional metadata filters
        fetch_k: Larger number of initial results to fetch for re-ranking
        
    Returns:
        List of retrieved documents
    """
    start_time = time.time()
    
    try:
        # Extract potential filters from query using simple keyword matching
        # In production, use a more sophisticated NLP approach
        filter_keywords = {
            "date": ["today", "yesterday", "last week", "recent", "latest", "new"],
            "source": ["document", "file", "book", "article", "report", "paper"]
        }
        
        # Initialize dynamic filters
        dynamic_filters = {}
        
        # Apply user-specified filters if provided
        if filter_metadata:
            dynamic_filters.update(filter_metadata)
            
        # Apply semantic search with filters
        results = vectorstore.similarity_search_with_score(
            query=query,
            k=fetch_k,  # Fetch more results initially for re-ranking
            filter=dynamic_filters
        )
        
        # Re-rank results using a simple relevance score
        # In production, use a more sophisticated re-ranking model
        if results:
            # Add distance score to document metadata
            docs_with_scores = []
            for doc, score in results:
                # Convert distance to similarity (1 - distance)
                similarity = 1 - score
                doc.metadata["similarity_score"] = similarity
                docs_with_scores.append((doc, similarity))
            
            # Sort by similarity score (higher is better)
            sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)
            
            # Take top k results
            final_docs = [doc for doc, _ in sorted_docs[:k]]
        else:
            final_docs = []
            
        if TELEMETRY_AVAILABLE:
            RETRIEVAL_LATENCY.observe(time.time() - start_time)
            
        logger.info(f"Retrieved {len(final_docs)} documents in {time.time() - start_time:.2f}s")
        return final_docs
        
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        if TELEMETRY_AVAILABLE:
            FAILED_QUERIES.inc()
        raise RetrievalError(f"Failed to retrieve documents: {e}")

# Create an advanced retriever tool with broader capabilities
def create_enhanced_retriever_tool(vectorstore: Chroma):
    """Create an advanced retriever tool with filtering capabilities."""
    
    def _retriever(query: str, metadata_filters: Dict = None, k: int = 5):
        """
        Enhanced retriever function that supports metadata filtering.
        
        Args:
            query: The search query
            metadata_filters: Optional metadata filters to apply
            k: Number of results to return
            
        Returns:
            Retrieval results as formatted text
        """
        if TELEMETRY_AVAILABLE:
            TOOL_CALLS.labels(tool_name="retriever").inc()
            
        # Log the retrieval attempt
        logger.info(f"Retrieval request: query='{query}', filters={metadata_filters}")
        
        # Retrieve documents with filtering
        docs = enhanced_retrieval(
            query=query,
            vectorstore=vectorstore,
            k=k,
            filter_metadata=metadata_filters
        )
        
        if not docs:
            return "No relevant documents found for this query."
        
        # Format results
        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            similarity = doc.metadata.get("similarity_score", 0)
            content = doc.page_content.strip()
            
            # Format each document with metadata
            formatted_doc = f"[Document {i+1}] (Relevance: {similarity:.2f}, Source: {source})\n{content}\n"
            formatted_docs.append(formatted_doc)
            
        return "\n\n".join(formatted_docs)
        
    # Create a structured tool
    return {
        "name": "retrieve_documents",
        "description": """
            Retrieves documents relevant to a query. Use this tool when you need to find specific information
            in the knowledge base, particularly for facts, references, or detailed information.
            
            You can optionally provide metadata filters to narrow down results.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Make it specific and focused for best results."
                },
                "metadata_filters": {
                    "type": "object",
                    "description": "Optional metadata filters to apply (e.g., {'source': 'specific_file.txt'})"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)"
                }
            },
            "required": ["query"]
        },
        "function": _retriever
    }

# Create a tool for question decomposition
def create_decompose_question_tool():
    """Create a tool for decomposing complex questions."""
    
    def _decompose_question(question: str):
        """
        Decompose a complex question into simpler sub-questions.
        
        Args:
            question: The complex question to decompose
            
        Returns:
            List of sub-questions
        """
        if TELEMETRY_AVAILABLE:
            TOOL_CALLS.labels(tool_name="decompose_question").inc()
            
        # Create an LLM instance for decomposition
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, 
            api_key=GOOGLE_API_KEY,
            temperature=0.2  # Lower temperature for more deterministic results
        )
        
        # Prompt for question decomposition
        decomposition_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert at breaking down complex questions into simpler sub-questions.
            Given a complex question, identify the atomic sub-questions that need to be answered first.
            Return ONLY the list of sub-questions, without any explanation or preamble.
            Format as a numbered list.
            """),
            ("human", "{question}")
        ])
        
        # Generate decomposition
        try:
            start_time = time.time()
            chain = decomposition_prompt | llm
            result = chain.invoke({"question": question})
            
            if TELEMETRY_AVAILABLE:
                LLM_LATENCY.observe(time.time() - start_time)
                
            return result.content
            
        except Exception as e:
            logger.error(f"Question decomposition failed: {e}")
            if TELEMETRY_AVAILABLE:
                FAILED_QUERIES.inc()
            return "Failed to decompose the question. Please try with a simpler query."
    
    # Create a structured tool
    return {
        "name": "decompose_question",
        "description": """
            Decomposes a complex question into simpler sub-questions. Use this when faced with a 
            multi-part or complex question that requires breaking down before answering.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The complex question to decompose"
                }
            },
            "required": ["question"]
        },
        "function": _decompose_question
    }

# Create a summarization tool
def create_summarization_tool():
    """Create a tool for summarizing retrieved content."""
    
    def _summarize_content(content: str, focus: str = None):
        """
        Summarize content with optional focus.
        
        Args:
            content: The content to summarize
            focus: Optional focus area for the summary
            
        Returns:
            Summarized content
        """
        if TELEMETRY_AVAILABLE:
            TOOL_CALLS.labels(tool_name="summarize").inc()
            
        # Create an LLM instance for summarization
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, 
            api_key=GOOGLE_API_KEY,
            temperature=0.3
        )
        
        # Prompt for summarization
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert summarizer. Create a concise, informative summary of the provided content.
            Highlight the key points and important details. If a specific focus is provided, emphasize
            that aspect in your summary.
            """),
            ("human", """
            Please summarize the following content:
            
            {content}
            
            {focus_instruction}
            """)
        ])
        
        focus_instruction = f"Focus on aspects related to: {focus}" if focus else ""
        
        # Generate summary
        try:
            start_time = time.time()
            chain = summary_prompt | llm
            result = chain.invoke({
                "content": content,
                "focus_instruction": focus_instruction
            })
            
            if TELEMETRY_AVAILABLE:
                LLM_LATENCY.observe(time.time() - start_time)
                
            return result.content
            
        except Exception as e:
            logger.error(f"Content summarization failed: {e}")
            if TELEMETRY_AVAILABLE:
                FAILED_QUERIES.inc()
            return "Failed to summarize the content. The content might be too long or complex."
    
    # Create a structured tool
    return {
        "name": "summarize",
        "description": """
            Summarizes provided content. Use this when you have retrieved multiple documents
            and need to condense them into a concise summary before formulating your final answer.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to summarize"
                },
                "focus": {
                    "type": "string",
                    "description": "Optional focus area for the summary"
                }
            },
            "required": ["content"]
        },
        "function": _summarize_content
    }

# Create a fact-checking tool
def create_fact_checking_tool(vectorstore: Chroma):
    """Create a tool for fact-checking claims against the knowledge base."""
    
    def _fact_check(claim: str):
        """
        Check a claim against the knowledge base.
        
        Args:
            claim: The claim to verify
            
        Returns:
            Fact-checking result
        """
        if TELEMETRY_AVAILABLE:
            TOOL_CALLS.labels(tool_name="fact_check").inc()
            
        # Retrieve relevant documents
        docs = enhanced_retrieval(
            query=claim,
            vectorstore=vectorstore,
            k=3
        )
        
        if not docs:
            return "No relevant information found to verify this claim."
        
        # Create an LLM instance for fact checking
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, 
            api_key=GOOGLE_API_KEY,
            temperature=0.2  # Lower temperature for more deterministic results
        )
        
        # Prepare context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Prompt for fact checking
        fact_check_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert fact-checker. Given a claim and supporting context,
            determine whether the claim is:
            
            1. SUPPORTED - The context clearly supports the claim
            2. REFUTED - The context clearly contradicts the claim
            3. INSUFFICIENT EVIDENCE - The context doesn't provide enough information
            
            Provide your verdict and a brief explanation based ONLY on the provided context.
            """),
            ("human", """
            Claim: {claim}
            
            Context:
            {context}
            
            Verdict (SUPPORTED/REFUTED/INSUFFICIENT EVIDENCE):
            """)
        ])
        
        # Generate fact check
        try:
            start_time = time.time()
            chain = fact_check_prompt | llm
            result = chain.invoke({
                "claim": claim,
                "context": context
            })
            
            if TELEMETRY_AVAILABLE:
                LLM_LATENCY.observe(time.time() - start_time)
                
            return result.content
            
        except Exception as e:
            logger.error(f"Fact checking failed: {e}")
            if TELEMETRY_AVAILABLE:
                FAILED_QUERIES.inc()
            return "Failed to verify the claim due to an error in processing."
    
    # Create a structured tool
    return {
        "name": "fact_check",
        "description": """
            Verifies a claim against the knowledge base. Use this when you need to
            verify the accuracy of specific information before including it in your answer.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "The claim to verify"
                }
            },
            "required": ["claim"]
        },
        "function": _fact_check
    }

# Build the ReAct Agent with advanced tools
def build_rag_agent(vectorstore: Chroma):
    """Build a full ReAct RAG agent with multiple tools."""
    
    # Create LLM with streaming capability
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        api_key=GOOGLE_API_KEY,
        temperature=TEMPERATURE,
        callbacks=[StreamingStdOutCallbackHandler()],
        streaming=True
    )
    
    # Create the tools
    retriever_tool = create_enhanced_retriever_tool(vectorstore)
    decompose_tool = create_decompose_question_tool()
    summarize_tool = create_summarization_tool()
    fact_check_tool = create_fact_checking_tool(vectorstore)
    
    tools = [retriever_tool, decompose_tool, summarize_tool, fact_check_tool]
    
    # Initialize tool executor
    tool_executor = ToolExecutor(tools)
    
    # Create ReAct agent prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an intelligent agent that uses specialized tools to answer questions optimally.
        
        Follow these guidelines:
        1. For complex questions, consider using decompose_question to break them down
        2. For factual information, use retrieve_documents to search the knowledge base
        3. When you have multiple retrieved documents, use summarize to condense them
        4. For uncertain claims, use fact_check to verify them against the knowledge base
        
        Always approach questions methodically:
        - Consider what information you need to answer the question
        - Use appropriate tools to gather that information
        - Synthesize a clear, accurate answer
        
        Your responses should be:
        - Conversational and engaging
        - Factually accurate (based on retrieved information)
        - Well-organized and easy to understand
        
        Respond only with information that is supported by the retrieved documents
        or general knowledge. If you're unsure, acknowledge uncertainty rather than guessing.
        """),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the ReAct agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Return the fully configured agent with metadata
    return {
        "agent": agent, 
        "tool_executor": tool_executor, 
        "tools": tools,
        "metadata": {
            "model": LLM_MODEL,
            "temperature": TEMPERATURE,
            "tool_count": len(tools),
            "created_at": datetime.now().isoformat()
        }
    }

# Define a stateful graph with checkpointing
def create_agent_graph(vectorstore: Chroma):
    """Create a stateful agent graph with checkpointing."""
    
    # Build the RAG agent components
    agent_components = build_rag_agent(vectorstore)
    agent = agent_components["agent"]
    tools = agent_components["tools"]
    tool_executor = agent_components["tool_executor"]
    
    # Set up checkpointer for persistence
    checkpointer = SqliteSaver(SESSIONS_DB_PATH)
    
    # Define the agent state
    class AgentState(TypedDict):
        messages: List[BaseMessage]
        agent_scratchpad: List[BaseMessage]
        context_docs: Optional[List[Document]]
        agent_action: Optional[Any]
        error: Optional[str]
        
    # Define the agent nodes
    def agent_node(state: AgentState):
        """Agent node that processes messages and decides actions."""
        if TELEMETRY_AVAILABLE:
            with tracer.start_as_current_span("agent_node"):
                current_span = trace.get_current_span()
                current_span.set_attribute("messages_count", len(state["messages"]))
        
        # Extract the last message if it exists
        messages = state["messages"]
        
        try:
            # Get the next action from the agent
            agent_response = agent.invoke({
                "messages": messages,
                "agent_scratchpad": state["agent_scratchpad"]
            })
            
            # Return the action based on agent's response
            return {"agent_action": agent_response}
        except Exception as e:
            logger.error(f"Agent node failed: {e}")
            return {"agent_action": None, "error": str(e)}
            
    def tool_node(state: Dict[str, Any]):
        """Tool node that executes the selected tool."""
        if TELEMETRY_AVAILABLE:
            with tracer.start_as_current_span("tool_node"):
                pass
                
        # Extract the agent action
        agent_action = state["agent_action"]
        
        try:
            # Check if we have a valid agent action
            if not agent_action:
                # Return an error message if the agent action is missing
                error_message = state.get("error", "Unknown error in agent processing")
                error_response = AIMessage(content=f"I encountered an error: {error_message}. Please try rephrasing your question.")
                return {
                    "messages": state["messages"] + [error_response],
                    "agent_scratchpad": state["agent_scratchpad"]
                }
                
            # Check if the agent wants to respond directly (final answer)
            if agent_action.tool == "final_answer":
                final_answer = AIMessage(content=agent_action.tool_input)
                return {
                    "messages": state["messages"] + [final_answer],
                    "agent_scratchpad": state["agent_scratchpad"]
                }
            
            # Execute the tool
            start_time = time.time()
            tool_result = tool_executor.invoke(
                ToolInvocation(
                    tool=agent_action.tool,
                    tool_input=agent_action.tool_input
                )
            )
            if TELEMETRY_AVAILABLE:
                TOOL_CALLS.labels(tool_name=agent_action.tool).inc()
                
            logger.info(f"Tool '{agent_action.tool}' executed in {time.time() - start_time:.2f}s")
            
            # Create the tool message
            tool_message = FunctionMessage(
                name=agent_action.tool,
                content=str(tool_result)
            )
            
            # Update the agent scratchpad with the tool call and result
            new_scratchpad = state["agent_scratchpad"] + [
                AIMessage(content=str(agent_action)),
                tool_message
            ]
            
            return {
                "messages": state["messages"],
                "agent_scratchpad": new_scratchpad
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            if TELEMETRY_AVAILABLE:
                FAILED_QUERIES.inc()
            error_message = f"Tool execution failed: {str(e)}"
            error_response = AIMessage(content=f"I encountered an error: {error_message}. Please try rephrasing your question.")
            
            return {
                "messages": state["messages"] + [error_response],
                "agent_scratchpad": state["agent_scratchpad"]
            }
            
    # Function to determine next step based on agent output        
    def should_continue(state: Dict[str, Any]) -> str:
        """Determine whether to continue with tool execution or finish."""
        # Check if we have a final answer
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        
        # If the last message is an AI message, it means we've reached a final answer
        if isinstance(last_message, AIMessage):
            return "end"
            
        # If we have agent_action and it's a final_answer, we should end
        agent_action = state.get("agent_action")
        if agent_action and agent_action.tool == "final_answer":
            return "end"
            
        # Check if we've exceeded the maximum number of tool calls (prevent infinite loops)
        if len(state.get("agent_scratchpad", [])) > 20:  # Arbitrary limit to prevent infinite loops
            logger.warning("Maximum tool call limit reached, forcing end of execution")
            return "end"
            
        # Otherwise, continue with tool execution
        return "continue_tool"
        
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tool", tool_node)
    
    # Set up the edges
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "tool")
    workflow.add_conditional_edges(
        "tool",
        should_continue,
        {
            "continue_tool": "agent",
            "end": END
        }
    )
    
    # Compile the graph with checkpointing
    return workflow.compile(checkpointer=checkpointer)