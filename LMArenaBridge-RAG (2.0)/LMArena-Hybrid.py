# LMArena-Hybrid.py - v2.0
# Combines advanced compression with full conversation context return

import asyncio
import json
import logging
import httpx
import uuid_utils as uuid
import re
import time
import random
from contextlib import asynccontextmanager
from typing import Dict, Optional, List, Tuple
import uuid as uuid_v4

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from starlette.responses import StreamingResponse, JSONResponse, Response
import tiktoken
import hashlib
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# --- STEALTH CONFIGURATION ---
STEALTH_CONFIG = {
    "enable_stealth": True,
    "tab_rotation_enabled": True,
    "max_requests_per_tab": 999,  # Effectively disabled - only rotate on 429
    "request_spacing": (2.0, 5.0),  # Increased from (0.3, 1.0) for more human-like behavior
}

# --- Tab Connection Manager ---
class TabConnectionManager:
    """Manage multiple browser tab connections"""
    
    def __init__(self):
        self.tabs = {}
        self.active_ws = None
        
    def register_tab(self, tab_id: str, websocket: WebSocket, request_count: int = 0):
        """Register a new tab connection"""
        self.tabs[tab_id] = {
            "ws": websocket,
            "request_count": request_count,
            "last_activity": time.time()
        }
        self.active_ws = websocket
        logging.info(f"📋 Registered tab {tab_id} (requests: {request_count})")
        
    def unregister_tab(self, websocket: WebSocket):
        """Remove a tab when it disconnects"""
        to_remove = None
        for tab_id, info in self.tabs.items():
            if info["ws"] == websocket:
                to_remove = tab_id
                break
        
        if to_remove:
            del self.tabs[to_remove]
            logging.info(f"📋 Unregistered tab {to_remove}")
            
            if self.tabs:
                self.active_ws = list(self.tabs.values())[0]["ws"]
            else:
                self.active_ws = None
    
    def get_active_websocket(self) -> Optional[WebSocket]:
        """Get the currently active websocket"""
        return self.active_ws
    
    def cleanup_old_tabs(self, max_age_seconds: int = 300):
        """Remove tabs that haven't been active recently"""
        now = time.time()
        to_remove = []
        
        for tab_id, info in self.tabs.items():
            if now - info["last_activity"] > max_age_seconds:
                to_remove.append(tab_id)
        
        for tab_id in to_remove:
            del self.tabs[tab_id]
            logging.info(f"🧹 Cleaned up old tab {tab_id}")

tab_manager = TabConnectionManager()

# --- Token Counting Functions ---
def estimate_tokens(text: str, model_name: str = "gpt-4") -> int:
    """Estimate token count for text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))

def count_message_tokens(messages: list, model_name: str = "gpt-4") -> int:
    """Count total tokens in message list"""
    total = 0
    for msg in messages:
        content = str(msg.get("content", ""))
        total += estimate_tokens(content, model_name) + 4
    return total

# --- Session Manager ---
class SessionManager:
    """Track and manage stateful sessions with LMArena"""
    
    def __init__(self):
        self.sessions = {}
        self.message_history = {}
        self.compressed_history = {}
    
    def get_or_create_session(self, conversation_id: str, messages: list) -> tuple:
        """
        Returns: (evaluation_id, messages_to_send, is_new_session)
        """
        
        if conversation_id in self.sessions:
            eval_id, last_count = self.sessions[conversation_id]
            stored_messages = self.message_history[conversation_id]
            
            current_count = len(messages)
            
            if current_count > last_count:
                new_messages = messages[last_count:]
                
                logging.info(f"📨 SESSION REUSE: {eval_id[:8]}... | "
                            f"Sending {len(new_messages)} new messages "
                            f"({current_count} total server-side)")
                
                self.sessions[conversation_id] = (eval_id, current_count)
                self.message_history[conversation_id] = messages.copy()
                
                return eval_id, new_messages, False
            
            elif current_count == last_count:
                logging.info(f"🔄 SESSION REGENERATE: {eval_id[:8]}... | "
                            f"Re-sending last message")
                return eval_id, messages[-1:], False
            
            else:
                logging.warning(f"⚠️ SESSION RESET: Message count decreased "
                               f"({last_count} → {current_count})")
        
        eval_id = str(uuid.uuid7())
        self.sessions[conversation_id] = (eval_id, len(messages))
        self.message_history[conversation_id] = messages.copy()
        
        logging.info(f"🆕 NEW SESSION: {eval_id[:8]}... | "
                    f"Sending {len(messages)} messages")
        
        return eval_id, messages, True
    
    def store_compressed_context(self, conversation_id: str, compressed_messages: list):
        """Store the compressed message history for this conversation"""
        self.compressed_history[conversation_id] = compressed_messages.copy()
        logging.info(f"💾 Stored {len(compressed_messages)} compressed messages for conversation {conversation_id[:8]}...")
    
    def get_compressed_context(self, conversation_id: str) -> list:
        """Retrieve compressed context for response building"""
        return self.compressed_history.get(conversation_id, [])

session_manager = SessionManager()

# --- Thinking Tag Stripper ---
def strip_thinking_tags(messages: list) -> tuple:
    """
    Remove <think>...</think> / <thinking>...</thinking> blocks from messages.
    Returns (cleaned_messages, tokens_saved).
    """
    thinking_pattern = re.compile(
        r'<\s*think(?:ing)?(?:\s+[^>]*)?\s*>.*?<\s*/\s*think(?:ing)?\s*>',
        re.DOTALL | re.IGNORECASE,
    )

    cleaned: list = []
    tokens_before = 0
    tokens_after = 0
    total_blocks_removed = 0

    for i, msg in enumerate(messages):
        content = str(msg.get("content", ""))
        tokens_before_msg = estimate_tokens(content, "gpt-4")
        tokens_before += tokens_before_msg

        cleaned_content, blocks_found = thinking_pattern.subn("", content)
        total_blocks_removed += blocks_found

        if blocks_found > 0:
            tokens_after_msg = estimate_tokens(cleaned_content, "gpt-4")
        else:
            tokens_after_msg = tokens_before_msg

        tokens_after += tokens_after_msg

        if blocks_found > 0:
            msg_tokens_saved = tokens_before_msg - tokens_after_msg
            logging.debug(
                f"[think-strip] msg={i}: "
                f"removed {blocks_found} blocks, saved {msg_tokens_saved} tokens"
            )

        new_msg = msg.copy()
        new_msg["content"] = cleaned_content
        cleaned.append(new_msg)

    tokens_saved = tokens_before - tokens_after
    if tokens_saved > 0:
        reduction_pct = (tokens_saved / max(tokens_before, 1)) * 100
        logging.info(
            f"🧠 Stripped {total_blocks_removed} thinking blocks: "
            f"saved {tokens_saved:,} tokens ({reduction_pct:.1f}% reduction)"
        )

    return cleaned, tokens_saved

# --- RAG + RECOMP Setup ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(allow_reset=False),
)

def rag_embed(text: str) -> List[float]:
    """Wrap embedding_model.encode into a JSON-serializable list."""
    return embedding_model.encode(text).tolist()

def store_message_rag(
    conversation_id: str,
    message_id: str,
    role: str,
    content: str,
    character_id: str | None = None,
    turn_index: int | None = None,
    extra_meta: Dict | None = None,
) -> None:
    if not content or not content.strip():
        return

    embedding = rag_embed(content)

    raw_metadata: Dict = {
        "conversation_id": conversation_id,
        "role": role,
        "timestamp": float(time.time()),
        "turn_index": int(turn_index) if turn_index is not None else None,
        "character_id": character_id,
    }
    if extra_meta:
        raw_metadata.update(extra_meta)

    metadata: Dict = {}
    for k, v in raw_metadata.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            metadata[k] = v

    history_collection.add(
        ids=[message_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[metadata],
    )

def retrieve_relevant_context(
    conversation_id: str,
    query: str,
    character_id: str | None = None,
    n_results: int = 8,
    max_chars_per_doc: int = 500,
) -> Tuple[List[str], List[Dict]]:
    """
    Retrieve top-k relevant past messages for this conversation.
    Returns (documents, metadatas).
    """
    if not query or not query.strip():
        return [], []

    query_embedding = rag_embed(query)

    where_filters: Dict = {"conversation_id": conversation_id}
    if character_id is not None:
        where_filters["character_id"] = character_id

    results = history_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filters,
    )

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]

    cleaned_docs: List[str] = []
    cleaned_metas: List[Dict] = []
    for d, m in zip(docs, metas):
        if not d:
            continue
        if max_chars_per_doc and len(d) > max_chars_per_doc:
            d = d[:max_chars_per_doc]
        cleaned_docs.append(d)
        cleaned_metas.append(m)

    return cleaned_docs, cleaned_metas

history_collection = chroma_client.get_or_create_collection(
    name="chat_history",
    metadata={"hnsw:space": "cosine"},
)

def select_rag_chunks_with_token_budget(
    retrieved_docs: List[str],
    max_rag_tokens: int,
    model_name: str = "gpt-4",
) -> List[str]:
    """
    Greedily select retrieved docs until the RAG token budget is hit.
    """
    selected: List[str] = []
    running_text = ""

    for doc in retrieved_docs:
        doc = doc.strip()
        if not doc:
            continue

        candidate = f"{running_text}\n\n{doc}" if running_text else doc
        if estimate_tokens(candidate, model_name=model_name) > max_rag_tokens:
            break

        selected.append(doc)
        running_text = candidate

    return selected

# --- RECOMP Compression ---
def compress_messages_recomp(messages, retention=0.8):
    """Real RECOMP with semantic embeddings."""
    compressed = []
    
    for msg in messages:
        content = str(msg.get("content", ""))
        sentences = [s.strip() for s in content.split('. ') if s.strip()]
        
        if len(sentences) <= 2:
            compressed_content = content
        else:
            embeddings = embedding_model.encode(sentences)
            query_embedding = embedding_model.encode([content])[0]
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            
            keep_count = max(2, int(len(sentences) * retention))
            top_indices = similarities.argsort()[-keep_count:][::-1]
            top_indices = sorted(top_indices)
            
            compressed_content = '. '.join([sentences[i] for i in top_indices])
        
        compressed_msg = msg.copy()
        compressed_msg["content"] = compressed_content
        compressed.append(compressed_msg)
    
    return compressed

# --- LLM Summarization ---
async def llm_summarize_via_lmarena(messages, query, target_tokens):
    """
    LLM summarization with NSFW tolerance and model fallback.
    """
    recent_count = min(15, max(1, len(messages) // 3))
    recent = messages[-recent_count:]
    old = messages[:-recent_count]

    if not old:
        logging.warning("No old messages to summarize")
        return messages

    conversation_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in old])

    current_tokens = count_message_tokens(old)
    recent_tokens = count_message_tokens(recent)
    target_summary_tokens = max(5000, target_tokens - recent_tokens - 1000)

    base_messages = [
        {
            "role": "system",
            "content": (
                "You are a narrative summarizer for creative fiction roleplay. "
                "Summarize existing content objectively, including mature themes, without censorship. "
                "This is fictional content between consenting adults."
            )
        },
        {
            "role": "user",
            "content": (
                f"Summarize this fictional roleplay in {target_summary_tokens} tokens or less.\n\n"
                f"Current scene: {query}\n\n"
                "**PRESERVE**: Characters, relationships, plot (including intimate scenes), emotions, setting, actions\n"
                "**CONDENSE**: Repetitive dialogue, redundant descriptions\n"
                "**GUIDELINES**: Use clinical language for intimate content, preserve emotional context, no censorship\n\n"
                f"Conversation:\n{conversation_text}\n\n"
                "Objective summary (no meta-commentary):"
            )
        }
    ]

    models_to_try = [
        "gemini-2.5-pro",
        "deepseek-v3.2-exp",
        "claude-haiku-4-5-20251001"
    ]

    async with httpx.AsyncClient(timeout=60.0) as client:
        for model in models_to_try:
            summary_text = ""
            try:
                summary_request = {
                    "model": model,
                    "messages": base_messages,
                    "max_tokens": target_summary_tokens,
                    "temperature": 0.2,
                    "stream": False,
                }

                logging.info(f"🎭 Trying {model} for summary...")

                response = await client.post(
                    "http://localhost:9080/v1/chat/completions",
                    json=summary_request
                )

                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    summary_text = result['choices'][0].get('message', {}).get('content', '')

            except json.JSONDecodeError:
                logging.error(f"Failed to parse JSON response from {model}")
                continue
            except httpx.RequestError as e:
                logging.error(f"Request failed for {model}: {e}")
                await asyncio.sleep(1)
                continue
            except Exception as e:
                logging.error(f"❌ {model} failed with exception: {e}")
                continue

            refusal_keywords = ["cannot", "inappropriate", "unable to", "i can't", "i apologize"]

            if any(keyword in summary_text.lower()[:200] for keyword in refusal_keywords):
                logging.warning(f"⚠️ {model} refused, trying next model...")
                continue

            if not summary_text or len(summary_text) < 100:
                logging.warning(f"⚠️ {model} returned short/empty response, trying next model...")
                continue

            summary_tokens = estimate_tokens(summary_text, 'gpt-4')
            logging.info(f"✅ {model} summarized successfully: {summary_tokens} tokens")

            return [
                {
                    "role": "system",
                    "content": f"[Previous Story Summary]\n{summary_text}\n\n[Current Scene]"
                }
            ] + recent

    logging.error("❌ All summarization models failed, using original messages")
    return messages

async def recursive_recomp(messages, query, target_tokens, retention_rate=0.8, max_iterations=4):
    """
    Recursive RECOMP with 80% retention per pass.
    """
    
    current_tokens = count_message_tokens(messages)
    iteration = 0
    
    while current_tokens > target_tokens and iteration < max_iterations:
        iteration += 1
        
        compressed = compress_messages_recomp(
            messages,
            retention=retention_rate
        )
        
        new_tokens = count_message_tokens(compressed)
        reduction_pct = ((current_tokens - new_tokens) / current_tokens) * 100
        
        logging.info(f"  🔄 RECOMP Pass {iteration}: {current_tokens}→{new_tokens} "
                    f"(-{reduction_pct:.1f}%)")
        
        if new_tokens >= current_tokens * 0.95:
            logging.warning(f"  ⚠️ Minimal reduction, stopping")
            break
        
        messages = compressed
        current_tokens = new_tokens
        
        if current_tokens <= target_tokens:
            logging.info(f"  ✅ Target reached in {iteration} passes")
            break
    
    return messages

async def aggressive_compression_loop(messages, query, target=35000, max_loops=3):
    """
    Outer loop: Cycles RECOMP → LLM Summary until under limit.
    """
    
    tokens = count_message_tokens(messages)
    loop_count = 0
    
    logging.info(f"🎯 Target: {target} | Current: {tokens}")
    
    while tokens > target and loop_count < max_loops:
        loop_count += 1
        logging.info(f"\n{'='*60}")
        logging.info(f"🔁 LOOP {loop_count}/{max_loops}")
        logging.info(f"{'='*60}")
        
        logging.info(f"📊 Before RECOMP: {tokens}")
        messages = await recursive_recomp(messages, query, target, 0.8, 4)
        tokens = count_message_tokens(messages)
        logging.info(f"✅ After RECOMP: {tokens}")
        
        if tokens <= target:
            logging.info(f"🎉 Success in loop {loop_count}")
            break
        
        logging.info(f"⚠️ Still over, triggering LLM summary")
        messages = await llm_summarize_via_lmarena(messages, query, target)
        tokens = count_message_tokens(messages)
        logging.info(f"✅ After LLM: {tokens}")
        
        if tokens <= target:
            logging.info(f"🎉 Success in loop {loop_count}")
            break
    
    if tokens > target:
        logging.error(f"🚨 Failed after {loop_count} loops, truncating")
        keep = int(len(messages) * (target / tokens))
        messages = messages[-keep:]
        tokens = count_message_tokens(messages)
        logging.info(f"✂️ Truncated to {tokens}")
    else:
        logging.info(f"\n{'='*60}")
        logging.info(f"✅ COMPLETE: {tokens}/{target} ({tokens/target*100:.1f}%)")
        logging.info(f"Loops: {loop_count}/{max_loops}")
        logging.info(f"{'='*60}\n")
    
    return messages

# --- Global State ---
response_queues: Dict[str, asyncio.Queue] = {}
model_registry: Dict[str, Dict] = {}
background_tasks = set()
ws_lock = asyncio.Lock()

# --- FastAPI App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Server starting up...")
    logging.info(f"🛡️ Stealth mode: {'ENABLED' if STEALTH_CONFIG['enable_stealth'] else 'DISABLED'}")
    logging.info(f"🔄 Tab rotation: Smart rotation (only on rate limit)")
    logging.info(f"⏱️ Request spacing: {STEALTH_CONFIG['request_spacing'][0]}-{STEALTH_CONFIG['request_spacing'][1]}s")
    yield
    logging.info("Server shutting down...")
    for task in background_tasks:
        task.cancel()

app = FastAPI(lifespan=lifespan)

# --- WebSocket Handler with Tab Management ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    tab_id = None
    
    try:
        while True:
            message_str = await websocket.receive_text()
            try:
                message = json.loads(message_str)
            except json.JSONDecodeError:
                logging.warning(f"Malformed JSON: {message_str[:200]}")
                continue

            if message.get("type") == "reconnection_handshake":
                tab_id = message.get("tab_id")
                request_count = message.get("request_count", 0)
                
                async with ws_lock:
                    tab_manager.register_tab(tab_id, websocket, request_count)
                
                logging.info(f"✅ Tab {tab_id} connected (requests: {request_count})")
                continue

            if message.get("type") == "model_registry":
                global model_registry
                models_data = message.get("models", {})
                
                new_registry = {}
                for model_name, model_info in models_data.items():
                    processed_info = model_info.copy()
                    
                    capabilities = model_info.get("capabilities", {})
                    context_limit = None
                    
                    if isinstance(capabilities, dict):
                        context_limit = (
                            capabilities.get("maxContextTokens") or
                            capabilities.get("contextWindow") or
                            capabilities.get("maxInputTokens") or
                            capabilities.get("max_context_length")
                        )
                    
                    if context_limit:
                        processed_info["context_limit"] = int(context_limit)
                    else:
                        processed_info["context_limit"] = 128000
                    
                    new_registry[model_name] = processed_info
                
                model_registry = new_registry
                logging.info(f"Updated model registry with {len(model_registry)} models.")
                await websocket.send_text(json.dumps({"type": "model_registry_ack", "count": len(model_registry)}))
                continue

            request_id = message.get("request_id")
            data = message.get("data")
            
            if request_id in response_queues:
                queue = response_queues[request_id]
                await queue.put(data)
                if data == "[DONE]":
                    if request_id in response_queues:
                        del response_queues[request_id]
            else:
                logging.warning(f"Received message for unknown request_id: {request_id}")
                    
    except WebSocketDisconnect:
        logging.warning(f"❌ Browser tab disconnected: {tab_id}")
    finally:
        async with ws_lock:
            tab_manager.unregister_tab(websocket)
            tab_manager.cleanup_old_tabs()

async def create_lmarena_request_body(openai_req: dict, conversation_id: str, _is_summarization=False) -> Tuple[dict, list, list]:
    """
    Creates a request body for LMArena with dual token + character limits
    Returns: (payload, files_to_upload, compressed_messages_for_response)
    """
    
    files_to_upload = []
    processed_messages = []

    model_name = openai_req["model"]
    if model_name not in model_registry:
        raise ValueError(f"Model '{model_name}' not found in registry.")
    
    model_info = model_registry[model_name]
    model_id = model_info.get("id", model_name)
    modality = model_info.get("type", "chat")

    SUMMARIZATION_MODELS = ["gemini-2.5-pro", "deepseek-v3.2-exp", "claude-haiku-4-5-20251001"]
    is_summarization = model_name in SUMMARIZATION_MODELS
    
    context_limit_tokens = model_info.get("context_limit", 128000)
    max_completion_tokens = openai_req.get("max_tokens", 4096)
    if max_completion_tokens is None:
        max_completion_tokens = 4096
    
    MAX_PROMPT_TOKENS_MODEL = context_limit_tokens - max_completion_tokens - 500
    MAX_PAYLOAD_CHARS = 150000
    MAX_PROMPT_TOKENS_API = 31000
    
    MAX_PROMPT_TOKENS = min(MAX_PROMPT_TOKENS_MODEL, MAX_PROMPT_TOKENS_API)
    
    logging.info(f"Model: {model_name} | Model context: {context_limit_tokens} tokens | "
                f"API limit: {MAX_PROMPT_TOKENS_API} tokens | "
                f"Effective limit: {MAX_PROMPT_TOKENS} tokens")
    
    MIME_TYPE_MAP = {'image/jpeg': 'jpg', 'image/png': 'png', 'image/gif': 'gif', 'image/webp': 'webp'}
    
    for msg in openai_req.get('messages', []):
        content = msg.get("content", "")
        new_msg = msg.copy()
        
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url", {}).get("url", "")
                    match = re.match(r"data:(image/[a-zA-Z0-9.+-]+);base64,(.*)", image_url)
                    if match:
                        mime_type, base64_data = match.groups()
                        file_ext = MIME_TYPE_MAP.get(mime_type, 'bin')
                        filename = f"upload-{uuid.uuid7()}.{file_ext}"
                        files_to_upload.append({"fileName": filename, "contentType": mime_type, "data": base64_data})
            new_msg["content"] = "\n".join(text_parts)
        
        processed_messages.append(new_msg)
    
    if not processed_messages:
        raise ValueError("Cannot create a request with no messages.")

    system_prompt_msg = None
    conversation_turns = processed_messages
    if processed_messages and processed_messages[0].get("role") == "system":
        system_prompt_msg = processed_messages[0]
        conversation_turns = processed_messages[1:]

    system_prompt_content = system_prompt_msg.get("content", "") if system_prompt_msg else ""
    system_tokens = estimate_tokens(system_prompt_content, model_name) if system_prompt_content else 0

    total_tokens = system_tokens + sum(
        estimate_tokens(str(msg.get("content", "")), model_name) + 4
        for msg in conversation_turns
    )

    logging.info("=" * 80)
    logging.info(f"📊 CONTEXT ANALYSIS")
    logging.info(f"Current conversation: {total_tokens:,} tokens")
    logging.info(f"Effective limit: {MAX_PROMPT_TOKENS:,} tokens")
    logging.info(f"Status: {'✓ FITS' if total_tokens <= MAX_PROMPT_TOKENS else '⚠ EXCEEDS LIMIT'}")
    logging.info("=" * 80)

    if total_tokens <= MAX_PROMPT_TOKENS:
        logging.info(f"✓ Fits: {total_tokens}/{MAX_PROMPT_TOKENS} tokens")
        messages_to_include = processed_messages
    elif _is_summarization:
        logging.warning("⚠️ Summarization request over limit - truncating")
        messages_to_include = processed_messages[:MAX_PROMPT_TOKENS // 100]
    else:
        logging.info(f"⚠️ Compression needed: {total_tokens}/{MAX_PROMPT_TOKENS}")
        
        current_query = conversation_turns[-1].get('content', '') if conversation_turns else ''
        
        try:
            messages_to_include = await aggressive_compression_loop(
                processed_messages,
                current_query,
                target=MAX_PROMPT_TOKENS,
                max_loops=3
            )
        except Exception as e:
            logging.error(f"❌ Compression failed: {e}")
            logging.warning("⚠️ Falling back to hard truncation")
            
            keep_count = int(len(processed_messages) * (MAX_PROMPT_TOKENS / total_tokens))
            messages_to_include = processed_messages[-max(keep_count, 5):]

    session_manager.store_compressed_context(conversation_id, messages_to_include)

    character_id = openai_req.get("character_id")
    current_query = ""
    for msg in reversed(messages_to_include):
        if msg.get("role") != "assistant":
            current_query = str(msg.get("content", ""))
            break

    retrieved_docs, _ = retrieve_relevant_context(
        conversation_id=conversation_id,
        query=current_query,
        character_id=character_id,
        n_results=16,
        max_chars_per_doc=500,
    )

    B_RAG = 5000
    selected_docs = select_rag_chunks_with_token_budget(
        retrieved_docs,
        max_rag_tokens=B_RAG,
        model_name=model_name,
    )

    rag_context_block = ""
    if selected_docs:
        context_lines = [f"[Past Context] {doc}" for doc in selected_docs]
        rag_context_block = "[Relevant Past Context]\n" + "\n".join(context_lines)

    conversation_text = "\n\n".join([
        msg.get("content", "") for msg in messages_to_include
        if msg.get("role") != "assistant"
    ])

    blocks: List[str] = []
    if rag_context_block:
        blocks.append(rag_context_block)
    if conversation_text:
        blocks.append(conversation_text)

    final_user_content = "\n\n".join(blocks)

    MAX_USER_CONTENT_CHARS = 140_000
    if len(final_user_content) > MAX_USER_CONTENT_CHARS:
        logging.warning(f"[SAFEGUARD] Content too long, trimming to {MAX_USER_CONTENT_CHARS} chars")
        final_user_content = final_user_content[-MAX_USER_CONTENT_CHARS:]

    evaluation_id = str(uuid.uuid7())
    message_ids = {i: str(uuid.uuid7()) for i in range(len(messages_to_include))}
    last_user_message_id = None
    arena_messages = []

    for i, msg in enumerate(messages_to_include):
        role = "user" if msg.get("role") != "assistant" else "assistant"
        msg_id = message_ids[i]
        if role != 'assistant':
            last_user_message_id = msg_id
        arena_messages.append({
            "id": msg_id,
            "role": role,
            "content": msg.get('content', ''),
            "parentMessageIds": [message_ids[i-1]] if i > 0 else [],
            "modelId": model_id if role == 'assistant' else None,
            "evaluationSessionId": evaluation_id,
        })

    if not last_user_message_id:
        raise ValueError("No user messages in conversation.")

    final_user_message_payload = {
        "id": last_user_message_id,
        "content": final_user_content,
        "role": "user",
        "parentMessageIds": [msg["id"] for msg in arena_messages if msg["id"] != last_user_message_id],
        "modelId": None,
        "evaluationSessionId": evaluation_id,
    }

    model_a_message_id = str(uuid.uuid7())
    model_a_placeholder_message = {
        "id": model_a_message_id,
        "role": "assistant",
        "content": "",
        "parentMessageIds": [last_user_message_id],
        "modelId": model_id,
        "evaluationSessionId": evaluation_id,
    }
    arena_messages.append(model_a_placeholder_message)

    generation_params = {
        "temperature": openai_req.get("temperature", 0.7),
        "top_p": openai_req.get("top_p", 1.0),
        "max_new_tokens": max_completion_tokens,
    }

    payload = {
        "id": evaluation_id,
        "mode": "direct",
        "modelAId": model_id,
        "userMessageId": last_user_message_id,
        "modelAMessageId": model_a_message_id,
        "messages": arena_messages,
        "modality": modality,
        "userMessage": final_user_message_payload,
        "parameters": generation_params,
    }
    
    return payload, files_to_upload, messages_to_include


# --- Stream Generator ---
async def stream_generator(request_id: str, model: str, conversation_id: str, character_id: Optional[str]):
    queue = response_queues.get(request_id)
    if not queue:
        logging.error(f"Queue not found for request_id: {request_id}")
        return

    response_id = f"chatcmpl-{uuid_v4.uuid4()}"
    accumulated_content = ""
    
    try:
        while True:
            data = await queue.get()
            if data == "[DONE]":
                break
            
            if isinstance(data, dict) and "error" in data:
                error_payload = {"error": {"message": data["error"], "type": "server_error"}}
                yield f"data: {json.dumps(error_payload)}\n\n"
                break

            try:
                prefix, content = data.split(":", 1)
                if prefix == "a0":
                    delta = json.loads(content)
                    accumulated_content += delta
                    chunk = {
                        "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": delta}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif prefix == "ad":
                    finish_data = json.loads(content)
                    finish_reason = finish_data.get("finishReason", "stop")
                    final_chunk = {
                        "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
            except (ValueError, json.JSONDecodeError):
                logging.warning(f"Could not parse data chunk: {data}")
                continue
        
        if accumulated_content.strip():
            try:
                msg_id = f"{conversation_id}-out-{uuid_v4.uuid4()}"
                store_message_rag(
                    conversation_id=conversation_id,
                    message_id=msg_id,
                    role="assistant",
                    content=accumulated_content,
                    character_id=character_id,
                    turn_index=None,
                )
            except Exception as e:
                logging.error(f"RAG store (assistant, stream) failed: {e}")

        yield "data: [DONE]\n\n"
    
    except asyncio.CancelledError:
        logging.warning(f"Client for request {request_id} disconnected.")
        browser_ws = tab_manager.get_active_websocket()
        if browser_ws:
            await browser_ws.send_text(json.dumps({"type": "abort_request", "request_id": request_id}))
    finally:
        if request_id in response_queues:
            del response_queues[request_id]


# --- Non-Stream Response ---
async def non_stream_response_with_context(request_id: str, model: str, conversation_id: str, character_id: Optional[str]):
    """
    Enhanced non-stream response that includes compressed context
    """
    response_id = f"chatcmpl-{uuid_v4.uuid4()}"
    
    full_content = []
    finish_reason = "stop"
    
    queue = response_queues.get(request_id)
    if not queue:
        logging.error(f"Queue not found for request_id: {request_id}")
        error_response = {
            "error": {
                "message": "Internal server error: response channel not found",
                "type": "bridge_error",
                "code": "queue_not_found"
            }
        }
        return Response(content=json.dumps(error_response, ensure_ascii=False), status_code=500, media_type="application/json")
    
    try:
        while True:
            data = await queue.get()
            if data == "[DONE]":
                break
            
            if isinstance(data, dict) and "error" in data:
                logging.error(f"NON-STREAM [ID: {request_id[:8]}]: Error in data: {data}")
                error_response = {
                    "error": {
                        "message": f"[LMArena Bridge Error]: {data.get('error', 'Unknown error')}",
                        "type": "bridge_error",
                        "code": "processing_error"
                    }
                }
                return Response(content=json.dumps(error_response, ensure_ascii=False), status_code=500, media_type="application/json")
            
            try:
                prefix, content = data.split(":", 1)
                if prefix == "a0":
                    delta = json.loads(content)
                    full_content.append(delta)
                elif prefix == "ad":
                    finish_data = json.loads(content)
                    finish_reason = finish_data.get("finishReason", "stop")
                    if finish_reason == 'content-filter':
                        full_content.append("\n\n[Response terminated - possible context limit or content filter]")
            except (ValueError, json.JSONDecodeError):
                continue
    
    except Exception as e:
        logging.error(f"NON-STREAM [ID: {request_id[:8]}]: Exception: {e}")
        error_response = {
            "error": {
                "message": f"[LMArena Bridge Error]: {str(e)}",
                "type": "bridge_error",
                "code": "processing_error"
            }
        }
        return Response(content=json.dumps(error_response, ensure_ascii=False), status_code=500, media_type="application/json")
    finally:
        if request_id in response_queues:
            del response_queues[request_id]

    final_content = "".join(full_content)
    
    if final_content.strip():
        try:
            msg_id = f"{conversation_id}-out-{uuid_v4.uuid4()}"
            store_message_rag(
                conversation_id=conversation_id,
                message_id=msg_id,
                role="assistant",
                content=final_content,
                character_id=character_id,
                turn_index=None,
            )
        except Exception as e:
            logging.error(f"RAG store (assistant, non-stream) failed: {e}")
    
    response_data = {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": final_content
            },
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(final_content) // 4,
            "total_tokens": len(final_content) // 4,
        },
    }
    
    return Response(content=json.dumps(response_data, ensure_ascii=False), media_type="application/json")


# --- API Endpoints ---
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # ENHANCED: Longer, more human-like delays
    if STEALTH_CONFIG["enable_stealth"]:
        delay = random.uniform(*STEALTH_CONFIG["request_spacing"])
        logging.info(f"⏱️ Request delay: {delay:.1f}s")
        await asyncio.sleep(delay)
    
    browser_ws = tab_manager.get_active_websocket()
    if not browser_ws:
        raise HTTPException(status_code=503, detail="Browser client not connected.")

    openai_req = await request.json()
    request_id = str(uuid_v4.uuid4())
    model_name = openai_req.get("model")

    conversation_id = openai_req.get("conversation_id", "default")
    character_id = openai_req.get("character_id")

    # Store incoming messages in RAG
    for idx, msg in enumerate(openai_req.get("messages", [])):
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))

        if not content.strip():
            continue

        msg_id = f"{conversation_id}-in-{idx}"
        try:
            store_message_rag(
                conversation_id=conversation_id,
                message_id=msg_id,
                role=role,
                content=content,
                character_id=character_id,
                turn_index=idx,
            )
        except Exception as e:
            logging.error(f"RAG store (incoming) failed for {msg_id}: {e}")

    try:
        lmarena_payload, files_to_upload, compressed_messages = await create_lmarena_request_body(
            openai_req, 
            conversation_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    response_queues[request_id] = asyncio.Queue()

    message_to_browser = {
        "request_id": request_id,
        "payload": lmarena_payload,
        "files_to_upload": files_to_upload
    }

    async def send_to_browser():
        try:
            async with ws_lock:
                active_ws = tab_manager.get_active_websocket()
                if active_ws:
                    await active_ws.send_text(json.dumps(message_to_browser))
                    logging.info(f"Sent request {request_id[:8]} to browser")
                else:
                    raise Exception("No active browser WebSocket")
        except Exception as e:
            logging.error(f"Failed to send request {request_id} to browser: {e}")
            if request_id in response_queues:
                await response_queues[request_id].put({"error": "Failed to send request to browser."})

    task = asyncio.create_task(send_to_browser())
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

    if openai_req.get("stream", False):
        return StreamingResponse(
            stream_generator(request_id, model_name, conversation_id, character_id),
            media_type="text/event-stream"
        )
    else:
        return await non_stream_response_with_context(request_id, model_name, conversation_id, character_id)


@app.get("/v1/models")
async def get_models():
    return {
        "object": "list",
        "data": [
            {
                "id": name, "object": "model", "created": int(time.time()),
                "owned_by": "lmarena", "type": info.get("type", "chat")
            }
            for name, info in model_registry.items()
        ],
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9080)