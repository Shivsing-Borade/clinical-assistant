import logging
from typing import List, Dict, Any, Tuple, Optional
from app.config import settings

logger = logging.getLogger(__name__)

def estimate_tokens(text: str) -> int:
    """
    Deterministic approximate token estimator.
    Rule of thumb: 1 token approx 4 characters for English.
    """
    if not text:
        return 0
    return len(text) // 4 + 1

def bound_context(messages: List[Dict[str, Any]],
                  max_turns: int = settings.CHAT_HISTORY_MAX_TURNS,
                  max_tokens: int = settings.CHAT_HISTORY_MAX_TOKENS) -> List[Dict[str, Any]]:
    """
    Bounds conversation history to a turn limit and a token budget.
    - Retains newest turns first.
    - Ensures complete user/assistant pairs.
    - Never drops the latest user message.
    """
    if not messages:
        return []

    # The latest message is always the current query; we preserve it.
    current_query = messages[-1]
    history = messages[:-1]

    bounded_pairs = []
    current_tokens = 0
    turns_count = 0

    # History is sorted ASC. Assume history[-1] is assistant, history[-2] is user.
    # We iterate backwards from the end of history to collect the most recent pairs.
    for i in range(len(history) - 1, 0, -2):
        if turns_count >= max_turns:
            break

        assistant_m = history[i]
        user_m = history[i-1]

        pair_tokens = estimate_tokens(user_m['text']) + estimate_tokens(assistant_m['text'])
        if current_tokens + pair_tokens > max_tokens:
            break

        bounded_pairs.append((user_m, assistant_m))
        current_tokens += pair_tokens
        turns_count += 1

    # Reverse the collected pairs to restore chronological order, then flatten.
    flat_history = []
    for user_m, assistant_m in reversed(bounded_pairs):
        flat_history.append(user_m)
        flat_history.append(assistant_m)

    return flat_history

def bound_evidence(chunks: List[Dict[str, Any]],
                  max_tokens: int = settings.GENERATION_EVIDENCE_MAX_TOKENS) -> List[Dict[str, Any]]:
    """
    Bounds retrieved evidence to a token budget and a maximum chunk count.
    - Keeps chunks whole.
    - Priority given to top-ranked chunks.
    - Obey both GENERATION_TOP_K_CONTEXT and GENERATION_EVIDENCE_MAX_TOKENS.
    """
    bounded_chunks = []
    current_tokens = 0
    max_chunks = settings.GENERATION_TOP_K_CONTEXT

    for chunk in chunks:
        if len(bounded_chunks) >= max_chunks:
            break

        chunk_tokens = estimate_tokens(chunk['content'])
        if current_tokens + chunk_tokens > max_tokens:
            break
        bounded_chunks.append(chunk)
        current_tokens += chunk_tokens

    return bounded_chunks
