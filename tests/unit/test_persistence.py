import pytest
from unittest.mock import MagicMock, patch
import os
import sqlite3
from app.storage.conversation_store import ConversationStore
from app.rag.context_manager import bound_context, estimate_tokens

def test_conversation_store_threads(tmp_path):
    db_path = str(tmp_path / "test_chat.sqlite3")
    store = ConversationStore(db_path=db_path)

    # Create threads
    tid1 = store.create_thread("Thread 1")
    tid2 = store.create_thread("Thread 2")

    threads = store.list_threads()
    assert len(threads) == 2
    assert threads[0]['title'] == "Thread 2" # LIFO check

    # Add messages
    store.add_message(tid1, "user", "Hello")
    store.add_message(tid1, "assistant", "Hi there")

    msgs = store.get_messages(tid1)
    assert len(msgs) == 2
    assert msgs[0]['text'] == "Hello"

    # Isolation check
    msgs2 = store.get_messages(tid2)
    assert len(msgs2) == 0

    # Rename and Delete
    store.rename_thread(tid1, "Renamed")
    # list_threads() returns threads ordered by updated_at DESC.
    # tid1 was just updated via rename, so it should be at index 0.
    assert store.list_threads()[0]['title'] == "Renamed"


    store.delete_thread(tid2)
    assert len(store.list_threads()) == 1

def test_context_bounding_logic():
    # Mock messages: User, Assistant, User, Assistant, User (Latest)
    messages = [
        {"role": "user", "text": "Question 1"},
        {"role": "assistant", "text": "Answer 1"},
        {"role": "user", "text": "Question 2"},
        {"role": "assistant", "text": "Answer 2"},
        {"role": "user", "text": "Question 3"},
    ]

    # Limit to 1 turn (1 pair)
    bounded = bound_context(messages, max_turns=1, max_tokens=1000)
    # Should contain: User 2, Assistant 2
    assert len(bounded) == 2
    assert bounded[0]['text'] == "Question 2"
    assert bounded[1]['text'] == "Answer 2"

    # Test token limit (very small)
    small_messages = [
        {"role": "user", "text": "Very long question " * 100},
        {"role": "assistant", "text": "Very long answer " * 100},
        {"role": "user", "text": "Current query"},
    ]
    bounded_small = bound_context(small_messages, max_turns=10, max_tokens=10)
    # Should be empty because the only available pair exceeds 10 tokens
    assert len(bounded_small) == 0

def test_token_estimation():
    assert estimate_tokens("") == 0
    assert estimate_tokens("Hello") == 2 # 5 // 4 + 1
    assert estimate_tokens("This is a test") == 4 # 14 // 4 + 1
