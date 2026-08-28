"""Tests for ConversationStore — the SQLite-backed rolling turn store."""

from missions.glass_cockpit.store import MAX_TURNS, ConversationStore, Turn


def test_add_and_read_back_turns_oldest_first(tmp_path):
    store = ConversationStore()
    store.add_turn("hi", "hello")
    store.add_turn("how are you", "well")

    assert store.recent_turns() == [Turn("hi", "hello"), Turn("how are you", "well")]


def test_keeps_only_the_last_10_turns(tmp_path):
    store = ConversationStore()
    for i in range(15):
        store.add_turn(f"q{i}", f"a{i}")

    turns = store.recent_turns()
    assert len(turns) == MAX_TURNS
    assert turns[0] == Turn("q5", "a5")
    assert turns[-1] == Turn("q14", "a14")


def test_turns_persist_across_connections(tmp_path):
    ConversationStore().add_turn("remember", "this")

    assert ConversationStore().recent_turns() == [Turn("remember", "this")]


def test_empty_store_returns_no_turns(tmp_path):
    assert ConversationStore().recent_turns() == []
