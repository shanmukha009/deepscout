import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from deepscout.rag import chunk_text, retrieve, EmbeddingModel, VectorStore


def test_chunk_text_splits_and_is_nonempty():
    text = "word " * 500
    chunks = chunk_text(text, size=200)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_mock_embedding_is_deterministic():
    e = EmbeddingModel(provider="mock")
    a = e.embed(["the capital of France is Paris"])[0]
    b = e.embed(["the capital of France is Paris"])[0]
    assert a == b
    assert len(a) == 256


def test_retrieve_ranks_relevant_chunk_first():
    text = (
        "Paris is the capital of France. The Eiffel Tower stands in Paris. "
        "Berlin is the capital of Germany. The Brandenburg Gate stands in Berlin. "
        "Tokyo is the capital of Japan and is the largest city in Japan."
    )
    top = retrieve("What is the capital of France?", text, k=1, chunk_chars=45)
    assert "France" in top[0] or "Paris" in top[0]


def test_vector_store_search_returns_k():
    store = VectorStore(EmbeddingModel(provider="mock"))
    store.add(["alpha beta", "gamma delta", "epsilon zeta"])
    hits = store.search("alpha", k=2)
    assert len(hits) == 2
    assert hits[0][1] >= hits[1][1]  # sorted by score desc
