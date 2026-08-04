"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    from .task4_chunking_indexing import chunk_documents, load_documents

    chunks = chunk_documents(load_documents())
    if not chunks:
        return []

    # Local dense representation: deterministic and usable without downloading
    # a large Hugging Face model during tests or offline classroom sessions.
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [chunk["content"] for chunk in chunks]
    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1
    )
    word_matrix = word_vectorizer.fit_transform(texts + [query])
    char_matrix = char_vectorizer.fit_transform(texts + [query])
    matrix = hstack([word_matrix, char_matrix], format="csr")
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked_indices = scores.argsort()[::-1][: min(top_k, len(chunks))]

    output = [
        {
            "content": chunks[int(index)]["content"],
            "score": float(scores[int(index)]),
            "metadata": dict(chunks[int(index)].get("metadata") or {}),
        }
        for index in ranked_indices
    ]

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("thời gian thử việc tối đa theo luật lao động", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
