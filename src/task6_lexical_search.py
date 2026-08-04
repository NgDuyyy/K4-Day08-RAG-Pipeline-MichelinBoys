"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re
from pathlib import Path

# TODO: Load corpus từ data/standardized/ hoặc từ vector store
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _tokenize(text: str) -> list[str]:
    """Tokenize consistently while preserving Vietnamese letters."""
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def _load_standardized_corpus() -> list[dict]:
    """Load non-empty Markdown paragraphs as searchable chunks."""
    corpus = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8", errors="replace")
        chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", content) if chunk.strip()]
        doc_type = "legal" if "legal" in md_file.parts else "news"
        for chunk_index, chunk in enumerate(chunks):
            corpus.append({
                "content": chunk,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "chunk_index": chunk_index,
                },
            })
    return corpus


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    # TODO: Implement BM25 index
    #
    # from rank_bm25 import BM25Okapi
    #
    # # Tokenize - có thể đơn giản split(), hoặc dùng underthesea cho tiếng Việt
    # tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    # bm25 = BM25Okapi(tokenized_corpus)
    # return bm25
    if not corpus:
        return None

    from rank_bm25 import BM25Plus

    tokenized_corpus = [_tokenize(doc.get("content", "")) for doc in corpus]
    bm25 = BM25Plus(tokenized_corpus)
    return bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    # TODO: Implement lexical search
    #
    # tokenized_query = query.lower().split()
    # scores = bm25.get_scores(tokenized_query)
    #
    # # Get top_k indices
    # import numpy as np
    # top_indices = np.argsort(scores)[::-1][:top_k]
    #
    # results = []
    # for idx in top_indices:
    #     if scores[idx] > 0:
    #         results.append({
    #             "content": CORPUS[idx]["content"],
    #             "score": float(scores[idx]),
    #             "metadata": CORPUS[idx]["metadata"]
    #         })
    # return results
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    if not CORPUS:
        CORPUS.extend(_load_standardized_corpus())
    if not CORPUS:
        return []

    bm25 = build_bm25_index(CORPUS)
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    import numpy as np

    top_indices = np.argsort(scores)[::-1]
    query_terms = set(tokenized_query)
    results = []
    for idx in top_indices:
        document = CORPUS[int(idx)]
        if not query_terms.intersection(_tokenize(document.get("content", ""))):
            continue
        results.append({
            "content": document.get("content", ""),
            "score": float(scores[idx]),
            "metadata": dict(document.get("metadata") or {}),
        })
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
