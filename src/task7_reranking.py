"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    # TODO: Implement cross-encoder reranking
    #
    # Option A: Jina Reranker API
    # import requests
    # response = requests.post(
    #     "https://api.jina.ai/v1/rerank",
    #     headers={"Authorization": f"Bearer {JINA_API_KEY}"},
    #     json={
    #         "model": "jina-reranker-v2-base-multilingual",
    #         "query": query,
    #         "documents": [c["content"] for c in candidates],
    #         "top_n": top_k
    #     }
    # )
    # reranked = response.json()["results"]
    # return [
    #     {**candidates[r["index"]], "score": r["relevance_score"]}
    #     for r in reranked
    # ]
    #
    # Option B: Local model (Qwen3-Reranker)
    # from transformers import AutoModelForSequenceClassification, AutoTokenizer
    # ...
    if not candidates or top_k <= 0:
        return []

    import os
    import requests

    api_key = os.getenv("JINA_API_KEY", "")
    if not api_key:
        raise ValueError("JINA_API_KEY is required for cross_encoder reranking")

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [candidate["content"] for candidate in candidates],
            "top_n": min(top_k, len(candidates)),
        },
        timeout=30,
    )
    response.raise_for_status()
    reranked = response.json()["results"]
    return [
        {
            **candidates[item["index"]],
            "score": float(item["relevance_score"]),
        }
        for item in reranked
    ]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    # TODO: Implement MMR
    #
    # selected = []
    # remaining = list(range(len(candidates)))
    #
    # for _ in range(min(top_k, len(candidates))):
    #     best_idx = None
    #     best_score = float('-inf')
    #
    #     for idx in remaining:
    #         # Relevance to query
    #         relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])
    #
    #         # Max similarity to already selected
    #         max_sim_to_selected = 0
    #         for sel_idx in selected:
    #             sim = cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
    #             max_sim_to_selected = max(max_sim_to_selected, sim)
    #
    #         # MMR score
    #         mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
    #
    #         if mmr_score > best_score:
    #             best_score = mmr_score
    #             best_idx = idx
    #
    #     selected.append(best_idx)
    #     remaining.remove(best_idx)
    #
    # return [candidates[i] for i in selected]
    if not candidates or top_k <= 0:
        return []
    if not 0 <= lambda_param <= 1:
        raise ValueError("lambda_param must be between 0 and 1")

    import numpy as np

    def cosine_sim(vector_a, vector_b):
        array_a = np.asarray(vector_a, dtype=float)
        array_b = np.asarray(vector_b, dtype=float)
        if array_a.shape != array_b.shape:
            raise ValueError("All embeddings must have the same dimension")
        denominator = np.linalg.norm(array_a) * np.linalg.norm(array_b)
        return float(np.dot(array_a, array_b) / denominator) if denominator else 0.0

    selected = []
    selected_scores = {}
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            embedding = candidates[idx].get("embedding")
            if embedding is None:
                raise ValueError("Each MMR candidate must contain an 'embedding'")

            relevance = cosine_sim(query_embedding, embedding)
            max_sim_to_selected = max(
                (
                    cosine_sim(embedding, candidates[sel_idx]["embedding"])
                    for sel_idx in selected
                ),
                default=0.0,
            )
            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * max_sim_to_selected
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        selected_scores[best_idx] = best_score
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = candidates[idx].copy()
        item["score"] = float(selected_scores[idx])
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    # TODO: Implement RRF
    #
    # rrf_scores = {}  # content -> score
    # content_map = {}  # content -> full dict
    #
    # for ranked_list in ranked_lists:
    #     for rank, item in enumerate(ranked_list, 1):
    #         key = item["content"]
    #         rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
    #         content_map[key] = item
    #
    # # Sort by RRF score
    # sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    #
    # results = []
    # for content, score in sorted_items[:top_k]:
    #     item = content_map[content].copy()
    #     item["score"] = score
    #     results.append(item)
    #
    # return results
    if top_k <= 0:
        return []
    if k < 0:
        raise ValueError("k must be non-negative")

    rrf_scores = {}
    content_map = {}
    first_seen = {}
    sequence = 0

    for ranked_list in ranked_lists:
        seen_in_list = set()
        for rank, item in enumerate(ranked_list, 1):
            key = str(item.get("content", "")).strip()
            if not key or key in seen_in_list:
                continue
            seen_in_list.add(key)
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            if key not in content_map:
                content_map[key] = item
                first_seen[key] = sequence
                sequence += 1

    sorted_items = sorted(
        rrf_scores.items(),
        key=lambda item: (-item[1], first_seen[item[0]]),
    )

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
    query_embedding: Optional[list[float]] = None,
    ranked_lists: Optional[list[list[dict]]] = None,
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        if not candidates or top_k <= 0:
            return []

        if query_embedding is not None:
            candidates_with_embeddings = candidates
        else:
            texts = [query] + [candidate["content"] for candidate in candidates]
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                matrix = TfidfVectorizer().fit_transform(texts).toarray()
            except (ImportError, ValueError):
                vocabulary = sorted({
                    token
                    for text in texts
                    for token in text.casefold().split()
                })
                matrix = [
                    [float(text.casefold().split().count(term)) for term in vocabulary]
                    for text in texts
                ]

            query_embedding = matrix[0].tolist() if hasattr(matrix[0], "tolist") else matrix[0]
            candidates_with_embeddings = []
            for candidate, embedding in zip(candidates, matrix[1:]):
                item = candidate.copy()
                item["embedding"] = (
                    embedding.tolist() if hasattr(embedding, "tolist") else embedding
                )
                candidates_with_embeddings.append(item)

        return rerank_mmr(
            query_embedding,
            candidates_with_embeddings,
            top_k=top_k,
        )
    elif method == "rrf":
        # RRF cần nhiều ranked lists - gọi riêng
        if ranked_lists is None:
            ranked_candidates = sorted(
                candidates,
                key=lambda item: item.get("score", 0),
                reverse=True,
            )
            ranked_lists = [ranked_candidates]
        return rerank_rrf(ranked_lists, top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 0.8, "metadata": {}},
        {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 0.6, "metadata": {}},
        {"content": "Quy định đăng bán sản phẩm dành cho người bán", "score": 0.5, "metadata": {}},
    ]
    results = rerank("chính sách trả hàng shopee", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
