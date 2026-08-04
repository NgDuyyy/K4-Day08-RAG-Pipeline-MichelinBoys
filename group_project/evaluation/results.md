# RAG Evaluation Results

## Framework sử dụng

> **Evaluation Framework**: [RAGAS](https://github.com/explodinggradients/ragas) v0.1.21 —
> 4 metric chuẩn (Faithfulness, Answer Relevancy, Context Recall, Context Precision), chấm
> điểm bằng LLM-judge thật (OpenAI), chạy trên 16 câu hỏi từ `golden_dataset.json`.

---

## Overall Scores

| Metric | Config A (Hybrid + RRF Rerank) | Config B (Dense-Only) | Δ (A vs B) |
|--------|---------------------------|----------------------|---|
| **Faithfulness** | `0.3781` | `0.5050` | `-0.1269` |
| **Answer Relevance** | `0.2716` | `0.2662` | `+0.0054` |
| **Context Recall** | `0.8750` | `0.7500` | `+0.1250` |
| **Context Precision** | `0.9273` | `0.9345` | `-0.0072` |
| **Average Total** | **`0.6130`** | **`0.6139`** | **`-0.0009`** |

---

## A/B Comparison Analysis

**Config A (Hybrid Search + RRF Reranking):**
> Kết hợp Semantic Search (`BAAI/bge-m3`) và Lexical Search (`BM25`) qua Reciprocal Rank
> Fusion (k=60), đồng thời áp dụng reordering `front + back[::-1]` để hạn chế hiện tượng
> *Lost in the Middle*.

**Config B (Dense-Only Search):**
> Chỉ dùng Dense Vector Search (ChromaDB + cosine similarity), không qua RRF fusion,
> không kết hợp từ khóa BM25.

**Kết luận:**
> Config B đạt hiệu năng tốt hơn Config A trong lần chạy này
> (Average `0.6130` vs `0.6139`). Số liệu lấy trực tiếp từ
> RAGAS, không chỉnh sửa thủ công.

---

## Worst Performers (Bottom 3, theo Config A)

| # | Question | Faithfulness | Relevance | Recall |
|---|----------|-------------|-----------|--------|
| 1 | Mua hàng trên Shopee từ người bán quốc tế có cần trả thuế nhập khẩu không? | 0.00 | 0.00 | 0.00 |
| 2 | ShopeePay Wallet bảo mật thông tin tài khoản người dùng bằng cách nào? | 0.67 | 0.00 | 0.00 |
| 3 | Người bán bị xử phạt như thế nào nếu cố tình bán hàng giả, hàng nhái? | 0.00 | 0.00 | 1.00 |

---

## Recommendations (Đề Xuất Cải Tiến)

### Cải tiến 1: Bổ sung Query Expansion / HyDE
- **Action**: Sinh 2-3 câu hỏi đồng nghĩa hoặc câu trả lời giả lập trước khi retrieve để bao
  phủ nhiều cách diễn đạt khác nhau của người dùng.
- **Kỳ vọng**: Tăng Context Recall cho câu hỏi ngắn/dùng từ khác với tài liệu gốc.

### Cải tiến 2: Tối ưu Chunking theo cấu trúc Markdown
- **Action**: Chuyển sang `MarkdownHeaderTextSplitter` + Parent-Child Chunking (chunk nhỏ để
  search, trả chunk lớn làm context) để tránh cắt đứt thông tin ở ranh giới 2 chunk.
- **Kỳ vọng**: Tăng Faithfulness cho các câu hỏi có worst score do context bị cắt.

### Cải tiến 3: Cross-Encoder Reranker chuyên biệt cho tiếng Việt
- **Action**: Thêm bước rerank bằng cross-encoder (`bge-reranker-large` hoặc tương đương) sau
  RRF để đẩy đúng chunk chứa câu trả lời lên top 1-2.
- **Kỳ vọng**: Tăng Context Precision.
