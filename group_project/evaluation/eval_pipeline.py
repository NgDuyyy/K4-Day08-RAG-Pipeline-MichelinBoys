"""
RAG Evaluation Pipeline — Checkpoint 5.

Sử dụng RAGAS để đánh giá chất lượng RAG pipeline với 4 metric chuẩn:
faithfulness, answer_relevancy, context_recall, context_precision.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question, sinh câu trả lời thật bằng LLM
    3. Evaluate bằng RAGAS (LLM-judge thật, KHÔNG phải heuristic đếm từ trùng)
    4. So sánh A/B 2 configs (Config A: Hybrid + RRF Rerank vs Config B: Dense-Only)
    5. Export results ra results.md

Cần OPENAI_API_KEY (hoặc OPENROUTER_API_KEY) trong .env — RAGAS gọi LLM thật để chấm
điểm, mỗi câu hỏi tốn nhiều lệnh gọi (sinh câu trả lời + 4 metric x LLM-judge), nên khi
thử nghiệm hãy chạy với SAMPLE_SIZE nhỏ trước, chỉ chạy full dataset 1 lần cuối để nộp bài.
"""

import json
import os
import sys
from pathlib import Path

# Fix Unicode encoding on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import (
    LLM_MODEL,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    format_context,
    reorder_for_llm,
)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

# Số câu hỏi chạy — None = chạy toàn bộ golden_dataset.json. Đặt số nhỏ (vd 5) khi thử
# nghiệm để tránh tốn quota/API cost, đặt None khi chạy lần cuối để nộp bài.
SAMPLE_SIZE = None


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if SAMPLE_SIZE:
        data = data[:SAMPLE_SIZE]
    return data


def _call_llm(context: str, query: str) -> str:
    """Gọi LLM sinh câu trả lời — cùng cấu hình (model/prompt/temperature) với Task 10."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not (openrouter_key or openai_key):
        raise RuntimeError("Cần OPENAI_API_KEY hoặc OPENROUTER_API_KEY trong .env để chạy evaluation")

    from openai import OpenAI

    if openrouter_key:
        client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1", timeout=30.0)
        model_name = LLM_MODEL
    else:
        client = OpenAI(api_key=openai_key, timeout=30.0)
        model_name = "gpt-4o-mini"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content or ""


def run_pipeline_config_a(query: str, top_k: int = 5) -> dict:
    """Config A: Hybrid Search (Semantic + BM25) + RRF Reranking (dùng retrieve() từ Task 9)."""
    chunks = retrieve(query, top_k=top_k, use_reranking=True)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    answer = _call_llm(context, query) if chunks else "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    return {"answer": answer, "sources": chunks}


def run_pipeline_config_b(query: str, top_k: int = 5) -> dict:
    """Config B: Dense-Only (chỉ Semantic Search, không BM25/RRF)."""
    chunks = semantic_search(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    answer = _call_llm(context, query) if chunks else "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    return {"answer": answer, "sources": chunks}


def build_ragas_dataset(golden_dataset: list[dict], config_fn) -> tuple[Dataset, list[dict]]:
    """Chạy pipeline trên từng câu hỏi, trả về Dataset cho RAGAS + raw results (để tìm worst performers)."""
    questions, answers, contexts_list, ground_truths = [], [], [], []
    raw_results = []

    for idx, item in enumerate(golden_dataset, 1):
        q = item["question"]
        print(f"  [{idx}/{len(golden_dataset)}] {q[:60]}...")
        result = config_fn(q)
        contexts = [s.get("content", "") for s in result["sources"]] or [""]

        questions.append(q)
        answers.append(result["answer"])
        contexts_list.append(contexts)
        ground_truths.append(item["expected_answer"])
        raw_results.append({"question": q, "answer": result["answer"]})

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })
    return dataset, raw_results


def evaluate_dataset(golden_dataset: list[dict]) -> tuple[dict, dict, list]:
    """Chạy đánh giá A/B trên toàn bộ golden dataset bằng RAGAS thật."""
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    print(f"\n=== Config A: Hybrid + RRF Rerank ({len(golden_dataset)} câu hỏi) ===")
    dataset_a, raw_a = build_ragas_dataset(golden_dataset, run_pipeline_config_a)
    print("Đang chấm điểm bằng RAGAS (Config A)...")
    scores_a = evaluate(dataset_a, metrics=metrics).to_pandas()

    print(f"\n=== Config B: Dense-Only ({len(golden_dataset)} câu hỏi) ===")
    dataset_b, raw_b = build_ragas_dataset(golden_dataset, run_pipeline_config_b)
    print("Đang chấm điểm bằng RAGAS (Config B)...")
    scores_b = evaluate(dataset_b, metrics=metrics).to_pandas()

    avg_a = {
        "faithfulness": round(float(scores_a["faithfulness"].mean()), 4),
        "answer_relevance": round(float(scores_a["answer_relevancy"].mean()), 4),
        "context_recall": round(float(scores_a["context_recall"].mean()), 4),
        "context_precision": round(float(scores_a["context_precision"].mean()), 4),
    }
    avg_a["average"] = round(sum(avg_a.values()) / 4, 4)

    avg_b = {
        "faithfulness": round(float(scores_b["faithfulness"].mean()), 4),
        "answer_relevance": round(float(scores_b["answer_relevancy"].mean()), 4),
        "context_recall": round(float(scores_b["context_recall"].mean()), 4),
        "context_precision": round(float(scores_b["context_precision"].mean()), 4),
    }
    avg_b["average"] = round(sum(avg_b.values()) / 4, 4)

    # Worst performers thực tế theo điểm trung bình 4 metric của Config A (không phải mẫu cứng)
    scores_a["avg_score"] = scores_a[["faithfulness", "answer_relevancy", "context_recall", "context_precision"]].mean(axis=1)
    worst_rows = scores_a.sort_values("avg_score").head(3)
    worst_performers = [
        {
            "question": row["question"],
            "faithfulness": round(float(row["faithfulness"]), 2),
            "answer_relevance": round(float(row["answer_relevancy"]), 2),
            "context_recall": round(float(row["context_recall"]), 2),
        }
        for _, row in worst_rows.iterrows()
    ]

    return avg_a, avg_b, worst_performers


def export_results_to_markdown(avg_a: dict, avg_b: dict, worst_performers: list, n_questions: int):
    """Xuất kết quả chi tiết ra file results.md."""

    def delta_str(val_a, val_b):
        d = val_a - val_b
        return f"+{d:.4f}" if d >= 0 else f"{d:.4f}"

    worst_table = ""
    for i, w in enumerate(worst_performers, 1):
        worst_table += (
            f"| {i} | {w['question']} | {w['faithfulness']:.2f} | "
            f"{w['answer_relevance']:.2f} | {w['context_recall']:.2f} |\n"
        )

    markdown_content = f"""# RAG Evaluation Results

## Framework sử dụng

> **Evaluation Framework**: [RAGAS](https://github.com/explodinggradients/ragas) v0.1.21 —
> 4 metric chuẩn (Faithfulness, Answer Relevancy, Context Recall, Context Precision), chấm
> điểm bằng LLM-judge thật (OpenAI), chạy trên {n_questions} câu hỏi từ `golden_dataset.json`.

---

## Overall Scores

| Metric | Config A (Hybrid + RRF Rerank) | Config B (Dense-Only) | Δ (A vs B) |
|--------|---------------------------|----------------------|---|
| **Faithfulness** | `{avg_a['faithfulness']:.4f}` | `{avg_b['faithfulness']:.4f}` | `{delta_str(avg_a['faithfulness'], avg_b['faithfulness'])}` |
| **Answer Relevance** | `{avg_a['answer_relevance']:.4f}` | `{avg_b['answer_relevance']:.4f}` | `{delta_str(avg_a['answer_relevance'], avg_b['answer_relevance'])}` |
| **Context Recall** | `{avg_a['context_recall']:.4f}` | `{avg_b['context_recall']:.4f}` | `{delta_str(avg_a['context_recall'], avg_b['context_recall'])}` |
| **Context Precision** | `{avg_a['context_precision']:.4f}` | `{avg_b['context_precision']:.4f}` | `{delta_str(avg_a['context_precision'], avg_b['context_precision'])}` |
| **Average Total** | **`{avg_a['average']:.4f}`** | **`{avg_b['average']:.4f}`** | **`{delta_str(avg_a['average'], avg_b['average'])}`** |

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
> {"Config A đạt hiệu năng tốt hơn Config B" if avg_a['average'] >= avg_b['average'] else "Config B đạt hiệu năng tốt hơn Config A trong lần chạy này"}
> (Average `{avg_a['average']:.4f}` vs `{avg_b['average']:.4f}`). Số liệu lấy trực tiếp từ
> RAGAS, không chỉnh sửa thủ công.

---

## Worst Performers (Bottom 3, theo Config A)

| # | Question | Faithfulness | Relevance | Recall |
|---|----------|-------------|-----------|--------|
{worst_table}
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
"""

    RESULTS_PATH.write_text(markdown_content, encoding="utf-8")
    print(f"\nSUCCESS: Exported evaluation results to `{RESULTS_PATH}`!")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases from golden_dataset.json")

    avg_a, avg_b, worst = evaluate_dataset(golden_dataset)
    export_results_to_markdown(avg_a, avg_b, worst, len(golden_dataset))
