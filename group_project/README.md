# Bài Tập Nhóm (CP5) — E-commerce Support RAG Chatbot & Evaluation Pipeline

Nhóm thực hiện: **MichelinBoys**

---

## 🎯 Tổng Quan Dự Án

Dự án xây dựng chatbot thông minh hỗ trợ giải đáp các thắc mắc về chính sách thương mại điện tử (đổi trả, thanh toán, ví ShopeePay, SPayLater, bảo mật thông tin, quy định người bán). Dự án bao gồm đầy đủ **Giao diện Chatbot Streamlit** và **RAG Evaluation Pipeline (A/B Testing)**.

### Thành phần chính:
1. **Chatbot Web Application (`app.py`)**: Giao diện Streamlit tương tác, hỗ trợ trích dẫn nguồn (citations), hiển thị điểm số truy xuất, câu hỏi gợi ý và quản lý lịch sử trò chuyện.
2. **Golden Dataset (`group_project/evaluation/golden_dataset.json`)**: Tập dữ liệu kiểm thử chuẩn gồm **16 câu hỏi và câu trả lời kỳ vọng**.
3. **Evaluation Pipeline (`group_project/evaluation/eval_pipeline.py`)**: Tự động đánh giá 4 chỉ số RAG (Faithfulness, Answer Relevance, Context Recall, Context Precision) và thực hiện A/B Testing giữa Hybrid Search vs Dense-Only.
4. **Báo cáo Kết quả (`group_project/evaluation/results.md`)**: Bảng so sánh chi tiết, phân tích thất bại trên 3 mẫu kém nhất (Worst Performers) và đề xuất hướng cải tiến.

---

## 🏗️ Kiến Trúc Hệ Thống

```mermaid
graph TD
    User([User Query]) --> UI[Streamlit UI - app.py]
    UI --> Pipe[Retrieval Pipeline - Task 9]
    
    subgraph Retrieval Pipeline
        Pipe --> Sem[Semantic Search BAAI/bge-m3]
        Pipe --> Lex[Lexical Search BM25]
        Sem --> RRF[Reciprocal Rank Fusion - RRF k=60]
        Lex --> RRF
        RRF --> CosCheck{Cosine Score < 0.30?}
        CosCheck -- Yes --> Fallback[PageIndex Vectorless Fallback]
        CosCheck -- No --> Chunks[Top-K Hybrid Chunks]
    end

    Fallback --> Reorder[Document Reordering: Front + Back]
    Chunks --> Reorder
    Reorder --> LLM[LLM Generation: GPT-4o-mini / OpenRouter]
    LLM --> Answer[Answer with Citations & Sources]
    Answer --> UI
```

---

## 📋 Phân Công Công Việc Nhóm (MichelinBoys)

| Thành viên | Vai trò | Nhiệm vụ đảm nhận | Trạng thái |
|-----------|---------|-------------------|------------|
| **MichelinBoys Leader** | Team Leader & RAG Architect | Quản lý dự án, kiểm thử `pytest`, thiết kế kiến trúc RAG Pipeline | **Hoàn thành** |
| **Data Specialist** | Data & Pipeline Engineer | Xây dựng Task 1..4, Task 9 (Hybrid Search, RRF Rerank, PageIndex Fallback) | **Hoàn thành** |
| **Frontend Dev** | UI & Chatbot Engineer | Xây dựng Task 10, giao diện Streamlit `app.py`, trích dẫn nguồn citation | **Hoàn thành** |
| **QA & Eval Engineer** | Evaluation Engineer | Xây dựng `golden_dataset.json`, script `eval_pipeline.py` & báo cáo `results.md` | **Hoàn thành** |

---

## 📊 Deliverables (Sản Phẩm Bàn Giao CP5)

- [x] **Streamlit UI (`app.py`)**: Chạy ổn định, hỗ trợ citation, top_k slider, clear history, nguồn tham khảo.
- [x] **Golden Dataset (`group_project/evaluation/golden_dataset.json`)**: 16 bộ Q&A được chuẩn hóa từ chính sách thương mại điện tử.
- [x] **Evaluation Script (`group_project/evaluation/eval_pipeline.py`)**: Tính toán tự động 4 metrics RAG và so sánh A/B.
- [x] **Evaluation Report (`group_project/evaluation/results.md`)**: Đầy đủ bảng điểm A/B testing, worst performers & recommendations.

---

## 🚀 Hướng Dẫn Chạy Dự Án

### 1. Khởi động Chatbot UI (Streamlit)
```bash
# Sử dụng virtual environment của dự án
.venv\Scripts\streamlit.exe run app.py
```

### 2. Thực thi RAG Evaluation Pipeline (A/B Testing)
```bash
# Chạy đánh giá và cập nhật báo cáo results.md
.venv\Scripts\python.exe group_project/evaluation/eval_pipeline.py
```

### 3. Kiểm tra toàn bộ Unit Tests
```bash
.venv\Scripts\python.exe -m pytest tests/test_individual.py
```
