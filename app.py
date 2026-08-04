"""
RAG Chatbot — E-commerce Support (Checkpoint 5)
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================

st.set_page_config(
    page_title="E-commerce Support RAG Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #EE4D2D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .source-badge {
        background-color: #e6f7ff;
        color: #1890ff;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stButton>button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("🛒 E-commerce RAG")
    st.caption("Trợ lý hỗ trợ khách hàng & chính sách e-commerce (Đổi trả, Thanh toán, SPayLater, Bảo mật, Quyền riêng tư)")

    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?",
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "SPayLater là gì và ai được phép sử dụng?",
        "Quy định đăng bán sản phẩm cho người bán?",
        "Quy trình xử lý khi nhận hàng bể vỡ như thế nào?",
        "Shopee thu thập những thông tin cá nhân nào?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{hash(s)}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Cấu hình Retrieval & RAG")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)
    
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Kiến trúc RAG Pipeline (MichelinBoys):**")
    st.caption("Hybrid Search (BAAI/bge-m3 + BM25) ➔ RRF Rerank ($k=60$) ➔ PageIndex Fallback ($cosine < 0.30$) ➔ LLM Generation với Reordering & Citations")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.markdown('<div class="main-header">🛒 E-commerce Support RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Trợ lý ảo tra cứu chính sách thương mại điện tử & giải đáp thắc mắc người dùng</div>', unsafe_allow_html=True)

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            ret_src = msg.get("retrieval_source", "hybrid")
            with st.expander(f"📚 Nguồn tham khảo ({len(msg['sources'])} chunks | Via: `{ret_src}`)"):
                for i, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("source", "Unknown")
                    doc_type = meta.get("type", "unknown")
                    score = src.get("score", 0)
                    st.markdown(f"**[{i}] {source_name}** `{doc_type}` | Score: `{score:.4f}`")
                    st.text(src.get("content", "")[:350] + "...")
                    st.divider()

# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách/hỗ trợ e-commerce...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    # Hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Sinh câu trả lời từ RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang truy xuất tài liệu và tổng hợp câu trả lời..."):
            try:
                from src.task10_generation import generate_with_citation
                response = generate_with_citation(query, top_k=top_k)
                answer = response.get("answer", "Chưa thể trả lời câu hỏi.")
                sources = response.get("sources", [])
                retrieval_source = response.get("retrieval_source", "hybrid")

            except Exception as e:
                answer = f"❌ **Lỗi RAG Pipeline:** {e}"
                sources = []
                retrieval_source = "error"

            st.markdown(answer)

            if sources:
                with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks | Via: `{retrieval_source}`)"):
                    for i, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        source_name = meta.get("source", "Unknown")
                        doc_type = meta.get("type", "unknown")
                        score = src.get("score", 0)
                        st.markdown(f"**[{i}] {source_name}** `{doc_type}` | Score: `{score:.4f}`")
                        st.text(src.get("content", "")[:350] + "...")
                        st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
    })
