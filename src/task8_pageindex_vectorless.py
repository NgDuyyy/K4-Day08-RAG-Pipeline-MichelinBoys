"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    # TODO: Implement upload
    #
    # Tham khảo: https://github.com/VectifyAI/PageIndex
    #
    # from pageindex.client import PageIndexClient
    #
    # client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    #
    # for md_file in STANDARDIZED_DIR.rglob("*.md"):
    #     # Lưu ý: PageIndex nhận PDF, không nhận .md trực tiếp — có thể cần
    #     # convert markdown sang PDF đơn giản bằng fpdf2 trước khi upload.
    #     resp = client.submit_document(str(pdf_path))
    #     doc_id = resp.get("doc_id") or resp.get("id")
    #     print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")
    if not PAGEINDEX_API_KEY:
        raise ValueError("PAGEINDEX_API_KEY is required to upload documents")

    import json

    try:
        from pageindex import PageIndexClient
    except ImportError:
        from pageindex.client import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    project_dir = Path(__file__).parent.parent
    cache_path = project_dir / "pageindex_doc_ids.json"

    cached_records = []
    if cache_path.exists():
        try:
            cached_records = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(cached_records, list):
                cached_records = []
        except (json.JSONDecodeError, OSError):
            cached_records = []

    records_by_source = {
        record["source"]: record
        for record in cached_records
        if isinstance(record, dict) and record.get("source") and record.get("doc_id")
    }

    markdown_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    upload_items = []

    if markdown_files:
        try:
            import inspect
            import shutil
            from fpdf import FPDF
        except ImportError as error:
            raise ImportError(
                "fpdf2 is required to convert standardized Markdown files to PDF"
            ) from error

        uses_fpdf2_api = "text" in inspect.signature(FPDF.multi_cell).parameters

        pdf_dir = project_dir / "pageindex_pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        font_candidates = [
            Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
        ]
        unicode_font = next((path for path in font_candidates if path.exists()), None)

        for md_file in markdown_files:
            source_key = str(md_file.resolve())
            if source_key in records_by_source:
                continue

            pdf_path = pdf_dir / md_file.relative_to(STANDARDIZED_DIR).with_suffix(".pdf")
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            if unicode_font:
                if uses_fpdf2_api:
                    pdf.add_font("DocumentFont", fname=str(unicode_font))
                else:
                    cached_font = pdf_dir / "DocumentFont.ttf"
                    if not cached_font.exists():
                        shutil.copyfile(unicode_font, cached_font)
                    pdf.add_font("DocumentFont", "", str(cached_font), uni=True)
                pdf.set_font("DocumentFont", size=10)
            else:
                pdf.set_font("Helvetica", size=10)

            content = md_file.read_text(encoding="utf-8", errors="replace")
            for line in content.splitlines() or [""]:
                if not unicode_font:
                    line = line.encode("latin-1", errors="replace").decode("latin-1")
                pdf.set_x(pdf.l_margin)
                if uses_fpdf2_api:
                    pdf.multi_cell(w=0, h=5, text=line or " ", wrapmode="CHAR")
                else:
                    pdf.multi_cell(0, 5, line or " ")
            pdf.output(str(pdf_path))
            upload_items.append((md_file, pdf_path))
    else:
        landing_dir = project_dir / "data" / "landing"
        upload_items.extend(
            (pdf_file, pdf_file)
            for pdf_file in sorted(landing_dir.rglob("*.pdf"))
            if str(pdf_file.resolve()) not in records_by_source
        )

    for source_path, upload_path in upload_items:
        source_key = str(source_path.resolve())
        response = client.submit_document(str(upload_path))
        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(f"PageIndex did not return doc_id for {source_path.name}")

        record = {
            "source": source_key,
            "uploaded_file": str(upload_path.resolve()),
            "doc_id": doc_id,
        }
        records_by_source[source_key] = record
        print(f"  ✓ Uploaded: {source_path.name} -> {doc_id}")

    cache_path.write_text(
        json.dumps(list(records_by_source.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return list(records_by_source.values())


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    # TODO: Implement PageIndex query
    #
    # from pageindex.client import PageIndexClient
    #
    # client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    # resp = client.submit_query(doc_id=doc_id, query=query)
    # retrieval_id = resp.get("retrieval_id") or resp.get("id")
    #
    # # Poll cho đến khi status == "completed"
    # retrieval = client.get_retrieval(retrieval_id)
    #
    # # Parse retrieval["retrieved_nodes"] — mỗi node có "relevant_contents"
    # results = []
    # for node in retrieval.get("retrieved_nodes", [])[:2]:
    #     for group in node.get("relevant_contents", []):
    #         for item in group:
    #             results.append({
    #                 "content": item.get("relevant_content", ""),
    #                 "score": ...,  # PageIndex không trả score trực tiếp — tự gán theo rank
    #                 "metadata": {"section": item.get("section_title")},
    #                 "source": "pageindex",
    #             })
    # return results[:top_k]
    if not PAGEINDEX_API_KEY or not query.strip() or top_k <= 0:
        return []

    import json
    import time

    try:
        from pageindex import PageIndexClient
    except ImportError:
        from pageindex.client import PageIndexClient

    project_dir = Path(__file__).parent.parent
    cache_path = project_dir / "pageindex_doc_ids.json"
    doc_ids = []

    configured_ids = os.getenv("PAGEINDEX_DOC_IDS", "")
    configured_ids += "," + os.getenv("PAGEINDEX_DOC_ID", "")
    doc_ids.extend(doc_id.strip() for doc_id in configured_ids.split(",") if doc_id.strip())

    if cache_path.exists():
        try:
            cached_records = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached_records, list):
                doc_ids.extend(
                    record.get("doc_id")
                    for record in cached_records
                    if isinstance(record, dict) and record.get("doc_id")
                )
        except (json.JSONDecodeError, OSError):
            pass

    doc_ids = list(dict.fromkeys(doc_ids))
    if not doc_ids:
        return []

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    try:
        poll_timeout = max(0.0, float(os.getenv("PAGEINDEX_POLL_TIMEOUT", "60")))
    except ValueError:
        poll_timeout = 60.0
    results_by_content = {}
    first_seen = {}
    sequence = 0

    def relevant_items(value):
        if isinstance(value, dict):
            if value.get("relevant_content"):
                yield value
        elif isinstance(value, list):
            for child in value:
                yield from relevant_items(child)

    try:
        for doc_id in doc_ids:
            if hasattr(client, "is_retrieval_ready") and not client.is_retrieval_ready(doc_id):
                continue

            response = client.submit_query(doc_id=doc_id, query=query, thinking=False)
            retrieval_id = response.get("retrieval_id") or response.get("id")
            if not retrieval_id:
                continue

            deadline = time.monotonic() + poll_timeout
            retrieval = {}
            while time.monotonic() <= deadline:
                retrieval = client.get_retrieval(retrieval_id)
                status = str(retrieval.get("status", "")).lower()
                if status == "completed":
                    break
                if status in {"failed", "error", "cancelled"}:
                    retrieval = {}
                    break
                time.sleep(1)

            if str(retrieval.get("status", "")).lower() != "completed":
                continue

            document_rank = 0
            seen_in_document = set()
            for node in retrieval.get("retrieved_nodes", []):
                for item in relevant_items(node.get("relevant_contents", [])):
                    content = str(item.get("relevant_content", "")).strip()
                    if not content or content in seen_in_document:
                        continue
                    seen_in_document.add(content)
                    document_rank += 1
                    rank_score = float(1 / document_rank)

                    if content in results_by_content:
                        results_by_content[content]["score"] += rank_score
                        continue

                    results_by_content[content] = {
                        "content": content,
                        "score": rank_score,
                        "metadata": {
                            "section": item.get("section_title") or node.get("title"),
                            "page_index": item.get("page_index"),
                            "node_id": node.get("node_id"),
                            "doc_id": doc_id,
                        },
                        "source": "pageindex",
                    }
                    first_seen[content] = sequence
                    sequence += 1
    except Exception:
        pass

    results = sorted(
        results_by_content.values(),
        key=lambda result: (-result["score"], first_seen[result["content"]]),
    )
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
