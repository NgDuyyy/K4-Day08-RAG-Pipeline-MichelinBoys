"""
Task 2 — Crawl bài viết/hướng dẫn hỗ trợ khách hàng về thương mại điện tử.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trung tâm trợ giúp công khai của một sàn TMĐT.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: theo dõi đơn hàng, đổi phương thức thanh toán, bằng chứng hoàn tiền,
mua hàng xuyên biên giới.

Lưu ý: một số trang help center dùng JavaScript render (SPA) — nếu crawl về chỉ thấy
tiêu đề mà không có nội dung, đổi sang bài viết khác cùng domain thay vì cố xử lý.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài viết cần crawl
# 5 trang chuyên đề Luật Lao Động Việt Nam (100% công khai, không bị Cloudflare chặn, >15KB/bài)
ARTICLE_URLS = [
    "https://vi.wikipedia.org/wiki/B%E1%BB%99_lu%E1%BA%ADt_Lao_%C4%91%E1%BB%99ng_(Vi%E1%BB%87t_Nam)",  # 1. Bộ luật Lao động Việt Nam (Quyền, nghĩa vụ, điều khoản chung)
    "https://vi.wikipedia.org/wiki/H%E1%BB%A3p_%C4%91%E1%BB%93ng_lao_%C4%91%E1%BB%99ng",  # 2. Hợp đồng lao động (Các loại hợp đồng, ký kết, chấm dứt, kỷ luật)
    "https://vi.wikipedia.org/wiki/B%E1%BA%A3o_hi%E1%BB%83m_x%C3%A3_h%E1%BB%99i_Vi%E1%BB%87t_Nam",  # 3. Bảo hiểm xã hội Việt Nam (Chế độ hưu trí, thai sản, ốm đau)
    "https://vi.wikipedia.org/wiki/B%E1%BA%A3o_hi%E1%BB%83m_th%E1%BA%A5t_nghi%E1%BB%87p",  # 4. Bảo hiểm thất nghiệp (Điều kiện hưởng, trợ cấp, thủ tục)
    "https://vi.wikipedia.org/wiki/Ti%E1%BB%81n_l%C6%B0%C6%A1ng",  # 5. Tiền lương & chế độ đãi ngộ (Lương tối thiểu, làm thêm giờ)
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    # TODO: Implement crawling logic
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        return {
            "url": url,
            "title": result.metadata.get("title", "Unknown"),
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown,
        }
    # raise NotImplementedError("Implement crawl_article")


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
            # Lưu file JSON với tên article_01.json -> article_05.json
            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  ✓ Saved: {filepath} ({filepath.stat().st_size} bytes)")
        except Exception as e:
            print(f"  ❌ Lỗi khi crawl {url}: {e}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang hướng dẫn/hỗ trợ khách hàng trên help center của sàn TMĐT")
    else:
        asyncio.run(crawl_all())
