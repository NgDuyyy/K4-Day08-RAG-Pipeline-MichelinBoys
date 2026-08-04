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


# 5 bài hướng dẫn hỗ trợ khách hàng từ Shopee (theo dõi đơn hàng, thanh toán, hoàn tiền)
ARTICLE_URLS = [
    "https://shopee.vn/blog/kiem-tra-don-hang-shopee/",  # 1. Kiểm tra đơn hàng Shopee
    "https://help.shopee.vn/portal/4/article/183296-[%C4%90%C6%A1n-Qu%E1%BB%91c-T%E1%BA%BF]-H%C6%B0%E1%BB%9Bng-d%E1%BA%ABn-b%E1%BB%95-sung-th%C3%B4ng-tin-Ng%C6%B0%E1%BB%9Di-mua-cho-%C4%91%C6%A1n-h%C3%A0ng",  # 2. Bổ sung thông tin người mua cho đơn hàng quốc tế
    "https://help.shopee.vn/portal/4/article/189473-[Tr%E1%BA%A3-h%C3%A0ng/-Ho%C3%A0n-ti%E1%BB%81n]-Th%E1%BB%9Di-gian-nh%E1%BA%ADn-ti%E1%BB%81n-ho%C3%A0n-v%C3%A0-c%C3%A1ch-ki%E1%BB%83m-tra-ti%E1%BB%81n-ho%C3%A0n",  # 3. Thời gian nhận tiền hoàn & cách kiểm tra
    "https://help.shopee.vn/portal/4/article/79295-[Mua-h%C3%A0ng]-H%C6%B0%E1%BB%9Bng-d%E1%BA%ABn-ch%E1%BB%8Dn-ph%C6%B0%C6%A1ng-th%E1%BB%A9c-thanh-to%C3%A1n-khi-nh%E1%BA%ADn-h%C3%A0ng-(COD)",  # 4. Hướng dẫn chọn phương thức thanh toán COD
    "https://help.shopee.vn/portal/4/article/89669-[SPayLater]-C%C3%A2u-h%E1%BB%8Fi-th%C6%B0%E1%BB%9Dng-g%E1%BA%B7p-khi-k%C3%ADch-ho%E1%BA%A1t-SPayLater",  # 5. Câu hỏi thường gặp SPayLater
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
