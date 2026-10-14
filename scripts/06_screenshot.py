# -*- coding: utf-8 -*-
"""⑥ 截取问答页(可选)。

  pip install playwright && python -m playwright install chromium
  python scripts/06_screenshot.py

会先把服务在本机拉起,再用无头浏览器问一道题,等答案渲染完再截全图,
最后自动关掉服务 —— 不需要手动开两个终端。
"""

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

QUESTION = "贵州茅台2025年合并资产负债表里合同负债的期末余额是多少?用一句话回答并给出处。"


def wait_port(port: int, timeout: float = 60) -> bool:
    import socket

    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.4)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=QUESTION)
    ap.add_argument("--out", default=str(config.DOCS_DIR / "screenshot.png"))
    ap.add_argument("--full", action="store_true", default=True, help="整页截图(默认开)")
    ap.add_argument("--port", type=int, default=config.APP_PORT)
    a = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("未安装 playwright。先跑:")
        print("  pip install playwright && python -m playwright install chromium")
        return 1

    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    server = subprocess.Popen([sys.executable, str(config.BASE_DIR / "app" / "server.py")],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    threading.Thread(target=lambda: [None for _ in server.stdout], daemon=True).start()
    try:
        if not wait_port(a.port):
            print("服务没起来,端口", a.port, "不通")
            return 1
        print(f"服务已就绪 http://127.0.0.1:{a.port}")

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1180, "height": 1500},
                                    device_scale_factor=2)
            page.goto(f"http://127.0.0.1:{a.port}", wait_until="networkidle")
            page.fill("#q", a.question)
            page.click("#go")
            # 等「召回明细」出现,说明答案和明细都渲染完了
            page.wait_for_selector("#out .hit", timeout=120_000)
            page.wait_for_timeout(1200)
            # 展开第一个块的完整原文,截图里能看到表格还原的效果
            try:
                page.click("#out details summary")
                page.wait_for_timeout(400)
            except Exception:  # noqa: BLE001
                pass
            page.screenshot(path=a.out, full_page=a.full)
            browser.close()
        print(f"已截图 -> {a.out}")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=8)
        except Exception:  # noqa: BLE001
            server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
