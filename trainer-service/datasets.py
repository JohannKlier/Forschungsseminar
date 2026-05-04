from __future__ import annotations

from pathlib import Path
import re
import urllib.parse
import urllib.request

from paths import DATA_DIR


DATASETS = {
    "bike.csv": "1FhiamQAkPqF0OH8vYfxoo98VLTbxHFjy",
}


def _download_from_drive(file_id: str, dest: Path) -> None:
    base_url = "https://drive.google.com/uc?export=download"
    url = f"{base_url}&id={urllib.parse.quote(file_id)}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        content_type = resp.headers.get("Content-Type", "")
        data = resp.read()

    if "text/html" in content_type:
        text = data.decode("utf-8", errors="ignore")
        match = re.search(r"confirm=([0-9A-Za-z_]+)", text)
        if match:
            token = match.group(1)
            url = f"{base_url}&confirm={token}&id={urllib.parse.quote(file_id)}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                data = resp.read()

    dest.write_bytes(data)


def download_datasets_if_missing() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, file_id in DATASETS.items():
        dest = DATA_DIR / name
        if dest.exists():
            continue
        _download_from_drive(file_id, dest)


if __name__ == "__main__":
    download_datasets_if_missing()
