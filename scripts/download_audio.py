"""Bounded, resumable downloads of WAV blobs at an immutable dataset commit."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "naweedfatima03/SentimentAnalysisDataset"
COMMIT = "8c764b0b778f90c8b60484afe21715c899d35663"


def fetch(item):
    target = ROOT / "data/raw" / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        content = target.read_bytes()
        digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if digest == item["sha"]:
            return
    for attempt in range(5):
        try:
            url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{item['path']}"
            content = urllib.request.urlopen(url, timeout=45).read()
            digest = hashlib.sha1(
                f"blob {len(content)}\0".encode() + content
            ).hexdigest()
            if digest != item["sha"] or content[:4] != b"RIFF":
                raise ValueError("Blob integrity or RIFF signature mismatch")
            target.write_bytes(content)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1 + attempt)


if __name__ == "__main__":
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1",
        headers={"User-Agent": "ML-CEP-study"},
    )
    tree = json.load(urllib.request.urlopen(request, timeout=45))
    assert not tree.get("truncated", False)
    files = [
        item
        for item in tree["tree"]
        if item["path"].startswith("NoiseAudioWAV/") and item["path"].endswith(".wav")
    ]
    assert len(files) == 7442
    started = time.time()
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(fetch, item) for item in files]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 250 == 0:
                print(
                    f"{count}/7442 verified WAVs, elapsed {time.time() - started:.1f}s",
                    flush=True,
                )
    provenance = {
        "repository": f"https://github.com/{REPO}",
        "commit": COMMIT,
        "wav_count": len(files),
        "download_method": "individual immutable blobs; every Git blob SHA-1 verified",
        "total_bytes": sum(item["size"] for item in files),
    }
    (ROOT / "data/provenance.json").write_text(json.dumps(provenance, indent=2))
    (ROOT / "data/source_inventory.json").write_text(json.dumps(files, indent=2))
    print(provenance, flush=True)
