"""Fetch a seeded, track-level subset of the original Slakh2100-redux mirror."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import re
import time
from urllib.parse import quote
from urllib.request import urlopen

HUB = "https://huggingface.co"
REPO = "J1mmymm/MIMuT_Data_v2"
ROOT = "data/Slakh2100_redux"
COUNTS = {"train": 120, "validation": 20, "test": 20}
SEED = 5305


def request(action):
    for attempt in range(6):
        try:
            return action()
        except (OSError, ValueError):
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** attempt))


def page(url):
    with urlopen(url, timeout=45) as response:
        return json.load(response), response.headers.get("Link", "")


def listing(revision, path, recursive=False):
    url = "{}/api/datasets/{}/tree/{}/{}?limit=1000".format(HUB, REPO, revision, quote(path))
    if recursive:
        url += "&recursive=true"
    while url:
        entries, link = request(lambda: page(url))
        yield from entries
        following = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = following.group(1) if following else None


def make_manifest():
    data, _ = request(lambda: page(HUB + "/api/datasets/" + REPO))
    revision = data["sha"]
    tracks = {}
    files = []
    rng = random.Random(SEED)
    for split, count in COUNTS.items():
        folders = sorted(Path(item["path"]).name for item in listing(revision, ROOT + "/" + split)
                         if item["type"] == "directory" and re.fullmatch(r"Track\d{5}", Path(item["path"]).name))
        tracks[split] = sorted(rng.sample(folders, count))
    with ThreadPoolExecutor(max_workers=6) as pool:
        paths = ["{}/{}/{}".format(ROOT, split, name) for split, names in tracks.items() for name in names]
        for entries in pool.map(lambda path: list(listing(revision, path, recursive=True)), paths):
            selected = [item for item in entries if item["type"] == "file" and
                        (item["path"].endswith((".flac", ".mid", "/metadata.yaml")))]
            required = {"mix.flac", "all_src.mid", "metadata.yaml"}
            if not required.issubset({Path(item["path"]).name for item in selected}):
                raise ValueError("Mirror track lacks required files: " + str(entries[:1]))
            files.extend({"path": item["path"], "size": item["size"],
                          "sha256": item.get("lfs", {}).get("oid")} for item in selected)
    return {"repo": REPO, "revision": revision, "seed": SEED, "tracks": tracks, "files": files}


def download_file(revision, target, entry):
    relative = entry["path"][len(ROOT) + 1:]
    path = target / relative
    if path.exists() and path.stat().st_size == entry["size"]:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    url = "{}/datasets/{}/resolve/{}/{}".format(HUB, REPO, revision, quote(entry["path"]))

    def fetch():
        digest = hashlib.sha256()
        count = 0
        with urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
                digest.update(chunk)
                count += len(chunk)
        if count != entry["size"] or (entry["sha256"] and digest.hexdigest() != entry["sha256"]):
            raise OSError("Invalid size or checksum: " + entry["path"])
        temporary.replace(path)
        return count

    return request(fetch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    if not path.exists():
        print("Resolving 160 tracks from the official split folders", flush=True)
        manifest = make_manifest()
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(path)
    else:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    files = manifest["files"]
    total = sum(entry["size"] for entry in files)
    print("{} files, {:.2f} GB".format(len(files), total / 1e9), flush=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download_file, manifest["revision"], root / "audio", entry): entry
                   for entry in files}
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            completed += futures[future]["size"]
            if index % 25 == 0 or index == len(files):
                print("{}/{} files · {:.2f}/{:.2f} GB".format(index, len(files), completed / 1e9, total / 1e9), flush=True)


if __name__ == "__main__":
    main()
