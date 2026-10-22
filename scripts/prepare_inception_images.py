#!/usr/bin/env python
"""Builds the image pool the InceptionV1 reports run on: `<out>/<wnid>/<n>.jpg`, a few images per class.

- Cats and cars come from a private S3 bucket of ImageNet train images, filtered to those classes.
  - The folder names there are the standard ImageNet-1k indices, so the wnid is the folder's rank among the sorted wnids.
- Everything else comes from the validation split of `timm/mini-imagenet`, 100 classes with 100 images each.
  - It supplies the dogs, which the bucket has almost none of.
- A class in both is taken from the bucket only, so its count stays at `--per-class`.

    uv run python scripts/prepare_inception_images.py --out data/inception_pool

Needs the `aws` CLI logged in with read access to the bucket.
"""

import argparse
import io
import json
import subprocess
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from in_tfm.models.stl_inception.source import spread_indices

BUCKET_NAME = "narang99-private"
BUCKET_PREFIX = f"s3://{BUCKET_NAME}/imagenet"

MINI_IMAGENET_URL = "https://huggingface.co/datasets/timm/mini-imagenet/resolve/main/data/validation-{shard:05d}-of-00003.parquet"
MINI_IMAGENET_SHARDS = 3

BUCKET_CLASSES = {
    281: "n02123045",  # tabby
    282: "n02123159",  # tiger cat
    283: "n02123394",  # Persian cat
    284: "n02123597",  # Siamese cat
    285: "n02124075",  # Egyptian cat
    286: "n02125311",  # cougar
    287: "n02127052",  # lynx
    288: "n02128385",  # leopard
    289: "n02128757",  # snow leopard
    290: "n02128925",  # jaguar
    291: "n02129165",  # lion
    292: "n02129604",  # tiger
    293: "n02130308",  # cheetah
    407: "n02701002",  # ambulance
    436: "n02814533",  # beach wagon
    468: "n02930766",  # cab
    511: "n03100240",  # convertible
    555: "n03345487",  # fire engine
    569: "n03417042",  # garbage truck
    573: "n03444034",  # go-kart
    661: "n03777568",  # Model T
    717: "n03930630",  # pickup
    734: "n03977966",  # police van
    751: "n04037443",  # racer
    817: "n04285008",  # sports car
}
"""Bucket folder to wnid."""


def run_aws(*args: str) -> str:
    result = subprocess.run(["aws", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"aws {' '.join(args[:3])} failed with {result.returncode}: {result.stderr}")
    return result.stdout


def keys_in_bucket_folder(folder: int) -> list[str]:
    """Only the objects that can be downloaded, a few are in GLACIER and need a restore first."""
    listing = run_aws(
        "s3api", "list-objects-v2", "--bucket", BUCKET_NAME, "--prefix", f"imagenet/{folder}/",
        "--query", "Contents[?StorageClass=='STANDARD'].Key", "--output", "text",
    )
    return sorted(key.rsplit("/", 1)[-1] for key in listing.split())


def chosen(keys: list[str], per_class: int) -> list[str]:
    """Evenly spread over the sorted keys, so the pick does not depend on listing order."""
    return [keys[i] for i in spread_indices(len(keys), min(per_class, len(keys)))]


def remove_files_not_in(class_dir: Path, keys: list[str]) -> None:
    """Leftovers of an earlier pick, so a rerun leaves exactly `per_class` images."""
    if class_dir.exists():
        for stale in set(path.name for path in class_dir.iterdir()) - set(keys):
            (class_dir / stale).unlink()


def fetch_bucket_class(folder: int, wnid: str, out: Path, per_class: int) -> None:
    keys = chosen(keys_in_bucket_folder(folder), per_class)
    remove_files_not_in(out / wnid, keys)
    include = [flag for key in keys for flag in ("--include", key)]
    run_aws("s3", "sync", f"{BUCKET_PREFIX}/{folder}/", str(out / wnid), "--exclude", "*", *include, "--only-show-errors")


def download_mini_imagenet_shard(shard: int, cache_dir: Path) -> Path:
    path = cache_dir / f"validation-{shard:05d}.parquet"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MINI_IMAGENET_URL.format(shard=shard), path)
    return path


def mini_imagenet_wnids(shard_path: Path) -> list[str]:
    metadata = pq.ParquetFile(shard_path).schema_arrow.metadata
    return json.loads(metadata[b"huggingface"])["info"]["features"]["label"]["names"]


def jpeg_bytes(encoded: bytes) -> bytes:
    image = Image.open(io.BytesIO(encoded))
    if image.format == "JPEG":
        return encoded
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=95)
    return buffer.getvalue()


def write_mini_imagenet_shard(shard_path: Path, out: Path, skip_wnids: set[str], per_class: int) -> None:
    wnids = mini_imagenet_wnids(shard_path)
    for row in pq.read_table(shard_path).to_pylist():
        wnid = wnids[row["label"]]
        class_dir = out / wnid
        if wnid in skip_wnids or (class_dir.exists() and len(list(class_dir.iterdir())) >= per_class):
            continue
        class_dir.mkdir(parents=True, exist_ok=True)
        (class_dir / f"mini_{len(list(class_dir.iterdir())):04d}.jpg").write_bytes(jpeg_bytes(row["image"]["bytes"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/inception_pool"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache/mini-imagenet"))
    parser.add_argument("--per-class", type=int, default=100)
    args = parser.parse_args()

    for folder, wnid in BUCKET_CLASSES.items():
        fetch_bucket_class(folder, wnid, args.out, args.per_class)
        print(f"bucket {folder} -> {wnid}")
    for shard in range(MINI_IMAGENET_SHARDS):
        write_mini_imagenet_shard(download_mini_imagenet_shard(shard, args.cache), args.out, set(BUCKET_CLASSES.values()), args.per_class)
        print(f"mini-imagenet shard {shard}")
    print(f"{len(list(args.out.rglob('*.jpg')))} images in {len(list(args.out.iterdir()))} classes")


if __name__ == "__main__":
    main()
