"""Every report of a neuron, built from one saved hit space, so the reports share cluster ids and link to each other.

Each neuron gets `<out_dir>/neuron_<idx>/`:

- `space/` holds the hits, the distance matrix, the forest and the probabilities. See in_tfm.hit_space.
  - It is collected once, and `collected.json` records the settings it was collected with.
  - A later run with the same settings reuses it, and a run with different ones stops instead of mixing them.
- `clusters/` is the cluster report: each cluster's sampled hits, plus the sheet of hits split between clusters.
  - Its labels are each confident hit's most probable cluster, the same clusters the bucket report names.
  - Every cluster is shown, even one drawn from a single sample, so every link from the bucket report lands.
- `buckets/` is the bucket report: the map of sampled hits, a page per bucket and the cards.
- `index.html` links the two reports, and `manifest.json` records the config, versions, counts and timings.

The run ends by checking that every link and image of every page under `out_dir` lands, and fails if one does not.
"""

import json
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel
from scipy.spatial.distance import cdist

from .activation_ranges import sample_noise
from .bucket_graph import peak_hits, sample_hits, sample_map
from .bucket_report import BucketReportInputs, write_bucket_report
from .cluster_membership import NOISE_LABEL
from .density_peaks import calculate_probs, comparable_rows
from .hadamard import near_square_shape
from .hit_buckets import Bucket, all_buckets, buckets_of_cluster, paged_buckets
from .hit_selection import select_hits
from .hit_space import (
    HitSpace, confident_hits, entropy_bits, load_distances, load_space, save_distances, save_forest, save_hits,
    space_is_complete,
)
from .html_report import PageLink, page, render_template
from .mixed_hits_report import write_mixed_hits_page
from .neuron_capture import ScanResult, merge_positions
from .neuron_report import AttrFn, NeuronClusterHits, ReportLinks
from .report_links import check_links
from .neuron_run import (
    PreparedRun, clustered_rows, density_peak_trees, layer_getter_for, polarities_for, report_name, timed,
)
from .run_config import ClusteringConfig, DensityPeaksConfig, Polarity, RunConfig
from .weighted_jaccard import weighted_jaccard_distances

ScanOnce = Callable[["StageTimer"], ScanResult]

COLLECTED = "collected.json"
MANIFEST = "manifest.json"
BUCKETS_JSON = "buckets.json"
CLUSTER_REPORT = "clusters"
BUCKET_REPORT = "buckets"
INDEX_PAGE = "index.html"
MIN_UNIQ_SAMPLES = 1
"""The cluster report keeps every cluster, so a link to any cluster has a section to land on."""


class CollectionSettings(BaseModel):
    """The settings that decide a space. Anything else, such as the report settings, can change between runs."""

    model: str
    target: str
    layer_idx: int
    layer_name: str | None
    neuron_idx: int
    polarity: Polarity
    seed: int
    data: dict[str, Any]
    hits: dict[str, Any]
    clustering: dict[str, Any]


class Manifest(BaseModel):
    neuron_idx: int
    polarity: Polarity
    written_at: str
    git_commit: str | None
    device: str
    versions: dict[str, str]
    collection: CollectionSettings
    space_reused: bool
    config: dict[str, Any]
    counts: dict[str, int]
    timings: dict[str, float]
    """Seconds per stage."""


class NeuronDirs(BaseModel):
    root: Path

    @property
    def space(self) -> Path:
        return self.root / "space"

    @property
    def clusters(self) -> Path:
        return self.root / CLUSTER_REPORT

    @property
    def buckets(self) -> Path:
        return self.root / BUCKET_REPORT


class StageTimer(BaseModel):
    seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, label: str) -> Iterator[None]:
        start = time.perf_counter()
        with timed(label):
            yield
        self.seconds[label] = round(time.perf_counter() - start, 2)


def density_peaks_of(clustering: ClusteringConfig) -> DensityPeaksConfig:
    strategy = clustering.strategy
    if not isinstance(strategy, DensityPeaksConfig):
        raise ValueError("the space reports read cluster probabilities, so clustering.strategy.method must be density_peaks")
    return strategy


def collection_settings(config: RunConfig, neuron_idx: int, polarity: Polarity) -> CollectionSettings:
    return CollectionSettings(
        model=config.model.value, target=config.target.value, layer_idx=config.layer_idx, layer_name=config.layer_name,
        neuron_idx=neuron_idx, polarity=polarity, seed=config.seed,
        data=getattr(config, config.modality).model_dump(mode="json"),
        hits=config.hits.model_dump(mode="json"), clustering=config.clustering.model_dump(mode="json"),
    )


def saved_settings(space_dir: Path) -> CollectionSettings | None:
    """None when the space was never finished: `collected.json` is written last."""
    if not (space_dir / COLLECTED).exists() or not space_is_complete(space_dir):
        return None
    return CollectionSettings.model_validate_json((space_dir / COLLECTED).read_text())


def space_to_reuse(space_dir: Path, wanted: CollectionSettings) -> bool:
    saved = saved_settings(space_dir)
    if saved is None:
        return False
    if saved != wanted:
        raise ValueError(
            f"{space_dir} was collected with other settings. Use another out_dir, or delete the space to recollect.\n"
            f"saved:  {saved.model_dump_json()}\nwanted: {wanted.model_dump_json()}"
        )
    return True


def hit_distances(hadamards: Float[np.ndarray, "n_hits hidden"], clustering: ClusteringConfig) -> Float[np.ndarray, "n_hits n_hits"]:
    """The same distance the forest grows on, between every pair of hits rather than distinct points."""
    if clustering.distance == "weighted_jaccard":
        return weighted_jaccard_distances(hadamards.astype(np.float64), normalise=clustering.normalise)
    rows = comparable_rows(hadamards.astype(np.float64), clustering.distance, clustering.normalise)
    return cdist(rows, rows)


def collect_space(
    config: RunConfig, run: PreparedRun, scan: ScanResult, neuron_idx: int, polarity: Polarity, space_dir: Path,
    timer: StageTimer,
) -> None:
    space_dir.mkdir(parents=True, exist_ok=True)
    selection = select_hits(scan, neuron_idx, polarity, config.hits, config.seed)
    merged = merge_positions([(selection.batch_idx, selection.token_idx)], scan.valid_mask.shape[1])
    with timer.stage("gather"):
        run.model.to(config.device)
        gathered = run.capture.gather(merged.batch_idx, merged.token_idx)
    worst = run.capture.check_consistent(scan, gathered, merged.batch_idx, merged.token_idx)
    print(f"gathered {len(merged.batch_idx)} positions; pass 2 matches pass 1 to {worst:.2e}")
    run.model.to("cpu")
    weight = run.layout.weight(layer_getter_for(config)(run.module).weight)
    hadamards = clustered_rows(gathered.inputs[merged.index_maps[0]], weight, neuron_idx, config.clustering)
    save_hits(hadamards, selection, scan.sample_ids, sample_noise(scan, neuron_idx, config.seed), space_dir)

    with timer.stage("distance matrix"):
        save_distances(hit_distances(hadamards, config.clustering), space_dir)
    strategy = density_peaks_of(config.clustering)
    with timer.stage("forest and probabilities"):
        forest, trees = density_peak_trees(hadamards, config.clustering, strategy)
        save_forest(forest, trees, calculate_probs(forest, trees, strategy.peak_share), space_dir)
    (space_dir / COLLECTED).write_text(collection_settings(config, neuron_idx, polarity).model_dump_json(indent=2))


def hit_names(space: HitSpace) -> list[str]:
    return [f"{space.sample_ids[b]}:{t}" for b, t in zip(space.batch_idx, space.token_idx)]


def write_buckets_json(buckets: list[Bucket], space: HitSpace, path: Path) -> None:
    """Every non-empty bucket, paged or not, with each hit named `<sample id>:<token idx>`."""
    names = hit_names(space)
    records = [{"name": b.name, "clusters": list(b.clusters), "hits": [names[i] for i in b.hits]} for b in buckets]
    path.write_text(json.dumps(records))


def confident_labels(space: HitSpace, max_entropy: float) -> Int[np.ndarray, "n_hits"]:
    labels = space.probabilities.argmax(axis=1)
    labels[~confident_hits(space.probabilities, max_entropy)] = NOISE_LABEL
    return labels


def space_cluster_hits(space: HitSpace, run: PreparedRun, config: RunConfig, neuron_idx: int, max_entropy: float) -> NeuronClusterHits:
    return NeuronClusterHits(
        model=run.module, source=run.source, presenter=run.presenter, sample_ids=space.sample_ids,
        layer_getter=layer_getter_for(config), neuron_idx=neuron_idx, selection=space.hit_selection(),
        labels=confident_labels(space, max_entropy), hdmds=space.hadamards,
        hadamard_shape=near_square_shape(space.hadamards.shape[1]), noise_activations=space.noise, device=config.device,
    )


def cluster_bucket_notes(paged: list[Bucket], n_clusters: int) -> dict[int, str]:
    """Html listing, for each cluster, the bucket pages it is in."""
    notes = {}
    for cluster in range(n_clusters):
        links = [
            f'<a href="../{BUCKET_REPORT}/{b.slug}.html">{b.name}</a> ({len(b.hits)})'
            for b in buckets_of_cluster(paged, cluster)
        ]
        notes[cluster] = (
            f'<p class="meta">in buckets: {" &middot; ".join(links)}</p>' if links
            else '<p class="meta">in no bucket with a page</p>'
        )
    return notes


def write_cluster_report(
    hits: NeuronClusterHits, space: HitSpace, paged: list[Bucket], dirs: NeuronDirs, config: RunConfig,
    max_entropy: float, attr_fn: AttrFn,
) -> Path:
    write_mixed_hits_page(
        hits, dirs.clusters, space.probabilities, entropy_bits(space.probabilities), max_entropy, MIN_UNIQ_SAMPLES, attr_fn,
    )
    links = ReportLinks(
        page_links=[PageLink(text="bucket report", href=f"../{BUCKET_REPORT}/{INDEX_PAGE}"), PageLink(text="neuron", href=f"../{INDEX_PAGE}")],
        cluster_notes=cluster_bucket_notes(paged, space.n_clusters),
    )
    return hits.write_report(
        dirs.root, name=CLUSTER_REPORT, attr_fn=attr_fn, max_n=config.report.max_hits_per_cluster,
        min_uniq_images=MIN_UNIQ_SAMPLES, links=links,
    )


def write_space_bucket_report(
    hits: NeuronClusterHits, space: HitSpace, paged: list[Bucket], dirs: NeuronDirs, config: RunConfig,
    max_entropy: float, attr_fn: AttrFn, timer: StageTimer,
) -> tuple[Path, int]:
    """Returns the index and how many hits are on the map."""
    peaks = peak_hits(space.distinct_of_hit, space.roots)
    sample = sample_hits(paged, space.probabilities, peaks, config.space.map.sampled_per_bucket)
    with timer.stage("layout"):
        mapped = sample_map(load_distances(dirs.space), space.probabilities, sample, peaks, config.space.map)
    print(f"{len(sample)} sampled hits, {mapped.neighbours} neighbours each")
    inputs = BucketReportInputs(
        hits=hits, probabilities=space.probabilities, confident=confident_hits(space.probabilities, max_entropy),
        sample=mapped, numbered_per_bucket=config.space.map.numbered_per_bucket, attr_fn=attr_fn,
        cluster_report=f"../{CLUSTER_REPORT}/{INDEX_PAGE}", report_dir=dirs.buckets,
    )
    with timer.stage("bucket pages"):
        index = write_bucket_report(inputs, paged, config.space.buckets)
    return index, len(sample)


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str]:
    return {name: version(name) for name in ("torch", "numpy", "scipy", "networkx", "nnsight")}


def write_neuron_index(dirs: NeuronDirs, manifest: Manifest) -> Path:
    title = f"Neuron {manifest.neuron_idx}"
    body = render_template("space_neuron_index.html", title=title, m=manifest, clusters=CLUSTER_REPORT, buckets=BUCKET_REPORT)
    path = dirs.root / INDEX_PAGE
    path.write_text(page(title, body))
    return path


def write_run_index(out_dir: Path) -> Path:
    """Every neuron reported into `out_dir`, by this run or an earlier one."""
    manifests = [Manifest.model_validate_json(p.read_text()) for p in sorted(out_dir.glob(f"*/{MANIFEST}"))]
    title = f"Neuron reports in {out_dir.name}"
    entries = [(f"{m.collection.layer_name or m.collection.layer_idx}:{m.neuron_idx}", report_name(m.neuron_idx, m.polarity), m) for m in manifests]
    path = out_dir / INDEX_PAGE
    path.write_text(page(title, render_template("space_run_index.html", title=title, entries=entries)))
    return path


def report_neuron_space(
    config: RunConfig, run: PreparedRun, scan_once: ScanOnce, neuron_idx: int, polarity: Polarity,
) -> Manifest:
    """`scan_once` returns the scan, running it on the first call only, so a run that reuses every space never scans."""
    strategy = density_peaks_of(config.clustering)
    dirs = NeuronDirs(root=Path(config.out_dir) / report_name(neuron_idx, polarity))
    timer = StageTimer()
    wanted = collection_settings(config, neuron_idx, polarity)
    reused = space_to_reuse(dirs.space, wanted)
    if reused:
        print(f"[neuron {neuron_idx} {polarity}] reusing {dirs.space}")
    else:
        collect_space(config, run, scan_once(timer), neuron_idx, polarity, dirs.space, timer)

    space = load_space(dirs.space)
    buckets = all_buckets(space.probabilities, config.space.buckets)
    write_buckets_json(buckets, space, dirs.space / BUCKETS_JSON)
    paged = paged_buckets(buckets, config.space.buckets)
    print(f"[neuron {neuron_idx} {polarity}] {len(space.probabilities)} hits, {space.n_clusters} clusters, "
          f"{len(buckets)} buckets, {len(paged)} with a page")

    hits = space_cluster_hits(space, run, config, neuron_idx, strategy.max_entropy)
    with timer.stage("cluster report"):
        write_cluster_report(hits, space, paged, dirs, config, strategy.max_entropy, run.attr_fn)
    _, n_sampled = write_space_bucket_report(hits, space, paged, dirs, config, strategy.max_entropy, run.attr_fn, timer)

    confident = confident_hits(space.probabilities, strategy.max_entropy)
    manifest = Manifest(
        neuron_idx=neuron_idx, polarity=polarity, written_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=git_commit(), device=config.device, versions=package_versions(), collection=wanted,
        space_reused=reused, config=config.model_dump(mode="json"),
        counts={
            "hits": len(space.probabilities), "unique_samples": len(set(space.batch_idx.tolist())),
            "confident_hits": int(confident.sum()), "clusters": space.n_clusters, "buckets": len(buckets),
            "paged_buckets": len(paged), "sampled_on_map": n_sampled,
        },
        timings=timer.seconds,
    )
    (dirs.root / MANIFEST).write_text(manifest.model_dump_json(indent=2))
    print(f"[neuron {neuron_idx} {polarity}] -> {write_neuron_index(dirs, manifest)}")
    return manifest


def run_space_reports(config: RunConfig, run: PreparedRun) -> Path:
    """Scans at most once for every neuron and polarity of the run, and only when some space must be collected."""
    scans: list[ScanResult] = []

    def scan_once(timer: StageTimer) -> ScanResult:
        if not scans:
            with timer.stage("scan"):
                run.model.to(config.device)
                scans.append(run.capture.scan())
        return scans[0]

    for neuron_idx in config.neuron_idxs:
        for polarity in polarities_for(config):
            report_neuron_space(config, run, scan_once, neuron_idx, polarity)
    index = write_run_index(Path(config.out_dir))
    links = check_links(Path(config.out_dir))
    print(f"{links.pages} pages, {links.checked} links checked, {len(links.broken)} broken")
    if links.broken:
        raise RuntimeError(f"broken links, for example {[b.model_dump(mode='json') for b in links.broken[:5]]}")
    return index
