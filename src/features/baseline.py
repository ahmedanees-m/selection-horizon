"""The non-evolutionary baseline feature matrix.

This is the M0 comparator in primary estimand 2. Its job is to be a fair
alternative to the evolutionary scores, which means two things: it has to be
informative, and it must not contain evolutionary information under another
name.

That second requirement rules out the STRING channel of the Open Targets
interaction dataset. STRING's combined score aggregates a phylogenetic profile
and a genomic-neighbourhood channel, both computed across genomes, so a degree
derived from it carries comparative-genomics signal. A baseline built on it
would not be a non-evolutionary baseline. The network features are therefore
built from the experimentally determined and curated sources only: IntAct,
Signor and Reactome.

Features produced per gene:

    ppi_degree                  neighbours in the curated interaction graph
    ppi_betweenness             sampled betweenness centrality on the same graph
    gtex_mean_expression        mean expression across tissues
    expression_breadth_tau      tissue-specificity index, 0 ubiquitous to 1 specific
    depmap_gene_effect          mean CRISPR gene effect across cell lines
    depmap_common_essential     flagged essential in the upstream release
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.ingest import open_targets

# Sources whose evidence is experimental or curated from human pathway
# annotation. STRING is excluded; see the module docstring.
CURATED_SOURCES = ("intact", "signor", "reactome")

BETWEENNESS_PIVOTS = 500

# The Platform ships four expression datasources on incompatible units. GTEx is
# the bulk-tissue set the plan names.
EXPRESSION_SOURCE = "gtex"


def interaction_edges(sources: tuple[str, ...] = CURATED_SOURCES) -> pd.DataFrame:
    """Undirected human interaction edges from the curated sources."""
    frame = open_targets.read(
        "interaction",
        columns=["sourceDatabase", "targetA", "targetB", "speciesA", "speciesB"],
    )
    frame = frame[frame["sourceDatabase"].isin(sources)]
    frame = frame.dropna(subset=["targetA", "targetB"])

    def taxon(value) -> int | None:
        return value["taxonId"] if isinstance(value, dict) else None

    human = (frame["speciesA"].map(taxon) == 9606) & (frame["speciesB"].map(taxon) == 9606)
    frame = frame[human]
    frame = frame[frame["targetA"] != frame["targetB"]]

    # The species fields do not catch everything: curated sources carry
    # homology-inferred edges whose partners are non-human Ensembl genes, and
    # those arrive with a version suffix that human Platform targets do not
    # have. Requiring the human unversioned shape on both sides is explicit and
    # cheap, where relying on the species annotation alone is neither.
    shape = r"^ENSG\d{11}$"
    frame = frame[
        frame["targetA"].str.match(shape, na=False) & frame["targetB"].str.match(shape, na=False)
    ]

    low = np.minimum(frame["targetA"].to_numpy(), frame["targetB"].to_numpy())
    high = np.maximum(frame["targetA"].to_numpy(), frame["targetB"].to_numpy())
    edges = pd.DataFrame({"source": low, "target": high}).drop_duplicates()
    return edges.reset_index(drop=True)


def _adjacency(edges: pd.DataFrame) -> tuple[list[str], list[np.ndarray]]:
    """Compact adjacency lists keyed by node index."""
    nodes = pd.unique(pd.concat([edges["source"], edges["target"]], ignore_index=True))
    index = {node: position for position, node in enumerate(nodes)}

    left = edges["source"].map(index).to_numpy()
    right = edges["target"].map(index).to_numpy()

    both_from = np.concatenate([left, right])
    both_to = np.concatenate([right, left])
    order = np.argsort(both_from, kind="stable")
    both_from, both_to = both_from[order], both_to[order]
    boundaries = np.searchsorted(both_from, np.arange(len(nodes) + 1))
    neighbours = [both_to[boundaries[i] : boundaries[i + 1]] for i in range(len(nodes))]
    return list(nodes), neighbours


def sampled_betweenness(
    nodes: list[str], neighbours: list[np.ndarray], pivots: int, seed: int
) -> np.ndarray:
    """Brandes betweenness accumulated from a random sample of source nodes.

    Exact betweenness is O(VE) and not affordable on a graph this size. Sampling
    sources and rescaling gives an unbiased estimate; the pivot count is
    recorded so the precision is auditable rather than implied.
    """
    n = len(nodes)
    rng = np.random.default_rng(seed)
    sources = rng.choice(n, size=min(pivots, n), replace=False)
    centrality = np.zeros(n, dtype=float)

    for source in sources:
        stack: list[int] = []
        predecessors: list[list[int]] = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=float)
        distance = np.full(n, -1, dtype=np.int64)
        sigma[source] = 1.0
        distance[source] = 0

        queue = [int(source)]
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            stack.append(current)
            for neighbour in neighbours[current]:
                neighbour = int(neighbour)
                if distance[neighbour] < 0:
                    distance[neighbour] = distance[current] + 1
                    queue.append(neighbour)
                if distance[neighbour] == distance[current] + 1:
                    sigma[neighbour] += sigma[current]
                    predecessors[neighbour].append(current)

        delta = np.zeros(n, dtype=float)
        for node in reversed(stack):
            for predecessor in predecessors[node]:
                delta[predecessor] += (sigma[predecessor] / sigma[node]) * (1.0 + delta[node])
            if node != source:
                centrality[node] += delta[node]

    scale = n / len(sources) if len(sources) else 0.0
    return centrality * scale / 2.0


def network_features(seed: int, pivots: int = BETWEENNESS_PIVOTS) -> pd.DataFrame:
    edges = interaction_edges()
    nodes, neighbours = _adjacency(edges)
    degree = np.array([len(block) for block in neighbours], dtype=float)
    betweenness = sampled_betweenness(nodes, neighbours, pivots, seed)

    frame = pd.DataFrame(
        {
            "ensembl_gene_id": nodes,
            "ppi_degree": degree,
            "ppi_betweenness": betweenness,
        }
    )
    frame.attrs["n_edges"] = len(edges)
    frame.attrs["n_pivots"] = min(pivots, len(nodes))
    return frame


def tau(values: np.ndarray) -> float:
    """Tissue-specificity index.

    tau = sum_i (1 - x_i / max x) / (n - 1)

    Zero for a gene expressed evenly everywhere, one for a gene expressed in a
    single tissue. Undefined when a gene is not expressed at all.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float("nan")
    peak = finite.max()
    if peak <= 0:
        return float("nan")
    return float(np.sum(1.0 - finite / peak) / (finite.size - 1))


def expression_features(datasource: str = EXPRESSION_SOURCE) -> pd.DataFrame:
    """Mean expression and tissue specificity, from GTEx bulk tissue medians.

    The Platform ships four expression datasources in release 26.06. GTEx is the
    one the plan names and the only bulk-tissue set among them: Tabula Sapiens is
    single-cell pseudobulk in counts per million, DICE carries no tissue
    identifier, and PRIDE is proteomic intensity. Mixing units across those would
    make a mean that means nothing, so one source is used and named.

    Expression is summarised on the log scale, because TPM spans five orders of
    magnitude and an arithmetic mean across tissues would be the top one or two
    tissues and nothing else.
    """
    frames = []
    for part in sorted(open_targets.dataset_dir("baseline_expression").glob("*.parquet")):
        chunk = pd.read_parquet(
            part, columns=["targetId", "datasourceId", "tissueBiosampleId", "median"]
        )
        chunk = chunk[chunk["datasourceId"] == datasource]
        if not chunk.empty:
            frames.append(chunk.drop(columns="datasourceId"))
    if not frames:
        raise ValueError(f"no {datasource} rows in the baseline expression dataset")

    long = pd.concat(frames, ignore_index=True).dropna(subset=["targetId", "median"])
    long = long.drop_duplicates(["targetId", "tissueBiosampleId"])
    long["log_expression"] = np.log1p(long["median"].clip(lower=0))

    grouped = long.groupby("targetId")["log_expression"]
    table = pd.DataFrame(
        {
            "ensembl_gene_id": grouped.mean().index,
            "gtex_mean_expression": grouped.mean().to_numpy(),
            "gtex_max_expression": grouped.max().to_numpy(),
            "n_tissues": grouped.size().to_numpy(),
        }
    )
    table["expression_breadth_tau"] = grouped.apply(
        lambda values: tau(values.to_numpy())
    ).to_numpy()
    return table.reset_index(drop=True)


def _sequence(value) -> list:
    """Nested Platform fields arrive as numpy arrays, where `or []` is ambiguous."""
    if value is None:
        return []
    return list(value)


def essentiality_features() -> pd.DataFrame:
    """DepMap gene effect and the essential flag from the Platform target essentiality."""
    frame = open_targets.read("target_essentiality")
    rows = []
    for gene, record in zip(frame["id"], frame["geneEssentiality"], strict=True):
        effects: list[float] = []
        essential = False
        for entry in _sequence(record):
            if not isinstance(entry, dict):
                continue
            essential = essential or bool(entry.get("isEssential"))
            for screen in _sequence(entry.get("depMapEssentiality")):
                if not isinstance(screen, dict):
                    continue
                for line in _sequence(screen.get("screens")):
                    if isinstance(line, dict) and line.get("geneEffect") is not None:
                        effects.append(float(line["geneEffect"]))
        rows.append(
            {
                "ensembl_gene_id": gene,
                "depmap_gene_effect": float(np.mean(effects)) if effects else np.nan,
                "depmap_n_lines": len(effects),
                "depmap_common_essential": essential,
            }
        )
    return pd.DataFrame(rows)


def build(seed: int | None = None) -> pd.DataFrame:
    seed = seed if seed is not None else config.analysis()["seed"]

    network = network_features(seed)
    expression = expression_features()
    essentiality = essentiality_features()

    table = network.merge(expression, on="ensembl_gene_id", how="outer").merge(
        essentiality, on="ensembl_gene_id", how="outer"
    )
    table.attrs.update(network.attrs)
    return table.sort_values("ensembl_gene_id").reset_index(drop=True)


def run() -> pd.DataFrame:
    table = build()
    target = config.interim_dir() / "baseline_features.parquet"
    table.to_parquet(target, index=False)
    print(
        f"baseline_features: {len(table):,} genes from "
        f"{table.attrs.get('n_edges', 0):,} curated interaction edges, "
        f"betweenness over {table.attrs.get('n_pivots', 0)} pivots"
    )
    for column in table.columns:
        if column == "ensembl_gene_id":
            continue
        print(f"   {column}: {int(table[column].notna().sum()):,} non-null")
    return table


def load() -> pd.DataFrame:
    return pd.read_parquet(config.interim_dir() / "baseline_features.parquet")


if __name__ == "__main__":
    config.ensure_dirs()
    run()
