"""Tests for the benchmark-rule scoring, which needs no endpoint.

The scramble is the whole of the perturbation, so what has to hold is that it
preserves dinucleotide composition, keeps the ends fixed and is not the identity.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

from src.evo2 import benchmark


def pairs(sequence: str) -> collections.Counter:
    return collections.Counter(a + b for a, b in zip(sequence, sequence[1:], strict=False))


@pytest.mark.parametrize("seed", range(6))
def test_the_scramble_preserves_dinucleotide_composition(seed):
    rng = np.random.default_rng(seed)
    sequence = "".join(rng.choice(list("ACGT"), size=600))
    shuffled = benchmark.dinucleotide_shuffle(sequence, np.random.default_rng(seed + 100))
    assert pairs(shuffled) == pairs(sequence)
    assert collections.Counter(shuffled) == collections.Counter(sequence)
    assert len(shuffled) == len(sequence)


def test_the_scramble_keeps_the_first_and_last_base():
    rng = np.random.default_rng(3)
    sequence = "".join(rng.choice(list("ACGT"), size=400))
    shuffled = benchmark.dinucleotide_shuffle(sequence, np.random.default_rng(9))
    assert shuffled[0] == sequence[0]
    assert shuffled[-1] == sequence[-1]


def test_the_scramble_actually_moves_a_long_sequence():
    rng = np.random.default_rng(5)
    sequence = "".join(rng.choice(list("ACGT"), size=2000))
    shuffled = benchmark.dinucleotide_shuffle(sequence, np.random.default_rng(11))
    assert shuffled != sequence


def test_a_short_sequence_is_returned_unchanged():
    assert benchmark.dinucleotide_shuffle("AC", np.random.default_rng(0)) == "AC"


def test_a_homopolymer_has_only_one_euler_path():
    sequence = "A" * 50
    assert benchmark.dinucleotide_shuffle(sequence, np.random.default_rng(1)) == sequence


def test_an_unusable_sequence_is_reported_rather_than_scored():
    record = benchmark.score_gene("ENSG0", "ACGTNACGT", key="unused", seed=1)
    assert record["note"] == "sequence not usable"
    assert "difference" not in record


def test_a_gene_is_scored_from_the_two_likelihoods(monkeypatch):
    calls = []

    def fake_forward(sequence, key):
        calls.append(sequence)
        return np.zeros((len(sequence), 128))

    def fake_log_likelihood(sequence, logits):
        # Real sequence scores higher than its scramble, as an essential gene would.
        return 10.0 if sequence == calls[0] else 4.0

    monkeypatch.setattr(benchmark, "forward", fake_forward)
    monkeypatch.setattr(benchmark, "log_likelihood", fake_log_likelihood)

    rng = np.random.default_rng(2)
    sequence = "".join(rng.choice(list("ACGT"), size=300))
    record = benchmark.score_gene("ENSG1", sequence, key="unused", seed=7)

    assert len(calls) == 2
    assert calls[0] == sequence
    assert pairs(calls[1]) == pairs(sequence)
    assert record["difference"] == pytest.approx(6.0)
    assert record["difference_per_base"] == pytest.approx(6.0 / 300)
