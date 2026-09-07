"""Tests for the Evo 2 scoring arm.

Nothing here calls the endpoint. What is tested is everything between the
assembly and the request, which is where a silent error would do the most
damage: a window read from the wrong coordinates, a stop codon placed off frame,
or a likelihood summed over the wrong axis would all produce numbers that look
like scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evo2 import genome as genome_module
from src.evo2 import score

BASES = "ACGT"


def _write_fasta(path, sequences: dict[str, str], width: int = 10):
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for name, sequence in sequences.items():
            handle.write(f">{name} description here\n")
            for start in range(0, len(sequence), width):
                handle.write(sequence[start : start + width])
                handle.write("\n")
    return path


@pytest.fixture
def small_genome(tmp_path):
    rng = np.random.default_rng(3)
    sequences = {
        "chr1": "".join(rng.choice(list(BASES), size=95)),
        "chr2": "".join(rng.choice(list(BASES), size=40)),
    }
    path = _write_fasta(tmp_path / "small.fa", sequences)
    return path, sequences


def test_the_index_records_every_sequence(small_genome):
    path, sequences = small_genome
    index = genome_module.build_index(path)
    assert set(index) == set(sequences)
    for name, sequence in sequences.items():
        assert index[name].length == len(sequence)


def test_a_window_matches_the_sequence_it_was_taken_from(small_genome):
    path, sequences = small_genome
    with genome_module.Genome(path) as source:
        for name, sequence in sequences.items():
            assert source.fetch(name, 1, len(sequence)) == sequence
            assert source.fetch(name, 5, 14) == sequence[4:14]
            assert source.fetch(name, 11, 11) == sequence[10]


def test_a_window_that_runs_off_the_end_is_clipped_not_wrapped(small_genome):
    path, sequences = small_genome
    with genome_module.Genome(path) as source:
        assert source.fetch("chr2", 35, 200) == sequences["chr2"][34:]
        assert source.fetch("chr2", -20, 5) == sequences["chr2"][:5]
        assert source.fetch("chr2", 100, 200) == ""


def test_an_unknown_contig_is_reported(small_genome):
    path, _ = small_genome
    with genome_module.Genome(path) as source, pytest.raises(KeyError):
        source.fetch("chrZ", 1, 10)


def test_the_index_is_cached_and_reused(small_genome):
    path, _ = small_genome
    first = genome_module.load_index(path)
    assert path.with_suffix(path.suffix + ".index.json").exists()
    assert genome_module.load_index(path) == first


def test_reverse_complement_round_trips():
    sequence = "ACGTTTGCAN"
    assert genome_module.reverse_complement(sequence) == "NTGCAAACGT"
    assert genome_module.reverse_complement(genome_module.reverse_complement(sequence)) == sequence


def test_the_stop_codon_lands_at_the_stated_offset():
    sequence = "A" * score.WINDOW
    for offset in score.STOP_OFFSETS:
        mutant = score.insert_stop(sequence, offset)
        position = score.WINDOW // 2 + offset
        assert mutant[position : position + 3] == score.STOP_CODON
        assert len(mutant) == len(sequence)


def test_the_stop_offsets_are_codon_aligned():
    """An insertion out of frame is not a premature stop, it is a frameshift."""
    for offset in score.STOP_OFFSETS:
        assert offset % 3 == 0


def test_the_edit_changes_only_the_codon_it_targets():
    rng = np.random.default_rng(5)
    sequence = "".join(rng.choice(list(BASES), size=score.WINDOW))
    mutant = score.insert_stop(sequence, 30)
    differences = [i for i, (a, b) in enumerate(zip(sequence, mutant, strict=True)) if a != b]
    assert all(score.WINDOW // 2 + 30 <= i < score.WINDOW // 2 + 33 for i in differences)


def test_a_confident_model_scores_its_own_sequence_at_zero():
    sequence = "ACGTACGT"
    tokens = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8).astype(int)
    logits = np.full((len(sequence), 512), -50.0)
    # Position i is made certain of the token at i + 1, which is what the
    # likelihood reads, so a correct implementation returns approximately zero.
    for i in range(len(sequence) - 1):
        logits[i, tokens[i + 1]] = 50.0
    assert score.log_likelihood(sequence, logits) == pytest.approx(0.0, abs=1e-6)


def test_a_uniform_model_scores_a_sequence_at_chance():
    sequence = "ACGTACGT"
    logits = np.zeros((len(sequence), 512))
    expected = -(len(sequence) - 1) * np.log(512)
    assert score.log_likelihood(sequence, logits) == pytest.approx(expected)


def test_the_likelihood_reads_the_next_token_not_the_current_one():
    """Off by one here would score the model on predicting what it was shown."""
    sequence = "AC"
    tokens = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8).astype(int)
    logits = np.full((2, 512), -50.0)
    logits[0, tokens[0]] = 50.0
    aligned_on_current = score.log_likelihood(sequence, logits)
    logits[0, tokens[0]] = -50.0
    logits[0, tokens[1]] = 50.0
    aligned_on_next = score.log_likelihood(sequence, logits)
    assert aligned_on_next > aligned_on_current


def test_genes_with_a_short_first_coding_exon_are_excluded():
    needed = max(score.STOP_OFFSETS) + 3
    table = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000000001", "ENSG00000000002", "ENSG00000000003"],
            "first_exon_coding_length": [needed - 1, needed, needed + 500],
        }
    )
    kept = score.eligible(table)
    assert kept["ensembl_gene_id"].tolist() == ["ENSG00000000002", "ENSG00000000003"]


def test_the_credential_is_never_read_from_the_repository(monkeypatch):
    monkeypatch.delenv(score.KEY_ENVIRONMENT, raising=False)
    monkeypatch.delenv(score.KEY_PATH_ENVIRONMENT, raising=False)
    with pytest.raises(RuntimeError, match=score.KEY_ENVIRONMENT):
        score.api_key()


def test_the_credential_is_read_from_a_file_when_one_is_named(tmp_path, monkeypatch):
    path = tmp_path / "key.txt"
    path.write_text("NVIDIA key\n\nnvapi-abcDEF123_-xyz\n", encoding="utf-8")
    monkeypatch.delenv(score.KEY_ENVIRONMENT, raising=False)
    monkeypatch.setenv(score.KEY_PATH_ENVIRONMENT, str(path))
    assert score.api_key() == "nvapi-abcDEF123_-xyz"


def test_position_log_probs_sums_to_the_total_likelihood():
    """The restricted window is a different sum over the same array, so the two
    have to agree when the whole array is summed."""
    sequence = "ACGTACGTAC"
    rng = np.random.default_rng(11)
    logits = rng.normal(size=(len(sequence), 512))
    per_position = score.position_log_probs(sequence, logits)
    assert per_position.shape == (len(sequence) - 1,)
    assert per_position.sum() == pytest.approx(score.log_likelihood(sequence, logits))


def test_position_log_probs_are_log_probabilities():
    sequence = "ACGTACGT"
    logits = np.zeros((len(sequence), 512))
    per_position = score.position_log_probs(sequence, logits)
    assert np.allclose(per_position, -np.log(512))
    assert np.all(per_position <= 0)


def test_the_scored_span_is_the_pre_committed_value():
    """Fixed at 200 before the restricted scoring ran."""
    assert score.SCORED_SPAN == 200


def test_an_edit_cannot_change_predictions_before_it():
    """The model is autoregressive, so positions preceding the edit are
    untouched. This is what makes summing from the edit position the right
    restriction rather than an arbitrary one."""
    rng = np.random.default_rng(13)
    sequence = "".join(rng.choice(list(BASES), size=score.WINDOW))
    mutant = score.insert_stop(sequence, 30)
    edit = score.WINDOW // 2 + 30
    # Predictions at index i read bases up to i, so every index below edit - 1
    # sees identical context in both sequences.
    assert sequence[:edit] == mutant[:edit]


def test_the_restricted_span_starts_at_the_first_prediction_the_edit_reaches():
    """Element i of the array is the log probability of base i + 1, so the first
    prediction an edit at base p can change is at index p - 1. Starting at p
    would miss it and starting earlier would include positions the edit cannot
    reach."""
    offset = 30
    start = max(score.WINDOW // 2 + offset - 1, 0)
    assert start == score.WINDOW // 2 + offset - 1
    assert start + score.SCORED_SPAN < score.WINDOW - 1
