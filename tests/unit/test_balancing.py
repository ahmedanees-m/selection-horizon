"""Tests for the balancing-selection arm.

The released archives are 300 MB and are not in the repository, so what is
tested here is the reading and the aggregation, against archives built in the
test. Two of these tests exist because of defects the arm actually had: the
three population archives do not share a file naming scheme, and an empty
per-population summary used to arrive without columns and fail the merge that
combines populations with a missing-key error rather than a useful one.
"""

from __future__ import annotations

import io
import tarfile

import numpy as np
import pandas as pd
import pytest

from src.models import balancing


def _archive(path, members: dict[str, str]):
    with tarfile.open(path, "w:gz") as archive:
        for name, text in members.items():
            payload = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


SCORES = "Position\tBeta2\tBeta2_std\n100\t1.0\t0.5\n200\t4.0\t2.5\n300\t2.0\t1.0\n"


def test_both_released_naming_schemes_are_read():
    """CEU and CHB ship chrN_B2std.out; YRI ships chrN_B2.out."""
    assert balancing.SCORE_MEMBER.match("chr10_B2std.out").group(1) == "10"
    assert balancing.SCORE_MEMBER.match("chr10_B2.out").group(1) == "10"
    assert balancing.SCORE_MEMBER.match("chrX_B2std.out").group(1) == "X"


def test_a_file_that_is_not_a_score_file_is_not_read():
    assert balancing.SCORE_MEMBER.match("README_B2stdscores.txt") is None
    assert balancing.SCORE_MEMBER.match("chr10_B2std.out.gz") is None


def test_scores_are_read_from_either_layout(tmp_path):
    for name, member in (
        ("std.tar.gz", "StdB2Scores/chr21_B2std.out"),
        ("all.tar.gz", "AllB2StdScores/chr21_B2.out"),
    ):
        scores = balancing.read_scores(_archive(tmp_path / name, {member: SCORES}))
        assert set(scores) == {"21"}
        assert len(scores["21"]) == 3


def test_an_archive_with_no_score_files_is_reported(tmp_path):
    path = _archive(tmp_path / "empty.tar.gz", {"README.txt": "nothing here"})
    with pytest.raises(ValueError, match="no per-chromosome score files"):
        balancing.read_scores(path)


def test_a_release_without_the_standardised_column_is_refused(tmp_path):
    raw = "Position\tBeta2\n100\t1.0\n"
    path = _archive(tmp_path / "raw.tar.gz", {"chr21_B2.out": raw})
    with pytest.raises(ValueError, match="not the standardised statistic"):
        balancing.read_scores(path)


def test_the_summary_takes_the_maximum_over_sites_inside_the_span():
    spans = pd.DataFrame(
        [{"ensembl_gene_id": "ENSG00000000001", "chrom": "21", "start": 50, "end": 250}]
    )
    scores = {"21": pd.DataFrame({"Position": [100, 200, 300], "Beta2_std": [0.5, 2.5, 9.0]})}
    summary = balancing.aggregate(spans, scores, "CEU")
    assert len(summary) == 1
    assert summary["b2_max_CEU"].iloc[0] == pytest.approx(2.5)
    assert summary["b2_n_sites_CEU"].iloc[0] == 2


def test_a_gene_with_no_scored_site_is_dropped_rather_than_scored_as_zero():
    spans = pd.DataFrame(
        [{"ensembl_gene_id": "ENSG00000000002", "chrom": "21", "start": 400, "end": 500}]
    )
    scores = {"21": pd.DataFrame({"Position": [100, 200], "Beta2_std": [0.5, 2.5]})}
    assert balancing.aggregate(spans, scores, "CEU").empty


def test_an_empty_summary_still_carries_its_columns():
    """The merge across populations keys on the gene column, so an empty frame
    without columns fails with a missing key rather than an empty result."""
    spans = pd.DataFrame(columns=["ensembl_gene_id", "chrom", "start", "end"])
    summary = balancing.aggregate(spans, {}, "YRI")
    assert summary.empty
    assert "ensembl_gene_id" in summary.columns
    assert "b2_max_YRI" in summary.columns


def test_a_chromosome_with_no_scores_is_skipped_not_matched_elsewhere():
    spans = pd.DataFrame(
        [
            {"ensembl_gene_id": "ENSG00000000001", "chrom": "21", "start": 50, "end": 250},
            {"ensembl_gene_id": "ENSG00000000003", "chrom": "22", "start": 50, "end": 250},
        ]
    )
    scores = {"21": pd.DataFrame({"Position": [100, 200], "Beta2_std": [0.5, 2.5]})}
    summary = balancing.aggregate(spans, scores, "CEU")
    assert summary["ensembl_gene_id"].tolist() == ["ENSG00000000001"]


def test_the_missing_sites_do_not_leak_across_a_span_boundary():
    """searchsorted bounds are half open on the right, so a site exactly on the
    end coordinate belongs to the gene and one past it does not."""
    spans = pd.DataFrame(
        [{"ensembl_gene_id": "ENSG00000000001", "chrom": "21", "start": 100, "end": 200}]
    )
    scores = {
        "21": pd.DataFrame({"Position": [99, 100, 200, 201], "Beta2_std": [9.0, 1.0, 2.0, 9.0]})
    }
    summary = balancing.aggregate(spans, scores, "CEU")
    assert summary["b2_n_sites_CEU"].iloc[0] == 2
    assert summary["b2_max_CEU"].iloc[0] == pytest.approx(2.0)


def test_statistics_that_are_not_computed_carry_a_reason():
    assert balancing.NOT_COMPUTED
    for reason in balancing.NOT_COMPUTED.values():
        assert len(reason) > 30


def test_nan_scores_do_not_propagate_into_the_summary():
    spans = pd.DataFrame(
        [{"ensembl_gene_id": "ENSG00000000001", "chrom": "21", "start": 50, "end": 350}]
    )
    scores = {"21": pd.DataFrame({"Position": [100, 200, 300], "Beta2_std": [0.5, np.nan, 1.0]})}
    summary = balancing.aggregate(spans, scores, "CEU")
    assert summary["b2_max_CEU"].iloc[0] == pytest.approx(1.0)
