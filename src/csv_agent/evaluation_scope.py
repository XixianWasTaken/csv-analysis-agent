"""Validate and describe a diagnostic batch against its immutable data manifest."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from csv_agent.benchmark import Manifest, Sample, Split, TableSource, load_manifest

Purpose = Literal["optimization", "holdout"]


class FileGroup(BaseModel):
    selected_ids: list[int]
    available_ids: list[int]
    complete: bool
    source: TableSource


class EvaluationSelection(BaseModel):
    purpose: Purpose
    sample_ids: list[int]
    samples: list[Sample]
    total: int
    file_groups: dict[str, FileGroup]


def select_samples(
    manifest: Manifest,
    split: Split,
    sample_ids: list[int] | None,
    purpose: Purpose,
) -> EvaluationSelection:
    selected = list(manifest.splits[split] if sample_ids is None else sample_ids)
    if not selected or any(type(i) is not int for i in selected):
        raise ValueError("Selection requires a nonempty list of integer sample IDs")
    if len(set(selected)) != len(selected):
        raise ValueError("Selection contains duplicate sample IDs")
    if set(selected) - set(manifest.splits[split]):
        raise ValueError("Selected sample IDs must belong to the locked split")
    if purpose not in ("optimization", "holdout"):
        raise ValueError("Unknown evaluation purpose")
    by_id = {sample.id: sample for sample in manifest.samples}
    samples = [by_id[i] for i in selected]
    groups = {}
    for sample in samples:
        if sample.file_name in groups:
            continue
        chosen = [s.id for s in samples if s.file_name == sample.file_name]
        available = [s.id for s in manifest.samples if s.file_name == sample.file_name]
        groups[sample.file_name] = FileGroup(
            selected_ids=chosen,
            available_ids=available,
            complete=set(chosen) == set(available),
            source=manifest.tables[sample.file_name],
        )
    if purpose == "holdout":
        if split != "final":
            raise ValueError(
                "Holdout evaluation requires the file-isolated final split"
            )
        if any(not group.complete for group in groups.values()):
            raise ValueError("Holdout selection must include complete file groups")
        # ADR 0001: the original public final split has already been used to tune
        # this project. A different manifest or filename cannot undo that use.
        original = load_manifest(
            Path(__file__).resolve().parents[2] / "benchmarks/dabench/manifest.json"
        )
        optimized_ids = set(original.splits["final"])
        optimized_hashes = {
            original.tables[s.file_name].sha256
            for s in original.samples
            if s.id in optimized_ids
        }
        if any(group.source.sha256 in optimized_hashes for group in groups.values()):
            raise ValueError("These files already belong to the optimization set")
    return EvaluationSelection(
        purpose=purpose,
        sample_ids=selected,
        samples=samples,
        total=len(selected),
        file_groups=groups,
    )
