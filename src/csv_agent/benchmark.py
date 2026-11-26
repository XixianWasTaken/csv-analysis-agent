"""Pinned public DABench data and disjoint, immutable sample manifests."""

import hashlib
import json
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Split = Literal["development", "final"]


def filename(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("Benchmark paths must be plain filenames")
    return value


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DatasetSource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    repository: str
    revision: str
    questions_file: str
    questions_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    labels_file: str
    labels_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    scope: str

    _filenames = field_validator("questions_file", "labels_file")(filename)


class ScorerSource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    repository: str
    revision: str
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    vendored_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class TableSource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=1)


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: int = Field(ge=0)
    file_name: str

    _filename = field_validator("file_name")(filename)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    dataset: DatasetSource
    scorer: ScorerSource
    tables: dict[str, TableSource]
    samples: list[Sample] = Field(min_length=1)
    splits: dict[Split, list[int]]

    @model_validator(mode="after")
    def validate_partition(self) -> "Manifest":
        ids = [sample.id for sample in self.samples]
        if len(set(ids)) != len(ids) or set(self.splits) != {"development", "final"}:
            raise ValueError("Duplicate samples or missing benchmark split")
        selected = [i for values in self.splits.values() for i in values]
        if len(selected) != len(set(selected)) or set(selected) != set(ids):
            raise ValueError("Splits must partition every sample exactly once")
        for name in self.tables:
            filename(name)
        files = {sample.id: sample.file_name for sample in self.samples}
        if set(files.values()) != set(self.tables):
            raise ValueError("Table inventory must match the sample manifest")
        dev_files = {files[i] for i in self.splits["development"]}
        final_files = {files[i] for i in self.splits["final"]}
        if dev_files & final_files:
            raise ValueError("Development and final splits share a data file")
        return self


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate_json(path.read_bytes())


def verified_bytes(path: Path, digest: str) -> bytes:
    data = path.read_bytes()
    if sha256(data) != digest:
        raise ValueError("Benchmark file checksum mismatch")
    return data


class Question(BaseModel):
    model_config = ConfigDict(strict=True)
    id: int
    file_name: str
    question: str = Field(min_length=1)
    constraints: str
    format: str = Field(min_length=1)


def load_questions(root: Path, manifest: Manifest) -> dict[int, Question]:
    source = manifest.dataset
    data = verified_bytes(root / source.questions_file, source.questions_sha256)
    questions = [
        Question.model_validate(json.loads(line))
        for line in data.splitlines()
        if line.strip()
    ]
    by_id = {q.id: q for q in questions}
    if len(by_id) != len(questions) or {q.id: q.file_name for q in questions} != {
        sample.id: sample.file_name for sample in manifest.samples
    }:
        raise ValueError("Questions do not match the locked sample manifest")
    return by_id


def prepare_dataset(root: Path, manifest_path: Path, *, split: Split) -> None:
    """Download only pinned inputs for the chosen split; never contact a model."""
    manifest = load_manifest(manifest_path)
    source = manifest.dataset
    if source.repository != "https://huggingface.co/datasets/infiagent/DABench":
        raise ValueError("Preparation requires the official DABench repository")
    if len(source.revision) != 40 or any(
        c not in "0123456789abcdef" for c in source.revision
    ):
        raise ValueError("Preparation requires a pinned dataset commit")
    root.mkdir(parents=True, exist_ok=True)
    selected = set(manifest.splits[split])
    files = {
        source.questions_file: source.questions_sha256,
        source.labels_file: source.labels_sha256,
        **{
            "da-dev-tables/" + sample.file_name: manifest.tables[
                sample.file_name
            ].sha256
            for sample in manifest.samples
            if sample.id in selected
        },
    }
    with httpx.Client(timeout=60, follow_redirects=True, trust_env=False) as client:
        for name, digest in files.items():
            target = root / name
            if target.is_file() and sha256(target.read_bytes()) == digest:
                continue
            response = client.get(
                f"{source.repository}/resolve/{source.revision}/{quote(name, safe='/')}"
            )
            response.raise_for_status()
            if sha256(response.content) != digest:
                raise ValueError("Downloaded benchmark file checksum mismatch")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(response.content)
    load_questions(root, manifest)
