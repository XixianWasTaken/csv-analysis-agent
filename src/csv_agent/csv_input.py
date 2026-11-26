"""CSV validation and the data overview shared by every analysis entry point."""

import csv
import hashlib
import io

import pandas as pd


def inspect_csv(data: bytes) -> dict[str, object]:
    if b"\x00" in data:
        raise ValueError("Invalid CSV: NUL bytes are not supported")
    reader = csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
    # csv.reader represents a literally empty record as [], although it can be
    # the empty header field of a one-column CSV. Row widths below still apply.
    original_columns = next(reader) or [""]
    named_columns = [column for column in original_columns if column.strip()]
    if len(set(named_columns)) != len(named_columns):
        raise ValueError("CSV requires unique named column names")
    used_names = set(named_columns)
    columns = []
    for index, original in enumerate(original_columns):
        name = original
        if not original.strip():
            base = f"Unnamed: {index}"
            name = base
            suffix = 1
            while name in used_names:
                name = f"{base}.{suffix}"
                suffix += 1
        used_names.add(name)
        columns.append(name)
    rows = 0
    for row in reader:
        if not row:
            continue
        if len(row) != len(columns):
            raise ValueError("CSV rows have inconsistent field counts")
        rows += 1
    if rows == 0:
        raise ValueError("CSV contains no data rows")
    # Skip one CSV record explicitly: a whitespace-only header would otherwise
    # be skipped as a blank line and pandas would consume the first data row.
    read_csv_options = {"header": None, "names": columns, "skiprows": 1}
    frame = pd.read_csv(io.BytesIO(data), **read_csv_options)
    if len(frame) == 0:
        raise ValueError("CSV contains no data rows")
    return {
        "columns": columns,
        "column_mapping": [
            {"position": index + 1, "original": original, "name": name}
            for index, (original, name) in enumerate(zip(original_columns, columns))
        ],
        "column_note": "Positions are 1-based. Empty or whitespace-only headers receive positional Unnamed: N identifiers (N is 0-based), with numeric suffixes to avoid all named columns. Keep every column; an unnamed column is not necessarily an index.",
        "read_csv_options": read_csv_options,
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "rows": len(frame),
        "types": {column: str(frame[column].dtype) for column in columns},
        "missing": {column: int(frame[column].isna().sum()) for column in columns},
        "sample": [
            [str(cell)[:200] for cell in row]
            for row in frame.head(5)
            .astype(object)
            .fillna("")
            .itertuples(index=False, name=None)
        ],
        "sample_note": "At most 5 rows; each cell truncated to 200 characters. Missing values and types follow pandas.read_csv defaults.",
    }
