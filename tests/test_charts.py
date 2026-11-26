import base64
import io
import json

import pytest
from PIL import Image
from test_analysis import AGGREGATE, FixedModel
from test_repair import SequenceModel

from csv_agent.analysis import analyze
from csv_agent.config import RunLimits

COMPARISON = """
import json
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('/input/data.csv')
totals = df.groupby('region')['sales'].sum().sort_index()
fig, ax = plt.subplots()
totals.plot.bar(ax=ax)
ax.set_xlabel('Region')
ax.set_ylabel('Sales (CNY)')
ax.set_title('Total sales by region')
fig.tight_layout()
fig.savefig('/output/comparison.png')
json.dump({
    'answer': 'east: 25 CNY; west: 20 CNY',
    'table': [{'region': region, 'sales_cny': int(total)} for region, total in totals.items()],
    'charts': [{'path': 'comparison.png', 'caption': 'Total sales by region (CNY)'}],
}, open('/output/result.json', 'w'))
"""


@pytest.mark.docker
def test_group_comparison_returns_verifiable_table_and_readable_chart():
    result = analyze(
        b"region,sales\neast,10\nwest,20\neast,15\n",
        "Compare total sales by region in CNY with a chart",
        model=FixedModel(COMPARISON),
    )
    assert result.status == "completed", result.error
    assert result.table == [
        {"region": "east", "sales_cny": 25},
        {"region": "west", "sales_cny": 20},
    ]
    assert len(result.charts) == 1
    chart = result.charts[0]
    assert chart.path == "comparison.png"
    assert chart.caption == "Total sales by region (CNY)"
    with Image.open(io.BytesIO(base64.b64decode(chart.data_base64))) as image:
        assert image.format == "PNG"
        assert image.width > 100 and image.height > 100
        image.load()
    assert result.code == COMPARISON
    assert result.execution.exit_code == 0
    assert result.attempts[-1].code == COMPARISON
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored.charts == result.charts

    following = analyze(b"sales\n10\n", "sum", model=FixedModel(AGGREGATE))
    assert following.status == "completed", following.error
    assert following.charts == []


@pytest.mark.docker
@pytest.mark.parametrize(
    ("path", "setup"),
    [
        ("/tmp/chart.png", "Path('/tmp/chart.png').write_bytes(b'outside')"),
        ("../tmp/chart.png", "Path('/tmp/chart.png').write_bytes(b'outside')"),
        ("escape.png", "Path('/output/escape.png').symlink_to('/input/data.csv')"),
        (
            "patched.png",
            "import os; from PIL import Image; Image.new('RGB', (10, 10)).save('/tmp/outside.png'); Path('/output/patched.png').symlink_to('/tmp/outside.png'); os.O_NOFOLLOW = 0",
        ),
        ("directory.png", "Path('/output/directory.png').mkdir()"),
        ("pipe.png", "import os; os.mkfifo('/output/pipe.png')"),
        ("missing.png", "pass"),
        ("invalid.png", "Path('/output/invalid.png').write_bytes(b'not a PNG')"),
        ("large.png", "Path('/output/large.png').write_bytes(b'x' * 524289)"),
    ],
)
def test_unauthorized_or_unreadable_charts_fail_without_publishing_results(path, setup):
    code = f"""
import json
from pathlib import Path
{setup}
json.dump({{'answer': 'must not publish', 'charts': [{{'path': {path!r}, 'caption': 'test'}}]}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n",
        "chart",
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "failed", result.model_dump_json()
    assert result.answer is None
    assert result.table == []
    assert result.charts == []
    assert result.error


@pytest.mark.docker
def test_chart_collection_does_not_reuse_previous_task_artifacts():
    first = analyze(
        b"region,sales\neast,10\n",
        "chart",
        model=FixedModel(COMPARISON),
    )
    assert first.status == "completed", first.error
    code = """
import json
json.dump({'answer': 'stale', 'charts': [{'path': 'comparison.png', 'caption': 'stale'}]}, open('/output/result.json', 'w'))
"""
    second = analyze(
        b"sales\n10\n",
        "chart",
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert second.run_id != first.run_id
    assert second.status == "failed"
    assert second.charts == []


@pytest.mark.docker
@pytest.mark.parametrize("case", ["dimensions", "count", "total_bytes"])
def test_chart_dimensions_count_and_total_size_are_bounded(case):
    code = f"""
import json
from pathlib import Path
from PIL import Image
case = {case!r}
charts = []
for index in range(5 if case == 'count' else 2 if case == 'total_bytes' else 1):
    name = f'chart-{{index}}.png'
    path = Path('/output') / name
    Image.new('RGB', (4097 if case == 'dimensions' else 10, 1)).save(path)
    if case == 'total_bytes':
        with path.open('ab') as handle:
            handle.write(b'x' * 270000)
    charts.append({{'path': name, 'caption': 'test'}})
json.dump({{'answer': 'must not publish', 'charts': charts}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n",
        "chart",
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "failed", result.model_dump_json()
    assert result.answer is None
    assert result.charts == []


@pytest.mark.docker
def test_invalid_chart_is_repaired_using_execution_feedback():
    invalid = COMPARISON.replace(
        "fig.savefig('/output/comparison.png')",
        "open('/output/comparison.png', 'wb').write(b'broken image')",
    )
    model = SequenceModel(
        [json.dumps({"code": invalid}), json.dumps({"code": COMPARISON})]
    )
    result = analyze(
        b"region,sales\neast,10\nwest,20\neast,15\n",
        "Compare sales in CNY",
        model=model,
    )
    assert result.status == "completed", result.error
    assert result.execution_attempts == 2
    assert "chart artifacts" in result.attempts[0].error
    assert result.table == [
        {"region": "east", "sales_cny": 25},
        {"region": "west", "sales_cny": 20},
    ]
    assert len(result.charts) == 1
    assert result.code == COMPARISON
