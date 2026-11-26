"""Controlled instructions for repeatable UI acceptance; computation stays in Docker."""

import json

from csv_agent.model import ModelReply

CSV = b"region,sales\neast,10\nwest,20\neast,15\n"
CASES = {
    "Aggregation: total sales": "What are total sales in CNY?",
    "Repair: recover from initial failure": "What are total sales in CNY?",
    "Stop: execution limit reached": "What are total sales in CNY?",
    "Insufficient input: missing cost": "What is the sales profit?",
    "Filter: east sales": "Calculate total east sales in CNY.",
    "Chart: sales by region": "Summarize sales by region in CNY with a table and bar chart.",
    "Explore data and submit": "Inspect categories, types and missing values, then calculate total sales in CNY.",
    "Review: recompute an incorrect result": "Calculate total east sales in CNY; review the filter.",
    "Disagreement: investigate and select": "Calculate east sales and valid row count; investigate disagreements and explain the selection.",
}
INSPECTION_CODE = """
import json, pandas as pd
df = pd.read_csv('/input/data.csv')
json.dump({'observations': {
    'sales_dtype': str(df['sales'].dtype),
    'sales_missing': int(df['sales'].isna().sum()),
    'regions': sorted(df['region'].unique().tolist()),
}}, open('/output/result.json', 'w'))
"""
TOTAL_CODE = """
import json, pandas as pd
df = pd.read_csv('/input/data.csv')
total = int(df['sales'].sum())
json.dump({'answer': f'Total sales {total} CNY',
           'table': [{'total_sales_cny': total}]}, open('/output/result.json', 'w'))
"""
FILTER_CODE = """
import json, pandas as pd
df = pd.read_csv('/input/data.csv')
total = int(df.loc[df['region'] == 'east', 'sales'].sum())
json.dump({'answer': f'East sales {total} CNY',
           'table': [{'region': 'east', 'sales_cny': total}]}, open('/output/result.json', 'w'))
"""
RECOMPUTE_FILTER_CODE = """
import csv, json
with open('/input/data.csv') as source:
    rows = list(csv.DictReader(source))
total = sum(int(row['sales']) for row in rows if row['region'] == 'east')
json.dump({'answer': f'East sales {total} CNY',
           'table': [{'region': 'east', 'sales_cny': total}]}, open('/output/result.json', 'w'))
"""
DISAGREEMENT_CODE = """
import json, pandas as pd
df = pd.read_csv('/input/data.csv')
json.dump({'answer_fields': {'total': int(df['sales'].sum()),
                            'count': int(df['sales'].count())}},
          open('/output/result.json', 'w'))
"""
INDEPENDENT_FIELDS_CODE = """
import csv, json
with open('/input/data.csv') as source:
    values = [int(row['sales']) for row in csv.DictReader(source)
              if row['region'] == 'east' and row['sales']]
json.dump({'answer_fields': {'total': sum(values), 'count': len(values)}},
          open('/output/result.json', 'w'))
"""
CHECK_FILTER_CODE = """
import csv, json
with open('/input/data.csv') as source:
    values = [int(row['sales']) for row in csv.DictReader(source)
              if row['region'] == 'east' and row['sales']]
json.dump({'observations': {'east_values': values, 'east_total': sum(values),
                          'east_present_count': len(values)}},
          open('/output/result.json', 'w'))
"""
CHART_CODE = """
import json, pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('/input/data.csv')
totals = df.groupby('region', sort=True)['sales'].sum()
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(totals.index, totals.values)
ax.set(xlabel='Region', ylabel='Sales (CNY)', title='Total sales by region')
fig.tight_layout()
fig.savefig('/output/comparison.png', dpi=100)
plt.close(fig)
table = [{'region': region, 'sales_cny': int(value)} for region, value in totals.items()]
json.dump({'answer': '; '.join(f'{region}: {value} CNY' for region, value in totals.items()),
           'table': table, 'charts': [{'path': 'comparison.png', 'caption': 'Total sales by region (CNY)'}]},
          open('/output/result.json', 'w'))
"""


class DemoModel:
    """A model-boundary fixture, never a real provider or token measurement."""

    def __init__(self, case: str):
        if case not in CASES:
            raise ValueError("Unknown demo case")
        self.case = case
        self.calls = 0

    def generate(self, prompt: str, *, timeout_seconds: float) -> ModelReply:
        self.calls += 1
        if self.case in {"Review: recompute an incorrect result", "Disagreement: investigate and select"}:
            checks = {
                "field_meaning": "sales is measured in CNY; region identifies the area.",
                "filters_and_groups": "Only east rows 10 and 15 are included, totaling 25 CNY.",
                "missing_values": "Neither east row has missing sales.",
                "statistical_definition": "Sum sales rather than count or average rows.",
                "algorithm_parameters": "Filter CSV rows and add integers; no fitted parameters.",
                "precision": "Keep integer values without rounding.",
            }
            if self.case == "Disagreement: investigate and select":
                resolution_actions: list[dict[str, object]] = [
                    {"action": "candidate", "code": DISAGREEMENT_CODE},
                    {
                        "action": "recompute",
                        "candidate": 1,
                        "code": INDEPENDENT_FIELDS_CODE,
                        "reason": "Check the east filter against the actual data.",
                        "method": "Filter rows with csv.DictReader instead of pandas aggregation.",
                    },
                    {
                        "action": "review",
                        "candidate": 2,
                        "outcome": "passed",
                        "checks": checks,
                    },
                    {"action": "inspect", "code": CHECK_FILTER_CODE},
                    {
                        "action": "select",
                        "candidate": 2,
                        "reason": "Check the east filter against the actual data.",
                        "evidence": [2, 4],
                        "resolutions": {
                            "total": {
                                "finding": "East sales total 25; the initial total includes west.",
                                "evidence": [4],
                            },
                            "count": {
                                "finding": "East has two valid rows; the initial count includes west.",
                                "evidence": [4],
                            },
                        },
                    },
                ]
                return ModelReply(json.dumps(resolution_actions[self.calls - 1]), None)
            verification_actions: list[dict[str, object]] = [
                {"action": "candidate", "code": TOTAL_CODE},
                {
                    "action": "review",
                    "candidate": 1,
                    "outcome": "needs_revision",
                    "checks": {
                        **checks,
                        "filters_and_groups": "Only east rows 10 and 15 are included, totaling 25 CNY.",
                    },
                },
                {"action": "recompute", "candidate": 1, "code": RECOMPUTE_FILTER_CODE},
                {
                    "action": "review",
                    "candidate": 3,
                    "outcome": "passed",
                    "checks": checks,
                },
                {"action": "submit", "candidate": 3},
            ]
            return ModelReply(json.dumps(verification_actions[self.calls - 1]), None)
        if self.case == "Explore data and submit":
            actions: list[dict[str, object]] = [
                {"action": "inspect", "code": INSPECTION_CODE},
                {"action": "candidate", "code": TOTAL_CODE},
                {"action": "submit", "candidate": 2},
            ]
            return ModelReply(json.dumps(actions[self.calls - 1]), None)
        if self.case == "Insufficient input: missing cost":
            return ModelReply(
                json.dumps(
                    {
                        "status": "insufficient_input",
                        "reason": "Missing cost field; cannot calculate profit.",
                    }
                ),
                None,
            )
        if self.case == "Filter: east sales":
            return ModelReply(json.dumps({"code": FILTER_CODE}), None)
        if self.case == "Chart: sales by region":
            return ModelReply(json.dumps({"code": CHART_CODE}), None)
        if self.case == "Stop: execution limit reached":
            return ModelReply(
                json.dumps({"code": "raise ValueError('demo-always-fails')"}), None
            )
        if self.case == "Repair: recover from initial failure" and self.calls == 1:
            return ModelReply(
                json.dumps({"code": "raise ValueError('demo-first-attempt')"}), None
            )
        return ModelReply(json.dumps({"code": TOTAL_CODE}), None)
