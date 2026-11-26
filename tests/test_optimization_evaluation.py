"""Public answer contracts found during the full optimization evaluation."""

import pytest
from test_analysis import FixedModel

from csv_agent.analysis import analyze


@pytest.mark.docker
@pytest.mark.parametrize("alias", ["country", "Country"])
def test_declared_string_stays_a_string_when_description_mentions_a_number(alias):
    code = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
country = df.groupby('country')['cases'].mean().idxmax()
json.dump({'answer_fields': {'country_name': country}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"country,cases\nAlpha,1\nBeta,10\nBeta,20\n",
        "Which country has the highest average number of cases?",
        answer_format=(
            f'@country_name[{alias}] where "{alias}" is a string representing '
            "the name of the country with the highest average number of cases."
        ),
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == "@country_name[Beta]"
