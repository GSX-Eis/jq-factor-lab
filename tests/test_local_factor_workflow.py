"""Execute the documented offline workflow through import, synthesis and build."""
import json
import pandas as pd
import pytest

from examples.local_factor_workflow import run_workflow


def test_synthetic_workflow_uses_price_labels_and_matching_deployment_formula(tmp_path):
    import sys
    aliases = ('factor_base', 'utils.factor_base', 'jq_adapter', 'utils.jq_adapter')
    missing = object()
    before = {name: sys.modules.get(name, missing) for name in aliases}
    result = run_workflow(tmp_path/'workflow')
    assert all(sys.modules.get(name, missing) is value for name, value in before.items())
    assert result['imported_rows'] == 80*12
    assert result['score_rows'] > result['evaluation_rows'] > 0
    labels = pd.read_csv(tmp_path/'workflow/labels.csv')
    source = pd.read_csv(tmp_path/'workflow/csv/daily/DEMO000.csv')
    first = labels[labels.stock_code == 'DEMO000'].iloc[0]
    assert first['1D'] == pytest.approx(source.close.iloc[1]/source.close.iloc[0]-1)
    assert first['5D'] == pytest.approx(source.close.iloc[5]/source.close.iloc[0]-1)
    assert (pd.Timestamp(result['bundle_previous_date']) + pd.offsets.BDay(1)
            == pd.Timestamp(result['bundle_execution_date']))
    assert json.loads((tmp_path/'workflow/summary.json').read_text()) == result
    with pytest.raises(FileExistsError):
        run_workflow(tmp_path/'workflow')
