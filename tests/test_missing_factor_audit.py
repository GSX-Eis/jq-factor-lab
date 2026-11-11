import logging

import pandas as pd

from factor_synthesis.pipeline.lasso_method import _filter_by_missing_rate


def test_configured_but_absent_factor_is_reported_as_removed():
    filtered, removed, audit = _filter_by_missing_rate(
        pd.DataFrame({"present": [1.0, 2.0]}),
        0.1,
        logging.getLogger(__name__),
        configured_factor_names=["present", "absent"],
        return_audit=True,
    )

    assert list(filtered) == ["present"]
    assert removed == ["absent"]
    assert audit["missing_rate_by_factor"]["absent"] == 1.0
