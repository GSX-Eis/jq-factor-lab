from __future__ import annotations

from scripts.migrate_symbolic_eval_to_namespace import (
    render_namespace_expression,
    rewrite_source,
)


def test_render_namespace_expression_supports_nested_symbolic_calls() -> None:
    rendered = render_namespace_expression(
        "cs_rank(ts_mean(features['daily_close'], 2))"
    )

    assert (
        rendered
        == "namespace['cs_rank'](namespace['ts_mean'](features['daily_close'], 2))"
    )


def test_rewrite_source_replaces_expression_code_eval() -> None:
    source = """
EXPRESSION_CODE = "add(features['daily_close'], 0.5)"

class Demo:
    def calculate_factor(self):
        namespace = runtime_namespace(features, dates, codes)
        try:
            values = eval(self.EXPRESSION_CODE, namespace, {})
        except Exception:
            return None
        return values
"""

    result = rewrite_source(source)

    assert result.changed is True
    assert "eval(" not in result.source
    assert "values = namespace['add'](features['daily_close'], 0.5)" in result.source


def test_rewrite_source_skips_non_matching_files() -> None:
    source = "EXPRESSION_CODE = 'features[\\'daily_close\\']'\nvalue = 1\n"

    result = rewrite_source(source)

    assert result.changed is False
    assert result.skipped_reason == "no_matching_eval_assignment"
