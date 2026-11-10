"""Offline contracts for explicit JoinQuant builds, using synthetic inputs only."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import build_jq_strategy as builder
from scripts.build.import_inliner import inline_factors_imports




@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / 'sample.py'
    source.write_text('''from utils.factor_base import FactorBase
class ExampleReturn(FactorBase):
    def calculate_factor(self, data):
        prices = data.pivot(index="date", columns="code", values="close")
        return prices.pct_change(fill_method=None)
''')
    requirements = tmp_path / 'requirements.json'
    requirements.write_text(json.dumps({'factors': {'ExampleReturn': {
        'requirements': {'daily': {'min_days': 3, 'fields': ['close']}}
    }}}))
    return source, requirements


def test_private_defaults_stay_legacy(monkeypatch):
    monkeypatch.setattr(builder, 'PUBLIC_RELEASE', False)
    args = builder.parse_cli_args([])
    assert not args.public
    assert args.requirements is None and args.cloud_api_mapping is None
    assert args.factor_files is None and args.factors is None
    assert args.index == '000300.XSHG'


@pytest.mark.parametrize('window', ['50', '77'])
def test_builder_rejects_obsolete_window(inputs, capsys, window):
    source, requirements = inputs
    with pytest.raises(SystemExit) as exc:
        builder.parse_cli_args([
            '--public', '--factor-files', str(source),
            '--requirements', str(requirements), '--window', window,
        ])
    assert exc.value.code == 2
    assert 'unrecognized arguments: --window ' + window in capsys.readouterr().err


def test_builder_help_omits_obsolete_window(capsys):
    with pytest.raises(SystemExit) as exc:
        builder.parse_cli_args(['--help'])
    assert exc.value.code == 0
    assert '--window' not in capsys.readouterr().out


@pytest.mark.parametrize('args', [[], ['--factor-files', 'missing.py']])
def test_public_default_requires_explicit_inputs(monkeypatch, args):
    monkeypatch.setattr(builder, 'PUBLIC_RELEASE', True)
    with pytest.raises(SystemExit):
        builder.parse_cli_args(args)


@pytest.mark.parametrize('mode', [['--freq', 'daily'], ['--single-file'], []])
def test_explicit_bundle_runs_without_repository_discovery(tmp_path, monkeypatch, inputs, mode):
    source, requirements = inputs
    def forbidden(*args, **kwargs):
        pytest.fail('explicit build must not read default inputs')
    monkeypatch.setattr(builder, 'discover_factor_files', forbidden)
    monkeypatch.setattr(builder, 'load_cloud_api_mapping', forbidden)
    out = tmp_path / 'bundle.py'
    assert builder.build_jq_strategy_main([
        '--public', '--factor-files', str(source), '--requirements', str(requirements),
        '--index', 'EXAMPLE.INDEX', '--out', str(out), *mode,
    ]) == 0
    if not mode:
        out = tmp_path / 'bundle_daily.py'
    spec = importlib.util.spec_from_file_location('example_bundle', out)
    module = importlib.util.module_from_spec(spec)
    # Generated bundles install compatibility modules; keep this test isolated.
    import sys
    for name in ('factor_base', 'utils.factor_base', 'jq_adapter', 'utils.jq_adapter'):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
    spec.loader.exec_module(module)
    assert set(module.FACTOR_REQUIREMENTS['factors']) == {'ExampleReturn'}
    assert module.FACTOR_REQUIRED_WINDOWS == {'ExampleReturn': {'daily': 3}}
    assert module.CLOUD_API_MAPPING == {}
    assert 'utils.data_access_config' not in out.read_text()
    assert 'config/data_access.json' not in out.read_text()
    assert "get_index_stocks('EXAMPLE.INDEX'" in out.read_text()
    def price(security, end_date, count=1, fields=None, **kwargs):
        stocks = security if isinstance(security, list) else [security]
        dates = pd.date_range(end=end_date, periods=count)
        return pd.DataFrame([{'date': day, 'code': stock, 'close': 10.0 + i}
                             for i, day in enumerate(dates) for stock in stocks])
    module.set_external_get_price(price)
    context = SimpleNamespace()
    module.initialize(context)
    result = module.compute_factor_batch(
        {'ExampleReturn': module.ExampleReturn}, module.FACTOR_REQUIREMENTS,
        module.FACTOR_DEPENDENCIES, ['DEMO.A'], '2020-01-03', module.JQFactorAdapter,
        windows=module.FACTOR_REQUIRED_WINDOWS,
    )
    assert result.loc['DEMO.A', 'ExampleReturn'] == pytest.approx(12 / 11 - 1)


def test_missing_or_unknown_inputs_fail(tmp_path, inputs):
    source, requirements = inputs
    with pytest.raises(SystemExit):
        builder.parse_cli_args(['--public', '--factor-files', str(tmp_path/'absent.py'),
                               '--requirements', str(requirements)])
    assert builder.build_jq_strategy_main(['--public', '--factor-files', str(source),
        '--requirements', str(requirements), '--factors', 'Absent']) == 1
    with pytest.raises(FileNotFoundError):
        builder.load_cloud_api_mapping(tmp_path, tmp_path / 'absent.json')


def test_mapping_override_is_explicit(tmp_path):
    mapping = tmp_path / 'mapping.json'
    mapping.write_text('{"get_valuation": {"market_cap": "market_cap"}}')
    assert builder.load_cloud_api_mapping(tmp_path, mapping)['get_valuation']['market_cap'] == 'market_cap'


@pytest.mark.parametrize('failure', ['duplicate_factor_class_in_explicit_files', 'requested_factor_class_missing'])
def test_explicit_class_errors_are_structured_and_do_not_write(tmp_path, inputs, capsys, failure):
    source, requirements = inputs
    files, selection = [str(source)], []
    if failure == 'duplicate_factor_class_in_explicit_files':
        duplicate = tmp_path / 'duplicate.py'
        duplicate.write_text(source.read_text())
        files.append(str(duplicate))
    else:
        selection = ['--factors', 'Absent']
    output = tmp_path / 'bundle.py'
    assert builder.build_jq_strategy_main([
        '--public', '--factor-files', *files, '--requirements', str(requirements),
        '--out', str(output), *selection,
    ]) == 1
    captured = capsys.readouterr()
    error = json.loads(captured.err)
    assert error['ok'] is False
    assert error['error']['code'] == failure
    assert error['error']['details']['classes'] == (['Absent'] if selection else ['ExampleReturn'])
    assert 'Traceback' not in captured.err + captured.out
    assert not output.exists()


@pytest.mark.parametrize(
    ('mapping_payload', 'message'),
    [
        (['not-a-mapping'], 'cloud_api_mapping must be a JSON object'),
        ({'get_price': 'close'}, 'cloud_api_mapping.get_price must be a list of non-empty strings'),
        ({'get_valuation': {'market_cap': []}}, 'cloud_api_mapping.get_valuation entries'),
        ({'get_factor_values': None}, 'cloud_api_mapping.get_factor_values must be a list of non-empty strings'),
        ({'get_money_flow': {'net_amount_main': {'field': []}}},
         'cloud_api_mapping.get_money_flow.net_amount_main.field must be a non-empty string'),
        ({'get_money_flow': {'flow20': {'field': 'net_amount_main', 'rolling_days': 20}}},
         'cloud_api_mapping.get_money_flow.flow20.min_periods must be a positive integer when rolling_days is set'),
        ({'get_money_flow': {'flow20': {'field': 'net_amount_main', 'rolling_days': 2, 'min_periods': 3}}},
         'cloud_api_mapping.get_money_flow.flow20.min_periods must not exceed rolling_days'),
    ],
)
def test_invalid_cloud_mapping_fails_before_bundle_write(tmp_path, capsys, inputs, mapping_payload, message):
    source, requirements = inputs
    mapping = tmp_path / 'mapping.json'
    mapping.write_text(json.dumps(mapping_payload))
    out = tmp_path / 'bundle.py'

    assert builder.build_jq_strategy_main([
        '--public', '--factor-files', str(source), '--requirements', str(requirements),
        '--cloud-api-mapping', str(mapping), '--out', str(out),
    ]) == 1

    captured = capsys.readouterr()
    assert message in captured.err
    assert 'Traceback' not in captured.err
    assert not out.exists()


def test_bundle_isolates_module_lambda_bindings_and_preserves_shadows(tmp_path, monkeypatch):
    source_one = tmp_path / 'lambda_one.py'
    source_two = tmp_path / 'lambda_two.py'
    source_three = tmp_path / 'default_one.py'
    source_one.write_text("\n".join([
        'from utils.factor_base import FactorBase',
        'transform = lambda transform: transform + 1',
        'class LambdaOne(FactorBase):',
        '    def calculate_factor(self, data):',
        '        return transform(data.pivot(index="date", columns="code", values="close"))',
    ]))
    source_two.write_text("\n".join([
        'from utils.factor_base import FactorBase',
        'offset = 10',
        'transform = lambda offset: offset * 10',
        'class LambdaTwo(FactorBase):',
        '    def calculate_factor(self, data):',
        '        return transform(data.pivot(index="date", columns="code", values="close"))',
    ]))
    source_three.write_text("\n".join([
        'from utils.factor_base import FactorBase',
        'offset = 10',
        'def transform(values, offset=offset):',
        '    return values + offset',
        'class DefaultOffset(FactorBase):',
        '    def calculate_factor(self, data):',
        '        return transform(data.pivot(index="date", columns="code", values="close"))',
    ]))
    requirements = tmp_path / 'requirements.json'
    requirements.write_text(json.dumps({'factors': {
        'LambdaOne': {'requirements': {'daily': {'min_days': 2, 'fields': ['close']}}},
        'LambdaTwo': {'requirements': {'daily': {'min_days': 2, 'fields': ['close']}}},
        'DefaultOffset': {'requirements': {'daily': {'min_days': 2, 'fields': ['close']}}},
    }}))
    frame = pd.DataFrame({
        'date': pd.date_range('2020-01-02', periods=2),
        'code': ['DEMO.A'] * 2,
        'close': [10.0, 11.0],
    })

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    sources = {
        'LambdaOne': load(source_one, 'source_lambda_one').LambdaOne,
        'LambdaTwo': load(source_two, 'source_lambda_two').LambdaTwo,
        'DefaultOffset': load(source_three, 'source_default_offset').DefaultOffset,
    }
    expected = {name: cls.calculate_factor(None, frame) for name, cls in sources.items()}
    assert [float(expected[name].iloc[-1, 0]) for name in expected] == [12.0, 110.0, 21.0]

    out = tmp_path / 'bundle.py'
    assert builder.build_jq_strategy_main([
        '--public', '--factor-files', str(source_one), str(source_two), str(source_three),
        '--requirements', str(requirements), '--single-file', '--out', str(out),
    ]) == 0
    import sys
    for name in ('factor_base', 'utils.factor_base', 'jq_adapter', 'utils.jq_adapter'):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
    generated = load(out, 'lambda_bundle')
    actual = {name: getattr(generated, name).calculate_factor(None, frame) for name in expected}
    for name in expected:
        pd.testing.assert_frame_equal(actual[name], expected[name])
    code = out.read_text()
    assert 'LambdaOne_transform' in code
    assert 'LambdaTwo_transform' in code
    assert 'DefaultOffset_transform' in code


@pytest.mark.parametrize(
    ('source', 'message'),
    [
        ('if True:\n    transform = lambda values: values\n',
         'unsupported_module_binding_in_control_flow'),
        ('offset = 10\nclass Factor:\n    offset = offset\n',
         'unsupported_class_binding_for_namespace_isolation'),
        ('offset = 10\nclass Factor:\n    value = offset\n    offset = 20\n',
         'unsupported_class_binding_for_namespace_isolation'),
    ],
)
def test_namespace_rewriter_rejects_ambiguous_bindings(source, message):
    from scripts.build.ast_rewriter import rename_top_level_functions

    with pytest.raises(ValueError, match=message):
        rename_top_level_functions(source, 'Example')


def test_builder_rejects_control_flow_module_binding_without_traceback(tmp_path, capsys):
    source = tmp_path / 'ambiguous.py'
    source.write_text("\n".join([
        'from utils.factor_base import FactorBase',
        'if True:',
        '    transform = lambda values: values',
        'class AmbiguousFactor(FactorBase):',
        '    def calculate_factor(self, data):',
        '        return transform(data)',
    ]))
    requirements = tmp_path / 'requirements.json'
    requirements.write_text(json.dumps({'factors': {
        'AmbiguousFactor': {'requirements': {'daily': {'min_days': 1, 'fields': ['close']}}},
    }}))
    out = tmp_path / 'bundle.py'

    assert builder.build_jq_strategy_main([
        '--public', '--factor-files', str(source), '--requirements', str(requirements),
        '--single-file', '--out', str(out),
    ]) == 1

    captured = capsys.readouterr()
    assert 'unsupported_module_binding_in_control_flow' in captured.err
    assert 'Traceback' not in captured.err
    assert not out.exists()


def test_inliner_requires_declared_transitive_dependencies(tmp_path):
    factors = tmp_path / 'factors'
    factors.mkdir()
    parent = factors / 'parent.py'
    helper = factors / 'helper.py'
    parent.write_text('from factors.helper import helper\nclass Parent: pass\n')
    helper.write_text('def helper(): return 1\n')
    source = 'from factors.parent import Parent\nclass Child(Parent): pass\n'
    with pytest.raises(ValueError, match='factor_dependency_not_explicit'):
        inline_factors_imports(source, tmp_path, set(), allowed_paths=set())
    with pytest.raises(ValueError, match='factor_dependency_inline_failed'):
        inline_factors_imports(source, tmp_path, set(), allowed_paths={str(parent.resolve())})
    explicit = inline_factors_imports(source, tmp_path, set(),
                allowed_paths={str(parent.resolve()), str(helper.resolve())})
    assert explicit == inline_factors_imports(source, tmp_path, set())
    assert 'def helper' in explicit


def test_adapter_research_adjusts_price_and_volume(monkeypatch):
    from utils import jq_factor_adapter as module
    requests = []
    def price(security, fields, fq, **kwargs):
        requests.append((fields, fq))
        frame = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=3),
                              'code': ['DEMO.A'] * 3})
        for field in fields:
            frame[field] = {'close': [5., 10., 20.], 'volume': [100., 100., 100.],
                            'factor': [.5, .5, 1.]}[field]
        return frame
    monkeypatch.setattr(module, 'EXTERNAL_PIT_MODE', None)
    adapter = module.JQFactorAdapter({}, window=3, pit_mode='research', get_price_func=price)
    result = adapter.fetch_price(['DEMO.A'], '2020-01-03', ['close', 'volume'])
    assert result.close.tolist() == [2.5, 5., 20.]
    assert result.volume.tolist() == [200., 200., 100.]
    assert (['factor'], 'pre') in requests
    assert any(fq is None for _, fq in requests)


def test_adapter_merges_valuation_factor_flow_and_industry(monkeypatch):
    import sys
    from utils import jq_factor_adapter as module
    dates = pd.date_range('2020-01-01', periods=3)
    def price(security, **kwargs):
        return pd.DataFrame({'date': dates, 'code': ['DEMO.A'] * 3, 'close': [10., 11., 12.]})
    def valuation(securities, **kwargs):
        return pd.DataFrame({'day': dates, 'code': ['DEMO.A'] * 3, 'market_cap': [100., 101., 102.]})
    def factors(**kwargs):
        return {'quality': pd.DataFrame({'DEMO.A': [1., 2., 3.]}, index=dates)}
    api = SimpleNamespace(
        get_money_flow=lambda *args, **kwargs: pd.DataFrame({
            'date': dates, 'sec_code': ['DEMO.A'] * 3, 'net_amount_main': [4., 5., 6.]}),
        get_trade_days=lambda **kwargs: dates,
        get_industry=lambda *args, **kwargs: {'DEMO.A': {'sw_l1': {'industry_code': 'DEMO.INDUSTRY'}}},
    )
    monkeypatch.setitem(sys.modules, 'jqdata', api)
    monkeypatch.setattr(module, 'EXTERNAL_GET_VALUATION', valuation)
    monkeypatch.setattr(module, 'EXTERNAL_GET_FACTOR_VALUES', factors)
    monkeypatch.setattr(module, 'EXTERNAL_PIT_MODE', None)
    monkeypatch.setattr(module, 'CLOUD_API_MAPPING', {
        'get_valuation': {'market_cap': 'market_cap'}, 'get_factor_values': ['quality'],
        'get_money_flow': {'net_amount_main': {'field': 'net_amount_main'}},
    })
    adapter = module.JQFactorAdapter({}, window=3, get_price_func=price)
    result = adapter.fetch_price(['DEMO.A'], '2020-01-03',
        ['close', 'market_cap', 'quality', 'net_amount_main', 'sw_l1_code'])
    assert result.shape[0] == 3
    assert result.quality.tolist() == [1., 2., 3.]
    assert result.market_cap.tolist() == [100., 101., 102.]
    assert result.net_amount_main.tolist() == [4., 5., 6.]
    assert result.sw_l1_code.tolist() == ['DEMO.INDUSTRY'] * 3
    with pytest.raises(ValueError, match='factor_input_fields_missing'):
        adapter.fetch_price(['DEMO.A'], '2020-01-03', ['close', 'unknown_field'])


def test_namespace_isolation_preserves_inlined_comprehension_scopes():
    from scripts.build.ast_rewriter import rename_top_level_functions

    source = '''offset = 10
helper = lambda x: x + offset

def calculate():
    values = [helper(offset) for offset in range(3)]
    nested = [[offset + extra for extra in range(2)] for offset in range(2)]
    mapping = {offset: helper(offset) for offset in range(2)}
    unique = {helper(offset) for offset in range(2)}
    delayed = [lambda: offset for offset in range(2)]
    return values, nested, mapping, unique, delayed[0](), offset
'''
    first = rename_top_level_functions(source, 'First')
    second = rename_top_level_functions(source.replace('offset = 10', 'offset = 100'), 'Second')
    namespace = {}
    exec(first + '\n' + second, namespace)
    assert namespace['First_calculate']() == ([10, 11, 12], [[0, 1], [1, 2]], {0: 10, 1: 11}, {10, 11}, 1, 10)
    assert namespace['Second_calculate']()[-1] == 100
