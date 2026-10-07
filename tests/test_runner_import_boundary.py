"""Imports used by actual notebook cells must survive the nb-strip boundary."""
import ast

import run_all


def test_lag_stage_imports_exist_in_executable_cells():
    steps, _ = run_all.build_plan()
    stage = next(item for item in steps if item['id'] == '6.6')
    nodes = []
    function = None
    for cell in stage['_cells']:
        for node in ast.parse(cell['code']).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                nodes.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name == 'run_lag_variant':
                function = node
    assert function is not None
    namespace = {'MODEL_REGISTRY': {'dummy': lambda *args: None},
                 'FINAL_MODEL_NAME': 'dummy', 'FEATURE_COLS': ['a'],
                 'run_cv': lambda *args, **kwargs: {'fit_sec': 0.0}}
    exec(compile(ast.Module([*nodes, function], type_ignores=[]), '<actual-stage-6.6>', 'exec'), namespace)
    assert callable(namespace['isolated_regime_predictions'])
    result = namespace['run_lag_variant']('test', 'Import boundary regression')
    assert result['vid'] == 'test' and result['n_feat'] == 1
