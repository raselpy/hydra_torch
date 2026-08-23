# tests/test_config_path.py
"""Regression test for the Phase 5 bug: Hydra resolves a relative
config_path against the __main__ script, not the file where @hydra.main
is actually defined. This broke only once main.py started importing and
calling a decorated function from train.py — see PLANNER.md section 1d.

Fixed by computing an absolute _CONFIG_PATH from __file__ at import time
in each script. This test just verifies that path still resolves to a
real configs/ directory, without needing to invoke the full hydra.main
decorator machinery.
"""

import importlib
import os


def _config_path_for(module_name: str) -> str:
    mod = importlib.import_module(module_name)
    return mod._CONFIG_PATH


def test_all_entrypoint_scripts_have_a_valid_config_path():
    for module_name in [
        "src.hydra_torch.scripts.train",
        "src.hydra_torch.scripts.prepare_data",
        "src.hydra_torch.scripts.evaluate",
    ]:
        config_path = _config_path_for(module_name)
        resolved = os.path.abspath(config_path)
        assert os.path.isdir(resolved), f"{module_name}._CONFIG_PATH does not exist: {resolved}"
        assert os.path.isfile(os.path.join(resolved, "config.yaml")), (
            f"{module_name}._CONFIG_PATH does not contain config.yaml: {resolved}"
        )
