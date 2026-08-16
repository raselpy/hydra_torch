"""Auto-register all schemas when hydra_torch is imported."""
from .config_schema import setup_config

# Register everything immediately on package load
setup_config()
