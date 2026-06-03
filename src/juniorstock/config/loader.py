# path: src/juniorstock/config/loader.py
#!/usr/bin/env python3
"""
Feature: Config Loader (Python 3.9 compatible)

Loads YAML/JSON config with fallback.
"""

import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class ConfigLoader:
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config_path = Path(config_path)

    def load(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}

        if self.config_path.suffix in [".yaml", ".yml"] and HAS_YAML:
            import yaml
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        else:
            with open(self.config_path) as f:
                return json.load(f)
