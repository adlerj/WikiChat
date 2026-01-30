"""Configuration utilities."""
from pathlib import Path
from typing import Any, Dict

import tomllib


def load_config(config_file: Path) -> Dict[str, Any]:
    """Load configuration from TOML file.

    Args:
        config_file: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_file, "rb") as f:
        return tomllib.load(f)


def get_nested_config(config: Dict[str, Any], *keys: str) -> Any:
    """Get nested configuration value.

    Args:
        config: Configuration dictionary
        *keys: Keys to traverse

    Returns:
        Configuration value or None
    """
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current
