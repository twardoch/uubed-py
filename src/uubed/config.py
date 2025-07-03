#!/usr/bin/env python3
# this_file: src/uubed/config.py
"""Configuration management for uubed with file-based settings.\n\nThis module provides a robust and flexible system for managing the configuration\nsettings of the `uubed` package. It supports loading configurations from various\nfile formats (TOML and JSON) and locations (current directory, user config, system config).\nThe configuration is accessible via a global singleton instance, allowing easy\nretrieval and modification of settings throughout the application.\n\n**Key Features:**\n- **Hierarchical Configuration:** Settings are organized into logical sections (e.g., `encoding`, `streaming`, `performance`, `output`).\n- **Default Values:** A comprehensive set of default configuration values is provided, ensuring the package can run out-of-the-box.\n- **File-Based Loading:** Automatically searches for configuration files in standard locations or loads from a specified path.\n- **Format Support:** Supports both TOML (`.toml`) and JSON (`.json`) file formats.\n- **Dot Notation Access:** Configuration values can be accessed and set using convenient dot notation (e.g., `encoding.default_method`).\n- **Merging Logic:** Loaded configurations are merged with default settings, allowing users to override specific values without redefining the entire configuration.\n- **Error Handling:** Provides specific `UubedConfigurationError` exceptions for issues like file not found or invalid file format.\n- **Global Singleton:** A single, globally accessible configuration instance ensures consistency across the application.\n\n**Configuration File Search Order (if `config_file` is not specified during initialization):**\n1. `uubed.toml` in the current working directory.\n2. `uubed.json` in the current working directory.\n3. `.uubed.toml` in the current working directory.\n4. `.uubed.json` in the current working directory.\n5. `~/.config/uubed/config.toml` (user-specific config directory).\n6. `~/.config/uubed/config.json` (user-specific config directory).\n7. `~/.uubed.toml` (user home directory).\n8. `~/.uubed.json` (user home directory).\n\n**Usage Example:**\n```python\nfrom uubed.config import get_config, load_config, set_setting, get_setting, create_default_config\nfrom pathlib import Path\n\n# Get the global configuration instance\nconfig = get_config()\n\n# Access a setting using dot notation\ndefault_encoding_method = config.get("encoding.default_method")\nprint(f"Default encoding method: {default_encoding_method}")\n\n# Set a new setting\nconfig.set("streaming.default_batch_size", 500)\nprint(f"New batch size: {config.get("streaming.default_batch_size")}")\n\n# Load configuration from a specific file\n# (Assuming you have a config.toml or config.json in your project root)\n# try:\n#     load_config("./my_custom_config.toml")\n#     print("Loaded custom config.")\n# except Exception as e:\n#     print(f"Could not load custom config: {e}")\n\n# Create a default configuration file\n# create_default_config("./new_uubed_config.toml")\n# print("Created new_uubed_config.toml with default settings.")\n\n# Access settings via module-level functions\nverbosity = get_setting("output.verbosity")\nprint(f"Current verbosity: {verbosity}")\nset_setting("output.verbosity", "debug")\nprint(f"Updated verbosity: {get_setting("output.verbosity")}")\n```\n"""

import os
import json
import toml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .exceptions import UubedConfigurationError, configuration_error


class UubedConfig:
    """Configuration manager for uubed package."""
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "encoding": {
            "default_method": "auto",
            "eq64": {},
            "shq64": {"planes": 64},
            "t8q64": {"k": 8},
            "zoq64": {}
        },
        "streaming": {
            "default_batch_size": 100,
            "progress": False,
            "memory_limit_mb": 1024
        },
        "performance": {
            "use_gpu": "auto",  # auto, true, false
            "parallel_workers": "auto",  # auto, or integer
            "cache_enabled": True,
            "cache_size_mb": 256
        },
        "output": {
            "verbosity": "info",  # debug, info, warning, error
            "progress_style": "rich",  # rich, simple, none
            "color": "auto"  # auto, always, never
        }
    }
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file. If None, searches standard locations.
        """
        self._config = self.DEFAULT_CONFIG.copy()
        self._config_file = None
        
        if config_file:
            self.load_config(config_file)
        else:
            self._load_default_config()
    
    def _load_default_config(self) -> None:
        """Load configuration from standard locations."""
        # Search order: current directory, user config, system config
        search_paths = [
            Path.cwd() / "uubed.toml",
            Path.cwd() / "uubed.json",
            Path.cwd() / ".uubed.toml",
            Path.cwd() / ".uubed.json",
            Path.home() / ".config" / "uubed" / "config.toml",
            Path.home() / ".config" / "uubed" / "config.json",
            Path.home() / ".uubed.toml",
            Path.home() / ".uubed.json",
        ]
        
        for config_path in search_paths:
            if config_path.exists():
                try:
                    self.load_config(config_path)
                    break
                except Exception as e:
                    # Continue searching if this config file fails to load
                    continue
    
    def load_config(self, config_file: Union[str, Path]) -> None:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file (JSON or TOML)
            
        Raises:
            UubedConfigurationError: If configuration file cannot be loaded or is invalid
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise configuration_error(
                f"Configuration file not found: {config_path}",
                suggestion="Check the file path or create a configuration file"
            )
        
        try:
            if config_path.suffix.lower() in ['.toml']:
                with open(config_path, 'r') as f:
                    config_data = toml.load(f)
            elif config_path.suffix.lower() in ['.json']:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            else:
                # Try to auto-detect format
                with open(config_path, 'r') as f:
                    content = f.read().strip()
                    if content.startswith('{'):
                        config_data = json.loads(content)
                    else:
                        config_data = toml.loads(content)
            
            # Merge with defaults
            self._merge_config(config_data)
            self._config_file = config_path
            
        except json.JSONDecodeError as e:
            raise configuration_error(
                f"Invalid JSON in configuration file: {e}",
                config_file=str(config_path),
                suggestion="Check JSON syntax and formatting"
            ) from e
        except toml.TomlDecodeError as e:
            raise configuration_error(
                f"Invalid TOML in configuration file: {e}",
                config_file=str(config_path),
                suggestion="Check TOML syntax and formatting"
            ) from e
        except Exception as e:
            raise configuration_error(
                f"Failed to load configuration: {e}",
                config_file=str(config_path)
            ) from e
    
    def _merge_config(self, config_data: Dict[str, Any]) -> None:
        """Recursively merge configuration data with defaults."""
        def merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively merge dictionaries."""
            result = base.copy()
            for key, value in update.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result
        
        self._config = merge_dict(self._config, config_data)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'encoding.default_method')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
    
    def save_config(self, config_file: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_file: Path to save configuration. If None, uses loaded config file.
        """
        if config_file is None:
            if self._config_file is None:
                raise configuration_error(
                    "No configuration file specified and none was loaded",
                    suggestion="Provide a config_file parameter"
                )
            config_file = self._config_file
        
        config_path = Path(config_file)
        
        try:
            # Create directory if needed
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            if config_path.suffix.lower() in ['.toml']:
                with open(config_path, 'w') as f:
                    toml.dump(self._config, f)
            else:  # Default to JSON
                with open(config_path, 'w') as f:
                    json.dump(self._config, f, indent=2)
                    
        except Exception as e:
            raise configuration_error(
                f"Failed to save configuration: {e}",
                config_file=str(config_path)
            ) from e
    
    def get_encoding_params(self, method: str) -> Dict[str, Any]:
        """Get parameters for specific encoding method."""
        return self.get(f"encoding.{method}", {})
    
    def get_streaming_config(self) -> Dict[str, Any]:
        """Get streaming configuration."""
        return self.get("streaming", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration.""" 
        return self.get("performance", {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.get("output", {})
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._config = self.DEFAULT_CONFIG.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()


# Global configuration instance
_global_config: Optional[UubedConfig] = None


def get_config() -> UubedConfig:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = UubedConfig()
    return _global_config


def load_config(config_file: Union[str, Path]) -> None:
    """Load configuration from file into global instance."""
    global _global_config
    _global_config = UubedConfig(config_file)


def get_setting(key: str, default: Any = None) -> Any:
    """Get configuration setting using dot notation."""
    return get_config().get(key, default)


def set_setting(key: str, value: Any) -> None:
    """Set configuration setting using dot notation."""
    get_config().set(key, value)


def create_default_config(config_file: Union[str, Path], format: str = "toml") -> None:
    """
    Create a default configuration file.
    
    Args:
        config_file: Path for the configuration file
        format: Configuration format ("toml" or "json")
    """
    config = UubedConfig()
    
    config_path = Path(config_file)
    if format.lower() == "toml" and not config_path.suffix:
        config_path = config_path.with_suffix(".toml")
    elif format.lower() == "json" and not config_path.suffix:
        config_path = config_path.with_suffix(".json")
    
    config.save_config(config_path)