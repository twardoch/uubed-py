#!/usr/bin/env python3
"""Test configuration management system."""

import pytest
import tempfile
import os
import json
from pathlib import Path

from uubed.config import (
    UubedConfig,
    get_config,
    load_config,
    get_setting,
    set_setting,
    create_default_config,
    _global_config
)
from uubed.exceptions import UubedConfigurationError


class TestUubedConfig:
    """Test UubedConfig class."""
    
    def test_default_config_initialization(self):
        """Test that default config is properly initialized."""
        config = UubedConfig()
        
        # Check that defaults are loaded
        assert config.get("encoding.default_method") == "auto"
        assert config.get("streaming.default_batch_size") == 100
        assert config.get("performance.use_gpu") == "auto"
        assert config.get("output.verbosity") == "info"
    
    def test_config_get_with_default(self):
        """Test get method with default values."""
        config = UubedConfig()
        
        # Existing key
        assert config.get("encoding.default_method") == "auto"
        
        # Non-existing key with default
        assert config.get("nonexistent.key", "default_value") == "default_value"
        
        # Non-existing key without default
        assert config.get("nonexistent.key") is None
    
    def test_config_set_get(self):
        """Test setting and getting config values."""
        config = UubedConfig()
        
        # Set a new value
        config.set("encoding.default_method", "shq64")
        assert config.get("encoding.default_method") == "shq64"
        
        # Set nested value
        config.set("new_section.new_key", "new_value")
        assert config.get("new_section.new_key") == "new_value"
        
        # Set multiple levels
        config.set("a.b.c.d", "deep_value")
        assert config.get("a.b.c.d") == "deep_value"
    
    def test_load_toml_config(self):
        """Test loading TOML configuration."""
        toml_content = """
[encoding]
default_method = "shq64"

[streaming]
default_batch_size = 200

[performance]
use_gpu = true
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            temp_path = f.name
        
        try:
            config = UubedConfig(temp_path)
            
            # Check that values were loaded
            assert config.get("encoding.default_method") == "shq64"
            assert config.get("streaming.default_batch_size") == 200
            assert config.get("performance.use_gpu") is True
            
            # Check that defaults are still there for unspecified values
            assert config.get("output.verbosity") == "info"
            
        finally:
            os.unlink(temp_path)
    
    def test_load_json_config(self):
        """Test loading JSON configuration."""
        json_content = {
            "encoding": {
                "default_method": "eq64"
            },
            "streaming": {
                "default_batch_size": 300
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_content, f)
            temp_path = f.name
        
        try:
            config = UubedConfig(temp_path)
            
            assert config.get("encoding.default_method") == "eq64"
            assert config.get("streaming.default_batch_size") == 300
            
        finally:
            os.unlink(temp_path)
    
    def test_auto_detect_format(self):
        """Test auto-detection of config file format."""
        # Test JSON auto-detection
        with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
            json.dump({"encoding": {"default_method": "t8q64"}}, f)
            temp_path = f.name
        
        try:
            config = UubedConfig(temp_path)
            assert config.get("encoding.default_method") == "t8q64"
        finally:
            os.unlink(temp_path)
        
        # Test TOML auto-detection
        with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
            f.write('[encoding]\ndefault_method = "zoq64"\n')
            temp_path = f.name
        
        try:
            config = UubedConfig(temp_path)
            assert config.get("encoding.default_method") == "zoq64"
        finally:
            os.unlink(temp_path)
    
    def test_config_merge(self):
        """Test that config merging works correctly."""
        config = UubedConfig()
        
        # Set initial value
        config.set("encoding.shq64.planes", 128)
        
        # Load config that partially overrides
        json_content = {
            "encoding": {
                "default_method": "shq64",
                "shq64": {
                    "planes": 256
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_content, f)
            temp_path = f.name
        
        try:
            config.load_config(temp_path)
            
            # Check that new values are set
            assert config.get("encoding.default_method") == "shq64"
            assert config.get("encoding.shq64.planes") == 256
            
            # Check that other defaults remain
            assert config.get("streaming.default_batch_size") == 100
            
        finally:
            os.unlink(temp_path)
    
    def test_save_config(self):
        """Test saving configuration to file."""
        config = UubedConfig()
        config.set("encoding.default_method", "custom_method")
        config.set("streaming.default_batch_size", 500)
        
        # Save as TOML
        with tempfile.NamedTemporaryFile(suffix='.toml', delete=False) as f:
            toml_path = f.name
        
        try:
            config.save_config(toml_path)
            
            # Load back and verify
            loaded_config = UubedConfig(toml_path)
            assert loaded_config.get("encoding.default_method") == "custom_method"
            assert loaded_config.get("streaming.default_batch_size") == 500
            
        finally:
            os.unlink(toml_path)
        
        # Save as JSON
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            config.save_config(json_path)
            
            # Load back and verify
            loaded_config = UubedConfig(json_path)
            assert loaded_config.get("encoding.default_method") == "custom_method"
            assert loaded_config.get("streaming.default_batch_size") == 500
            
        finally:
            os.unlink(json_path)
    
    def test_get_encoding_params(self):
        """Test getting encoding parameters."""
        config = UubedConfig()
        
        # Test default shq64 params
        shq64_params = config.get_encoding_params("shq64")
        assert shq64_params["planes"] == 64
        
        # Test t8q64 params
        t8q64_params = config.get_encoding_params("t8q64")
        assert t8q64_params["k"] == 8
        
        # Test non-existent method
        empty_params = config.get_encoding_params("nonexistent")
        assert empty_params == {}
    
    def test_config_sections(self):
        """Test getting configuration sections."""
        config = UubedConfig()
        
        streaming_config = config.get_streaming_config()
        assert streaming_config["default_batch_size"] == 100
        
        performance_config = config.get_performance_config()
        assert performance_config["use_gpu"] == "auto"
        
        output_config = config.get_output_config()
        assert output_config["verbosity"] == "info"
    
    def test_reset_to_defaults(self):
        """Test resetting configuration to defaults."""
        config = UubedConfig()
        
        # Modify some values
        config.set("encoding.default_method", "custom")
        config.set("streaming.default_batch_size", 999)
        
        # Reset to defaults
        config.reset_to_defaults()
        
        # Check that defaults are restored
        assert config.get("encoding.default_method") == "auto"
        assert config.get("streaming.default_batch_size") == 100
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = UubedConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "encoding" in config_dict
        assert "streaming" in config_dict
        assert "performance" in config_dict
        assert "output" in config_dict
        
        # Verify it's a copy
        config_dict["encoding"]["default_method"] = "modified"
        assert config.get("encoding.default_method") == "auto"


class TestConfigurationErrors:
    """Test configuration error handling."""
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(UubedConfigurationError, match="Configuration file not found"):
            UubedConfig("nonexistent_config.toml")
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json}')  # Invalid JSON
            temp_path = f.name
        
        try:
            with pytest.raises(UubedConfigurationError, match="Invalid JSON"):
                UubedConfig(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_invalid_toml(self):
        """Test loading invalid TOML configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write('[invalid toml')  # Invalid TOML
            temp_path = f.name
        
        try:
            with pytest.raises(UubedConfigurationError, match="Invalid TOML"):
                UubedConfig(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_save_without_config_file(self):
        """Test saving without specifying config file."""
        config = UubedConfig()
        
        with pytest.raises(UubedConfigurationError, match="No configuration file specified"):
            config.save_config()


class TestGlobalConfig:
    """Test global configuration functions."""
    
    def setup_method(self):
        """Reset global config before each test."""
        global _global_config
        _global_config = None
    
    def test_get_config_singleton(self):
        """Test that get_config returns singleton instance."""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
        assert isinstance(config1, UubedConfig)
    
    def test_load_config_global(self):
        """Test loading config into global instance."""
        json_content = {
            "encoding": {
                "default_method": "global_test"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_content, f)
            temp_path = f.name
        
        try:
            load_config(temp_path)
            
            config = get_config()
            assert config.get("encoding.default_method") == "global_test"
            
        finally:
            os.unlink(temp_path)
    
    def test_get_set_setting(self):
        """Test global get/set setting functions."""
        # Test getting default value
        assert get_setting("encoding.default_method") == "auto"
        
        # Test setting value
        set_setting("encoding.default_method", "global_shq64")
        assert get_setting("encoding.default_method") == "global_shq64"
        
        # Test getting with default
        assert get_setting("nonexistent.key", "default") == "default"
    
    def test_create_default_config(self):
        """Test creating default configuration file."""
        # Test TOML creation
        with tempfile.NamedTemporaryFile(suffix='.toml', delete=False) as f:
            toml_path = f.name
        
        try:
            create_default_config(toml_path, "toml")
            
            # Verify file was created and contains defaults
            assert Path(toml_path).exists()
            
            config = UubedConfig(toml_path)
            assert config.get("encoding.default_method") == "auto"
            assert config.get("streaming.default_batch_size") == 100
            
        finally:
            os.unlink(toml_path)
        
        # Test JSON creation
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            create_default_config(json_path, "json")
            
            assert Path(json_path).exists()
            
            config = UubedConfig(json_path)
            assert config.get("encoding.default_method") == "auto"
            
        finally:
            os.unlink(json_path)
        
        # Test auto-extension
        temp_dir = Path(tempfile.gettempdir())
        auto_path = temp_dir / "test_auto_config"
        
        try:
            create_default_config(auto_path, "toml")
            
            expected_path = auto_path.with_suffix(".toml")
            assert expected_path.exists()
            
        finally:
            if expected_path.exists():
                expected_path.unlink()


class TestConfigDefaultSearch:
    """Test default configuration file search."""
    
    def test_default_search_order(self):
        """Test that configuration files are found in correct order."""
        # Create a temporary directory to work in
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create config files in different locations
            config_files = [
                temp_path / "uubed.toml",
                temp_path / "uubed.json",
                temp_path / ".uubed.toml",
                temp_path / ".uubed.json"
            ]
            
            # Create first config file with unique content
            config_files[0].write_text('[encoding]\ndefault_method = "first_found"\n')
            
            # Create second config file that should not be loaded
            config_files[1].write_text('{"encoding": {"default_method": "second_found"}}')
            
            # Change to temp directory and create config
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                config = UubedConfig()
                
                # Should load the first file found
                assert config.get("encoding.default_method") == "first_found"
                
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__])