#!/usr/bin/env python3
"""Test CLI functionality."""

import pytest
import numpy as np
import tempfile
import os
from click.testing import CliRunner

from uubed.cli import main


class TestCLIBasic:
    """Test basic CLI functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "High-performance encoding for embedding vectors" in result.output
    
    def test_cli_version(self):
        """Test CLI version command."""
        result = self.runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "uubed" in result.output
    
    def test_info_command(self):
        """Test info command."""
        result = self.runner.invoke(main, ["info"])
        assert result.exit_code == 0
        assert "uubed v" in result.output
        assert "Build Information" in result.output
        assert "Available Encoders" in result.output


class TestEncodeCommand:
    """Test encode command functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_encode_help(self):
        """Test encode command help."""
        result = self.runner.invoke(main, ["encode", "--help"])
        assert result.exit_code == 0
        assert "Encode embedding vector" in result.output
    
    def test_encode_from_file(self):
        """Test encoding from file."""
        # Create test file
        test_data = np.random.randint(0, 256, 32, dtype=np.uint8).tobytes()
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(test_data)
            tmp_path = tmp.name
        
        try:
            result = self.runner.invoke(main, ["encode", tmp_path, "-m", "eq64"])
            assert result.exit_code == 0
            assert len(result.output.strip()) > 0
            
        finally:
            os.unlink(tmp_path)
    
    def test_encode_to_file(self):
        """Test encoding to output file."""
        # Create test input file
        test_data = np.random.randint(0, 256, 32, dtype=np.uint8).tobytes()
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
            tmp_in.write(test_data)
            input_path = tmp_in.name
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_out:
            output_path = tmp_out.name
        
        try:
            result = self.runner.invoke(main, [
                "encode", input_path, "-o", output_path, "-m", "shq64"
            ])
            assert result.exit_code == 0
            
            # Check output file was created
            assert os.path.exists(output_path)
            with open(output_path, 'r') as f:
                content = f.read().strip()
                assert len(content) > 0
                
        finally:
            os.unlink(input_path)
            os.unlink(output_path)
    
    def test_encode_different_methods(self):
        """Test encoding with different methods."""
        test_data = np.random.randint(0, 256, 64, dtype=np.uint8).tobytes()
        
        methods = ["eq64", "shq64", "t8q64", "zoq64", "auto"]
        
        for method in methods:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(test_data)
                tmp_path = tmp.name
            
            try:
                args = ["encode", tmp_path, "-m", method]
                if method == "t8q64":
                    args.extend(["--k", "8"])
                elif method == "shq64":
                    args.extend(["--planes", "64"])
                
                result = self.runner.invoke(main, args)
                assert result.exit_code == 0, f"Method {method} failed"
                assert len(result.output.strip()) > 0
                
            finally:
                os.unlink(tmp_path)
    
    def test_encode_stdin(self):
        """Test encoding from stdin."""
        test_data = b"Hello, World!"
        
        result = self.runner.invoke(main, ["encode", "-m", "eq64"], input=test_data)
        assert result.exit_code == 0
        assert len(result.output.strip()) > 0


class TestDecodeCommand:
    """Test decode command functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_decode_help(self):
        """Test decode command help."""
        result = self.runner.invoke(main, ["decode", "--help"])
        assert result.exit_code == 0
        assert "Decode encoded string" in result.output
    
    def test_encode_decode_roundtrip(self):
        """Test full encode-decode roundtrip."""
        # Create test data
        test_data = np.random.randint(0, 256, 32, dtype=np.uint8).tobytes()
        
        # Create input file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
            tmp_in.write(test_data)
            input_path = tmp_in.name
        
        # Create files for intermediate and final output
        with tempfile.NamedTemporaryFile(delete=False) as tmp_encoded:
            encoded_path = tmp_encoded.name
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_decoded:
            decoded_path = tmp_decoded.name
        
        try:
            # Encode
            result = self.runner.invoke(main, [
                "encode", input_path, "-o", encoded_path, "-m", "eq64"
            ])
            assert result.exit_code == 0
            
            # Decode
            result = self.runner.invoke(main, [
                "decode", encoded_path, "-o", decoded_path, "-m", "eq64"
            ])
            assert result.exit_code == 0
            
            # Check roundtrip
            with open(decoded_path, 'rb') as f:
                decoded_data = f.read()
                assert decoded_data == test_data
                
        finally:
            for path in [input_path, encoded_path, decoded_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def test_decode_stdin(self):
        """Test decoding from stdin."""
        # First encode some data
        test_data = np.random.randint(0, 256, 16, dtype=np.uint8).tobytes()
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(test_data)
            tmp_path = tmp.name
        
        try:
            # Get encoded string
            encode_result = self.runner.invoke(main, ["encode", tmp_path, "-m", "eq64"])
            assert encode_result.exit_code == 0
            encoded_string = encode_result.output.strip()
            
            # Decode from stdin
            decode_result = self.runner.invoke(main, [
                "decode", "-m", "eq64"
            ], input=encoded_string)
            assert decode_result.exit_code == 0
            
        finally:
            os.unlink(tmp_path)


class TestBenchCommand:
    """Test bench command functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_bench_help(self):
        """Test bench command help."""
        result = self.runner.invoke(main, ["bench", "--help"])
        assert result.exit_code == 0
        assert "Benchmark encoding performance" in result.output
    
    def test_bench_basic(self):
        """Test basic benchmarking."""
        result = self.runner.invoke(main, [
            "bench", "--size", "10", "--dims", "32", "--iterations", "2"
        ])
        assert result.exit_code == 0
        assert "Benchmark Results" in result.output
        assert "Total Time" in result.output
        assert "Throughput" in result.output
    
    def test_bench_specific_method(self):
        """Test benchmarking specific method."""
        result = self.runner.invoke(main, [
            "bench", "--size", "5", "--dims", "64", "--iterations", "1", "-m", "shq64"
        ])
        assert result.exit_code == 0
        assert "shq64" in result.output
    
    def test_bench_all_methods(self):
        """Test benchmarking all methods."""
        result = self.runner.invoke(main, [
            "bench", "--size", "5", "--dims", "32", "--iterations", "1", "-m", "all"
        ])
        assert result.exit_code == 0
        assert "eq64" in result.output
        assert "shq64" in result.output
        assert "t8q64" in result.output
        assert "zoq64" in result.output


class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_encode_nonexistent_file(self):
        """Test encoding nonexistent file."""
        result = self.runner.invoke(main, ["encode", "nonexistent.bin"])
        assert result.exit_code != 0
    
    def test_decode_invalid_method(self):
        """Test decoding with invalid method."""
        result = self.runner.invoke(main, [
            "decode", "-m", "shq64"
        ], input="SomeEncodedString")
        assert result.exit_code != 0
        assert "Error:" in result.output
    
    def test_invalid_method(self):
        """Test invalid encoding method."""
        test_data = b"test"
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(test_data)
            tmp_path = tmp.name
        
        try:
            result = self.runner.invoke(main, ["encode", tmp_path, "-m", "invalid"])
            assert result.exit_code != 0
            
        finally:
            os.unlink(tmp_path)


class TestCLIIntegration:
    """Test CLI integration and workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_pipeline_workflow(self):
        """Test a complete pipeline workflow."""
        # Create test embeddings file
        embeddings_data = b""
        for _ in range(5):
            embedding = np.random.randint(0, 256, 64, dtype=np.uint8)
            embeddings_data += embedding.tobytes()
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_input:
            tmp_input.write(embeddings_data)
            input_path = tmp_input.name
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_encoded:
            encoded_path = tmp_encoded.name
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_decoded:
            decoded_path = tmp_decoded.name
        
        try:
            # Step 1: Encode the embeddings
            result = self.runner.invoke(main, [
                "encode", input_path, "-o", encoded_path, "-m", "eq64"
            ])
            assert result.exit_code == 0
            
            # Step 2: Decode back
            result = self.runner.invoke(main, [
                "decode", encoded_path, "-o", decoded_path, "-m", "eq64"
            ])
            assert result.exit_code == 0
            
            # Step 3: Verify roundtrip
            with open(decoded_path, 'rb') as f:
                decoded_data = f.read()
                assert decoded_data == embeddings_data
            
            # Step 4: Benchmark the method
            result = self.runner.invoke(main, [
                "bench", "--size", "5", "--dims", "64", "--iterations", "1", "-m", "eq64"
            ])
            assert result.exit_code == 0
            
        finally:
            for path in [input_path, encoded_path, decoded_path]:
                if os.path.exists(path):
                    os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__])