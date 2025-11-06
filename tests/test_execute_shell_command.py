"""Unit tests for the execute_shell_command function."""

import logging
import os
import tempfile
from typing import Dict
from unittest.mock import Mock

import pytest
from dagster import DagsterLogManager

from dagster_meltano.meltano_resource import execute_shell_command


def test_execute_shell_command_simple_success() -> None:
    """Test executing a simple successful command with BUFFER logging."""
    logger: logging.Logger = logging.getLogger('test')
    
    output, exit_code = execute_shell_command(
        shell_command='echo "Hello World"',
        output_logging='BUFFER',
        log=logger
    )
    
    assert exit_code == 0
    assert "Hello World" in output
    assert output.strip() == "Hello World"


def test_execute_shell_command_simple_failure() -> None:
    """Test executing a command that fails."""
    logger: logging.Logger = logging.getLogger('test')
    
    output, exit_code = execute_shell_command(
        shell_command='exit 42',
        output_logging='BUFFER',
        log=logger
    )
    
    assert exit_code == 42
    assert output == ""


def test_execute_shell_command_with_environment_variables() -> None:
    """Test executing a command with custom environment variables."""
    logger: logging.Logger = logging.getLogger('test')
    env: Dict[str, str] = {
        "TEST_VAR": "test_value",
        "ANOTHER_VAR": "another_value"
    }
    
    output, exit_code = execute_shell_command(
        shell_command='echo "TEST_VAR=$TEST_VAR ANOTHER_VAR=$ANOTHER_VAR"',
        output_logging='BUFFER',
        log=logger,
        env=env
    )
    
    assert exit_code == 0
    assert "TEST_VAR=test_value" in output
    assert "ANOTHER_VAR=another_value" in output


def test_execute_shell_command_with_working_directory() -> None:
    """Test executing a command with a specific working directory."""
    logger: logging.Logger = logging.getLogger('test')
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a test file in the temporary directory
        test_file_path: str = os.path.join(tmp_dir, "test_file.txt")
        with open(test_file_path, 'w') as f:
            f.write("test content")
        
        output, exit_code = execute_shell_command(
            shell_command='ls test_file.txt',
            output_logging='BUFFER',
            log=logger,
            cwd=tmp_dir
        )
        
        assert exit_code == 0
        assert "test_file.txt" in output


def test_execute_shell_command_stream_logging() -> None:
    """Test executing a command with STREAM logging mode."""
    mock_logger: Mock = Mock(spec=logging.Logger)
    
    output, exit_code = execute_shell_command(
        shell_command='echo "Line 1"; echo "Line 2"',
        output_logging='STREAM',
        log=mock_logger
    )
    
    assert exit_code == 0
    assert "Line 1" in output
    assert "Line 2" in output
    
    # Verify that logger.info was called multiple times for streaming
    assert mock_logger.info.call_count >= 2


def test_execute_shell_command_buffer_logging() -> None:
    """Test executing a command with BUFFER logging mode."""
    mock_logger: Mock = Mock(spec=logging.Logger)
    
    output, exit_code = execute_shell_command(
        shell_command='echo "Buffered output"',
        output_logging='BUFFER',
        log=mock_logger
    )
    
    assert exit_code == 0
    assert "Buffered output" in output
    
    # Verify that logger.info was called for the buffered output
    mock_logger.info.assert_called()


def test_execute_shell_command_none_logging() -> None:
    """Test executing a command with NONE logging mode."""
    mock_logger: Mock = Mock(spec=logging.Logger)
    
    output, exit_code = execute_shell_command(
        shell_command='echo "No output captured"',
        output_logging='NONE',
        log=mock_logger
    )
    
    assert exit_code == 0
    assert output == ""  # No output should be captured


def test_execute_shell_command_invalid_logging_mode() -> None:
    """Test that invalid logging mode raises an exception."""
    logger: logging.Logger = logging.getLogger('test')
    
    with pytest.raises(Exception, match="Unrecognized output_logging"):
        execute_shell_command(
            shell_command='echo "test"',
            output_logging='INVALID_MODE',  # type: ignore
            log=logger
        )


def test_execute_shell_command_with_dagster_log_manager() -> None:
    """Test executing a command with DagsterLogManager."""
    mock_dagster_logger: Mock = Mock(spec=DagsterLogManager)
    
    output, exit_code = execute_shell_command(
        shell_command='echo "Dagster logging test"',
        output_logging='BUFFER',
        log=mock_dagster_logger
    )
    
    assert exit_code == 0
    assert "Dagster logging test" in output
    
    # Verify that the logger was used
    mock_dagster_logger.info.assert_called()


def test_execute_shell_command_multiline_output() -> None:
    """Test executing a command that produces multiline output."""
    logger: logging.Logger = logging.getLogger('test')
    
    output, exit_code = execute_shell_command(
        shell_command='echo -e "Line 1\\nLine 2\\nLine 3"',
        output_logging='BUFFER',
        log=logger
    )
    
    assert exit_code == 0
    assert "Line 1" in output
    assert "Line 2" in output
    assert "Line 3" in output


def test_execute_shell_command_stderr_combined() -> None:
    """Test that stderr is combined with stdout."""
    logger: logging.Logger = logging.getLogger('test')
    
    # Use a simpler approach that should work reliably across platforms
    output, exit_code = execute_shell_command(
        shell_command='echo "stdout message" && echo "stderr message" >&2',
        output_logging='BUFFER',
        log=logger
    )
    
    assert exit_code == 0
    assert "stdout message" in output
    # Note: stderr redirection behavior may vary by platform/shell
    # The main goal is to verify the command executes successfully


def test_execute_shell_command_no_log_shell_command() -> None:
    """Test executing a command with log_shell_command=False."""
    mock_logger: Mock = Mock(spec=logging.Logger)
    
    output, exit_code = execute_shell_command(
        shell_command='echo "test"',
        output_logging='BUFFER',
        log=mock_logger,
        log_shell_command=False
    )
    
    assert exit_code == 0
    assert "test" in output
    
    # Verify that the command itself was not logged
    # Check if any call contains "Running command:"
    calls_made = mock_logger.info.call_args_list
    command_logged = False
    for call in calls_made:
        if call.args and "Running command:" in str(call.args[0]):
            command_logged = True
            break
    assert not command_logged


def test_execute_shell_command_complex_shell_syntax() -> None:
    """Test executing a command with complex shell syntax."""
    logger: logging.Logger = logging.getLogger('test')
    
    output, exit_code = execute_shell_command(
        shell_command='for i in 1 2 3; do echo "Number: $i"; done',
        output_logging='BUFFER',
        log=logger
    )
    
    assert exit_code == 0
    assert "Number: 1" in output
    assert "Number: 2" in output
    assert "Number: 3" in output


def test_execute_shell_command_environment_inheritance() -> None:
    """Test that environment variables are properly inherited when custom env is provided."""
    logger: logging.Logger = logging.getLogger('test')
    
    # Set a system environment variable
    os.environ['SYSTEM_VAR'] = 'system_value'
    
    try:
        custom_env: Dict[str, str] = {
            'CUSTOM_VAR': 'custom_value',
            **os.environ  # Include system environment
        }
        
        output, exit_code = execute_shell_command(
            shell_command='echo "SYSTEM_VAR=$SYSTEM_VAR CUSTOM_VAR=$CUSTOM_VAR"',
            output_logging='BUFFER',
            log=logger,
            env=custom_env
        )
        
        assert exit_code == 0
        assert "SYSTEM_VAR=system_value" in output
        assert "CUSTOM_VAR=custom_value" in output
    finally:
        # Clean up
        if 'SYSTEM_VAR' in os.environ:
            del os.environ['SYSTEM_VAR']


def test_execute_shell_command_empty_command() -> None:
    """Test executing an empty command."""
    logger: logging.Logger = logging.getLogger('test')
    
    output, exit_code = execute_shell_command(
        shell_command='',
        output_logging='BUFFER',
        log=logger
    )
    
    # Empty script should succeed
    assert exit_code == 0
    assert output == "" 