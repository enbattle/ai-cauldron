"""
Utility helper functions for the RAG application.
"""

import os
from pathlib import Path
from typing import Optional


def ensure_directory(directory: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        directory: Directory path

    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size.

    Args:
        file_path: Path to file

    Returns:
        File size as string (e.g., "2.5 MB")
    """
    size_bytes = os.path.getsize(file_path)

    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.1f} TB"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_metadata(metadata: dict) -> str:
    """
    Format metadata dictionary as readable string.

    Args:
        metadata: Metadata dictionary

    Returns:
        Formatted string
    """
    lines = []
    for key, value in metadata.items():
        lines.append(f"**{key}**: {value}")
    return "\n".join(lines)


def load_env_variable(
    var_name: str,
    default: Optional[str] = None,
    required: bool = False,
) -> Optional[str]:
    """
    Load environment variable with optional default.

    Args:
        var_name: Environment variable name
        default: Default value if not found
        required: Raise error if not found and no default

    Returns:
        Variable value or None

    Raises:
        ValueError: If required and not found
    """
    value = os.getenv(var_name, default)

    if required and value is None:
        raise ValueError(
            f"Required environment variable '{var_name}' not set. "
            f"Please set it in your .env file or environment."
        )

    return value
