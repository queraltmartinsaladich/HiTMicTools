import os
import sys
import logging
import pandas as pd
import numpy as np
from contextlib import contextmanager
from typing import List, Dict, Optional


def prepare_dataframe_for_tracking(
    df: pd.DataFrame,
    features: List[str],
    rename_columns: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """
    Prepare DataFrame for btrack tracking by renaming columns and selecting features.

    Args:
        df: Input DataFrame with measurements
        features: List of feature columns to include
        rename_columns: Dictionary mapping old column names to new names

    Returns:
        Array of objects formatted for btrack
    """
    # Move import inside function to avoid circular dependency
    from btrack.io.utils import objects_from_array

    # Apply column renaming if provided
    tracking_df = df.copy()
    if rename_columns:
        tracking_df = tracking_df.rename(columns=rename_columns)
    required_cols = ["t", "y", "x", "z"]

    # Select required columns + features
    all_features = required_cols + features

    tracking_df["z"] = 0  # Default z value for 2D tracking

    # Check that all required columns exist
    missing_cols = [col for col in all_features if col not in tracking_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after renaming: {missing_cols}")
    # DF -> numpy array -> btrack objects
    tracking_data = tracking_df[all_features].to_numpy()
    # Convert to btrack objects
    return objects_from_array(tracking_data, default_keys=all_features)


def merge_tracking_results(
    original_df: pd.DataFrame, tracks_df: pd.DataFrame, merge_on: List[str]
) -> pd.DataFrame:
    """
    Merge tracking results back with original measurements.

    Args:
        original_df: Original measurements DataFrame
        tracks_df: Tracking results DataFrame
        merge_on: Columns to merge on

    Returns:
        Merged DataFrame with tracking results
    """
    return pd.merge(original_df, tracks_df, on=merge_on, how="left")


def validate_dataframe_integrity(df: pd.DataFrame) -> None:
    """
    Validate DataFrame has required structure for tracking.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If validation fails
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    # Check for basic required columns (before renaming)
    basic_required = ["frame", "centroid_0", "centroid_1"]
    missing = [col for col in basic_required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check for reasonable data ranges
    if df["frame"].min() < 0:
        raise ValueError("Frame numbers cannot be negative")


@contextmanager
def suppress_native_stdout_stderr(
    mode: str = "suppress", logger: Optional[logging.Logger] = None
):
    """Suppress or capture output from native libraries (C/C++).

    Args:
        mode: 'suppress' to discard output, 'capture' to send to logger
        logger: Logger instance to use when mode='capture'
    """
    if mode == "capture" and logger is None:
        raise ValueError("Logger must be provided when mode='capture'")

    # Save original file descriptors
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(original_stdout_fd)
    saved_stderr = os.dup(original_stderr_fd)

    if mode == "suppress":
        # Original suppression behavior
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, original_stdout_fd)
            os.dup2(devnull, original_stderr_fd)
            yield
        finally:
            os.dup2(saved_stdout, original_stdout_fd)
            os.dup2(saved_stderr, original_stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)
            os.close(devnull)

    elif mode == "capture" and logger is not None:
        # Capture output and send to logger
        import tempfile

        # Create temporary files for capturing
        stdout_temp = tempfile.NamedTemporaryFile(mode="w+", delete=False)
        stderr_temp = tempfile.NamedTemporaryFile(mode="w+", delete=False)

        try:
            os.dup2(stdout_temp.fileno(), original_stdout_fd)
            os.dup2(stderr_temp.fileno(), original_stderr_fd)
            yield
        finally:
            # Restore original file descriptors
            os.dup2(saved_stdout, original_stdout_fd)
            os.dup2(saved_stderr, original_stderr_fd)
            os.close(saved_stdout)
            os.close(saved_stderr)

            # Read captured output and log it
            stdout_temp.seek(0)
            stderr_temp.seek(0)

            stdout_content = stdout_temp.read()
            stderr_content = stderr_temp.read()

            if stdout_content.strip():
                logger.info(f"Native stdout: {stdout_content.strip()}")
            if stderr_content.strip():
                logger.warning(f"Native stderr: {stderr_content.strip()}")

            # Clean up temporary files
            stdout_temp.close()
            stderr_temp.close()
            os.unlink(stdout_temp.name)
            os.unlink(stderr_temp.name)

    else:
        raise ValueError(
            f"Invalid mode: {mode}. Use 'suppress' or 'capture'. If using capture, provide a logger. Logger is {logger}."
        )
