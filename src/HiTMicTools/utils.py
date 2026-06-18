# Standard library imports
import re
import json
import copy
import logging
from datetime import timedelta
from typing import Dict, Any

# Third-party imports
import pandas as pd
import ome_types
import os


def remove_file_extension(filename: str) -> str:
    """
    Remove specific file extensions from a filename.

    Args:
        filename (str): The input filename.

    Returns:
        str: Filename with the extension removed.
    """
    extensions = ["nd2", "ome\.p\.tiff", "p\.tiff", "ome\.tiff", "tiff"]
    pattern = r"\.(?:" + "|".join(extensions) + ")$"
    return re.sub(pattern, "", filename)


def unit_converter(
    value: float, conversion_factor: float = 1, to_unit: str = "pixel"
) -> float:
    """
    Convert a value between pixels and micrometers (um) using a conversion factor.

    Args:
        value (float): The value to be converted.
        conversion_factor (float, optional, default 1): The conversion factor between pixels and micrometers.
        to_unit (str, optional, default pixel): The target unit for conversion. Can be either 'pixel' or 'um'.

    Returns:
        float: The converted value in the specified unit.

    Raises:
        ValueError: If an invalid unit is provided.
    """

    if to_unit == "um":
        return value * conversion_factor
    elif to_unit == "pixel":
        return value / conversion_factor
    else:
        raise ValueError("Invalid unit. Choose either 'um' or 'pixel'.")


def get_timestamps(
    metadata: ome_types.model.OME,
    ref_channel: int = 0,
    timeformat: str = "%Y-%m-%d %H:%M:%S",
) -> pd.DataFrame:
    """
    Extract timestamps from the metadata of an OME-TIFF file.

    Args:
        metadata (ome_types.model.OME): The metadata of the OME-TIFF file.
        ref_channel (int, optional): The reference channel for timestamps. Defaults to 0.
        timeformat (str, optional): The format string for the timestamp. Defaults to "%Y-%m-%d %H:%M:%S".

    Returns:
        pd.DataFrame: A DataFrame containing the timestamps for each frame.
    """
    base_time = metadata.images[0].acquisition_date
    z = metadata.images[0].pixels.planes
    base_lag = z[0].delta_t
    timestamps = []
    for i in range(len(z)):
        timestamp = z[i].delta_t
        if z[i].the_c == ref_channel:
            delta_t_s = timestamp / 1000
            timepoint = (
                base_time
                + timedelta(milliseconds=timestamp)
                - timedelta(milliseconds=base_lag)
            )
            formatted_timepoint = timepoint.strftime(timeformat)
            timestep = timestamp

            timestamps.append(
                {
                    "frame": z[i].the_t,
                    "date_time": formatted_timepoint,
                    "timestep": timestep,
                    "abslag_in_s": delta_t_s,
                }
            )
    df = pd.DataFrame(timestamps)

    df["timestep"] = df["timestep"] - df.loc[df["frame"] == 0, "timestep"].iloc[0]
    df["timestep"] = df["timestep"] / 3600000
    return df


def round_to_odd(number: float) -> int:
    """Round a number to the nearest odd integer."""
    return int(number) if number % 2 == 1 else int(number) + 1


def read_metadata(metadata_file: str) -> Dict[str, Any]:
    """Read metadata from a JSON file."""
    with open(metadata_file) as f:
        metadata = json.load(f)
    return metadata


def update_config(
    target_dict: Dict[str, Any],
    override_dict: Dict[str, Any],
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """
    Recursively update a nested dictionary with override values.

    Args:
        target_dict (Dict[str, Any]): The original dictionary to update.
        override_dict (Dict[str, Any]): Dictionary containing override values.
        logger (logging.Logger, optional): Logger for tracking changes.

    Returns:
        Dict[str, Any]: Updated dictionary with override values applied.
    """

    result = copy.deepcopy(target_dict)

    def _recursive_update(
        target: Dict[str, Any], override: Dict[str, Any], path: str = ""
    ) -> None:
        """Walk both dictionaries depth-first and override values in-place."""
        for key, value in override.items():
            current_path = f"{path}.{key}" if path else key

            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                _recursive_update(target[key], value, current_path)
            else:
                if key in target and logger:
                    old_value = target[key]
                    if old_value != value:
                        logger.info(f"Updating {current_path}: {old_value} -> {value}")
                elif logger:
                    logger.warning(f"Current path not found: {current_path}")
                target[key] = value

    _recursive_update(result, override_dict)
    return result


def check_btrack() -> bool:
    """
    Check if the btrack package contains the compiled library.

    Returns:
        bool: True if the compiled library is found, False otherwise.
    """
    try:
        import btrack

        # Get the btrack package path
        btrack_path = os.path.dirname(btrack.__file__)
        libs_path = os.path.join(btrack_path, "libs")

        if not os.path.exists(libs_path):
            return False

        # Check for compiled library files for different OS
        library_files = [
            "libtracker.dylib",  # macOS
            "libtracker.so",  # Linux
            "tracker.dll",  # Windows
        ]

        for lib_file in library_files:
            lib_path = os.path.join(libs_path, lib_file)
            if os.path.exists(lib_path):
                return True

        return False

    except ImportError:
        return False
