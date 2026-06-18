import pandas as pd
import numpy as np
import logging
import btrack

from btrack.constants import BayesianUpdates
from typing import Optional, Dict, Any, Tuple, List

from .config_validator import TrackingConfigValidator
from .config_loader import ConfigLoader
from .tracking_utils import (
    prepare_dataframe_for_tracking,
    merge_tracking_results,
    validate_dataframe_integrity,
    suppress_native_stdout_stderr,
)


class CellTracker:
    """
    Multi-object tracking module for cell tracking using btrack.

    This class provides a high-level interface for tracking cells across time frames
    using the btrack library. It handles configuration loading, data validation,
    and integrates seamlessly with the existing pipeline.

    Args:
        config_path: Path to tracking configuration file or zip archive
        features: List of feature columns to use for tracking
        volume_bounds: Optional volume bounds (xmax, ymax)
        tracking_updates: List of tracking update methods
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[dict] = None,
        override_args: Optional[dict] = None,
    ):
        """
        Load tracker configuration either from disk or from an in-memory dictionary.

        Args:
            config_path: Path to a YAML/JSON config (or bundle) describing tracker parameters.
            config_dict: Already-parsed configuration dictionary.
            override_args: Optional overrides applied when loading from config_path.

        Raises:
            ValueError: If neither a config_path nor config_dict is provided.
        """
        if config_dict is not None:
            self.config_path = None
            self.config = config_dict
        elif config_path is not None:
            self.config_path = config_path
            self.config = self._load_config(override_args)
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        self.tracking_updates = ["MOTION", "VISUAL"]

        # Load and validate configuration
        self.validator = TrackingConfigValidator()
        self._validate_configuration()
        self.features = self._get_default_features()
        self.volume_bounds = None

    def set_features(self, features: List[str]) -> None:
        """
        Set the features to use for the visual model in tracking.

        Args:
            features: List of feature columns to use for tracking
        """
        self.features = features

    def _get_default_features(self) -> List[str]:
        """Get default feature set for tracking."""
        return [
            # "t", "y", "x", "z", These are noted for the user but are implicit in the Motion model
            # "area",
            # "major_axis_length",
            # "minor_axis_length",
            # "solidity",
            # "orientation",
            # "rel_mean_intensity"
        ]

    def _load_config(
        self, override_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Load tracking configuration from file or zip."""
        return ConfigLoader.load_config(self.config_path, override_args=override_args)

    def _validate_configuration(self) -> None:
        """Validate the loaded configuration."""
        self.validator.validate_config_dimensions(self.config)

    def track_objects(
        self,
        measurements_df: pd.DataFrame,
        volume_bounds: Optional[Tuple[int, int]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Perform tracking on measurements DataFrame.

        Args:
            measurements_df: DataFrame with object measurements
            volume_bounds: Optional override for volume bounds

        Returns:
            DataFrame with tracking results merged

        Raises:
            ValueError: If input data validation fails
        """
        # Checks > Prepare > Tracking > Merge
        self._validate_input_data(measurements_df)
        tracking_data = self._prepare_tracking_data(measurements_df)
        tracks_df = self._run_tracking(tracking_data, volume_bounds, logger)

        return self._merge_tracking_results(measurements_df, tracks_df)

    def _validate_input_data(self, measurements_df: pd.DataFrame) -> None:
        """
        Validate input DataFrame for tracking.

        Args:
            measurements_df: DataFrame to validate

        Raises:
            ValueError: If validation fails
        """
        validate_dataframe_integrity(measurements_df)

        # Check for required tracking columns
        required_cols = ["frame", "centroid_0", "centroid_1"]
        missing_cols = [
            col for col in required_cols if col not in measurements_df.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _prepare_tracking_data(self, measurements_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare DataFrame for btrack tracking.

        Args:
            measurements_df: Original measurements DataFrame

        Returns:
            Array of objects formatted for btrack
        """
        # Create column mapping based on existing experiment code
        rename_mapping = {
            "frame": "t",
            "centroid_0": "y",
            "centroid_1": "x",
            "slice": "z",
            "label": "original_label",
        }

        out = prepare_dataframe_for_tracking(
            measurements_df, self.features, rename_columns=rename_mapping
        )
        self.xmax = measurements_df["centroid_1"].max()
        self.ymax = measurements_df["centroid_0"].max()
        return out

    def _run_tracking(
        self,
        tracking_data: np.ndarray,
        volume_bounds: Optional[Tuple[int, int]] = None,
        logger: logging.Logger = None,
    ) -> pd.DataFrame:
        """
        Execute btrack tracking algorithm.

        Args:
            tracking_data: Prepared tracking data
            volume_bounds: Optional volume bounds override

        Returns:
            DataFrame with tracking results
        """
        if volume_bounds is not None:
            xmax, ymax = volume_bounds
        else:
            xmax = self.xmax
            ymax = self.ymax

        # Initialize and configure tracker
        # Load the config file directly for testing
        if logger is not None:
            self._configure_btrack_logging(logger)
        with suppress_native_stdout_stderr(mode="capture", logger=logger):
            with btrack.BayesianTracker(verbose=False) as tracker:
                # Configure tracker with loaded config
                tracker.configure(self.config)
                # Set tracking parameters
                tracker.tracking_updates = self.tracking_updates
                tracker.update_method = BayesianUpdates.APPROXIMATE
                tracker.features = self.features

                # Append objects and set volume
                tracker.append(tracking_data)
                tracker.volume = ((0, xmax), (0, ymax))  # , (10e-6, 10e-6))
                # Run tracking
                tracker.track(step_size=10000)
                tracker.optimize()

                # Get results in napari format for merging
                data, properties, graph = tracker.to_napari()

        # Convert to DataFrame
        return pd.DataFrame(data, columns=["trackid", "t", "y", "x"])

    def _merge_tracking_results(
        self, measurements_df: pd.DataFrame, tracks_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge tracking results with original measurements.

        Args:
            measurements_df: Original measurements
            tracks_df: Tracking results

        Returns:
            Merged DataFrame
        """
        # Convert back to original column names for merging
        merge_df = tracks_df.rename(
            columns={"t": "frame", "y": "centroid_0", "x": "centroid_1"}
        )

        out = merge_tracking_results(
            measurements_df,
            merge_df[["trackid", "frame", "centroid_0", "centroid_1"]],
            merge_on=["frame", "centroid_0", "centroid_1"],
        )
        # Set trackid to integer so that it can be read by TrackMate
        out["trackid"] = out["trackid"].fillna(-1).astype("int32")
        return out

    def _configure_btrack_logging(self, target_logger: logging.Logger) -> None:
        """Configure all btrack-related loggers to use our target logger."""
        # Get all potential btrack loggers. This is necessary in order to avoid the
        # btrack loggers from writing over the image analysis loggers.
        btrack_loggers = [
            logging.getLogger("btrack"),
            logging.getLogger("btrack.core"),
            logging.getLogger("btrack.utils"),
        ]

        for btrack_logger in btrack_loggers:
            # Clear existing handlers
            btrack_logger.handlers.clear()
            # Add our target logger's handlers
            for handler in target_logger.handlers:
                btrack_logger.addHandler(handler)
            btrack_logger.setLevel(target_logger.level)
            btrack_logger.propagate = False
