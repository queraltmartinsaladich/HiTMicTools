import os
import glob
import time
import fnmatch
import zipfile
import tempfile
import yaml
import logging
import subprocess
import re
import shutil

# Resources imports
import concurrent.futures
import gc
import multiprocessing
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath

# Type annotations and
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
import numpy as np

# Local imports
import HiTMicTools
from HiTMicTools import __version__
from HiTMicTools.resource_management.memlogger import MemoryLogger
from HiTMicTools.model_components.segmentation_model import Segmentator
from HiTMicTools.model_components.cell_classifier import CellClassifier
from HiTMicTools.model_components.focus_restorer import FocusRestorer
from HiTMicTools.model_components.oof_detector import OofDetector
from HiTMicTools.model_components.scsegmenter import ScSegmenter
# CellTracker imported lazily in load_tracker() to avoid btrack dependency
# when using HungarianTracker backend
from HiTMicTools.resource_management.sysutils import get_device, get_system_info
from HiTMicTools.model_arch.nafnet import NAFNet
from HiTMicTools.model_arch.flexresnet import FlexResNet
from HiTMicTools.model_components.pi_classifier import PIClassifier
from monai.networks.nets import UNet as monai_unet
from HiTMicTools.utils import read_metadata, update_config


@contextmanager
def managed_resource(*objects):
    yield objects
    for obj in objects:
        del obj
    gc.collect()


def _get_hitmictools_source_path() -> str:
    """Return the imported package directory for provenance logs."""
    return str(Path(HiTMicTools.__file__).resolve().parent)


def _get_hitmictools_git_commit(source_path: str) -> str:
    """Return the current git commit for editable/source installs, if available."""
    try:
        commit_result = subprocess.run(
            ["git", "-C", source_path, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        commit = commit_result.stdout.strip()

        status_result = subprocess.run(
            ["git", "-C", source_path, "status", "--short"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        if status_result.stdout.strip():
            return f"{commit}+dirty"
        return commit
    except Exception:
        return "unknown"


def _get_source_pyproject_version(source_path: str) -> str:
    """Return pyproject project.version for source checkouts, if available."""
    for path in [Path(source_path), *Path(source_path).parents]:
        pyproject_path = path / "pyproject.toml"
        if not pyproject_path.exists():
            continue
        try:
            in_project_section = False
            for line in pyproject_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped == "[project]":
                    in_project_section = True
                    continue
                if stripped.startswith("[") and stripped.endswith("]"):
                    in_project_section = False
                if in_project_section:
                    match = re.match(r'version\s*=\s*"([^"]+)"', stripped)
                    if match:
                        return match.group(1)
        except OSError:
            return "unknown"
    return "unknown"


def get_hitmictools_provenance() -> Dict[str, str]:
    """Collect version/source provenance for analysis logs."""
    source_path = _get_hitmictools_source_path()
    return {
        "version": __version__,
        "source_version": _get_source_pyproject_version(source_path),
        "source_path": source_path,
        "git_commit": _get_hitmictools_git_commit(source_path),
    }


def _safe_zip_target_path(member_name: str, target_dir: str) -> Path:
    """Return a validated extraction target for a ZIP member."""
    normalized_name = member_name.replace("\\", "/")
    posix_path = PurePosixPath(normalized_name)
    windows_path = PureWindowsPath(member_name)

    if (
        not normalized_name
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part == ".." for part in posix_path.parts)
    ):
        raise ValueError(f"Unsafe path in model bundle: {member_name}")

    target_root = Path(target_dir).resolve()
    target_path = (target_root / Path(*posix_path.parts)).resolve()
    try:
        target_path.relative_to(target_root)
    except ValueError as exc:
        raise ValueError(f"Unsafe path in model bundle: {member_name}") from exc

    return target_path


def _safe_extract_zip(zip_file: zipfile.ZipFile, target_dir: str) -> None:
    """Extract a ZIP file after rejecting paths outside target_dir."""
    for member in zip_file.infolist():
        target_path = _safe_zip_target_path(member.filename, target_dir)
        if member.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with zip_file.open(member) as source, open(target_path, "wb") as target:
            shutil.copyfileobj(source, target)


class BasePipeline(ABC):
    """
    An abstract base class for performing standard analysis on microscopy images.

    This class provides the framework for image analysis pipelines but requires
    subclasses to implement the analyse_image method for specific analysis tasks.

    Methods:
        setup_logger: Set up a logger for logging the analysis progress.
        remove_logger: Remove logger, useful for concurrent parallel processing.
        load_model_fromdict: Load a model based on the specified model type and configuration dictionary.
        load_model_bundle: Load models and configurations from a bundled zip file.
        load_config_dict: Configure image analysis settings from a dictionary.
        config_image_analysis: Configure the image analysis settings.
        get_files: Retrieve a list of files from the specified input path, filtered by pattern and extension.
        process_folder: Process all files with the matching pattern and file extension in the input folder.
        process_folder_parallel: Process multiple image files in parallel using multiprocessing.
        analyse_image: (Abstract) Analyze a single image file. Must be implemented by subclasses.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        worklist_path: str = None,
        file_type: str = ".nd2",
    ):
        """Initialize the BasePipeline.

        Args:
            input_path (str): Path to the input directory containing the images.
            output_path (str): Path to the output directory for saving the analysis results.
            worklist_path (str, optional): Path to the worklist file. Defaults to None.
            file_type (str, optional): File extension of the image files. Defaults to '.nd2'.
        """
        self.input_path = input_path
        self.worklist_path = worklist_path
        last_folder = os.path.basename(os.path.normpath(self.input_path))

        worklist_id = ""
        if worklist_path:
            worklist_id = os.path.basename(worklist_path).split(".")[0]

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.main_logger = self.setup_logger(
            output_path, last_folder, logger_id=worklist_id, print_output=True
        )

        self.output_path = output_path
        self.file_type = file_type

        # Model attributes — initialised to None so pipelines can safely check
        # `if self.x is not None` without calling load_model_bundle first.
        self.pi_classifier = None
        self.cell_tracker = None
        self.bf_focus_restorer = None
        self.fl_focus_restorer = None
        self.assignment_scorer = None
        self.division_classifier = None

    def setup_logger(
        self,
        output_path: str,
        name: str,
        logger_id: str = "",
        print_output: bool = False,
    ) -> logging.Logger:
        """Set up a logger for logging the analysis progress."""

        # Set up logger file
        logging.setLoggerClass(MemoryLogger)
        os.path.basename(os.path.normpath(name))
        log_file = os.path.join(output_path, f"{name}_{logger_id}_analysis.log")
        logger_name = f"{output_path}_{name}_{logger_id}"  # Use a unique identifier for each instance important for parallelisation
        logger = logging.getLogger(logger_name)

        # Set logger level and format
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Set logger for console if required
        if print_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def remove_logger(self, logger):
        """Remove logger, useful for concurrent parallel processing."""

        # Remove all handlers from the logger
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        # Delete the logger instance
        del logging.Logger.manager.loggerDict[logger.name]

    def log_runtime_provenance(self) -> None:
        """Log enough package provenance to debug stale installs."""
        provenance = get_hitmictools_provenance()
        debug_provenance = bool(getattr(self, "debug_provenance", False))
        self.main_logger.info(
            f"Running hitmictools version {provenance['version']}"
        )
        self.main_logger.info(
            f"HiTMicTools git commit: {provenance['git_commit']}"
        )
        if debug_provenance:
            self.main_logger.info(
                f"HiTMicTools source pyproject version: {provenance['source_version']}"
            )
            self.main_logger.info(
                f"HiTMicTools source path: {provenance['source_path']}"
            )

    def load_model_fromdict(self, model_type: str, config_dic: Dict[str, Any]) -> None:
        """
        Load a model based on the specified model type and configuration dictionary.

        Args:
            model_type (str): Type of the model to load ('segmentator', 'segmentator2',
                            'cell-classifier', 'focus-restorer-fl', 'focus-restorer-bf', 'pi-classifier').
            config_dic (Dict[str, Any]): Dictionary containing model configuration including:
                            - 'model_path': Path to model weights file
                            - 'model_metadata': Path to model metadata (except for pi-classifier)
                            - 'inferer_args' or 'model_args': Additional configuration parameters

        Returns:
            None: The model is loaded and attached to the appropriate attribute in the class.

        Raises:
            ValueError: If an invalid model type is provided or required arguments are missing.
            KeyError: If required configuration keys are missing.
        """
        alias_map = {
            "segmentation": "segmentator",
            "cell_classifier": "cell-classifier",
            "bf_focus": "focus-restorer-bf",
            "fl_focus": "focus-restorer-fl",
            "pi_classification": "pi-classifier",
            "oof_detector": "oof-detector",
            "sc_segmenter": "sc-segmenter",
        }
        model_type = alias_map.get(model_type, model_type)

        if "model_path" not in config_dic:
            raise KeyError(
                f"Required key 'model_path' missing from configuration for {model_type}"
            )

        model_path = config_dic["model_path"]
        if model_type != "pi-classifier":
            if "model_metadata" not in config_dic:
                raise KeyError(
                    f"Required key 'model_metadata' missing from configuration for {model_type}"
                )
            model_configs = read_metadata(config_dic["model_metadata"])

        # Get compile_mode from pipeline config (defaults to False = no compilation)
        compile_mode = getattr(self, "compile_models", False)

        if model_type == "segmentator":
            model_graph = monai_unet(**model_configs["model_args"])
            self.image_segmentator = Segmentator(
                model_path, model_graph=model_graph, compile_mode=compile_mode, **config_dic["inferer_args"]
            )
            self.main_logger.info("Loaded model: segmentation (Monai UNet)")
        elif model_type == "cell-classifier":
            model_graph = FlexResNet(**model_configs["model_args"])
            self.object_classifier = CellClassifier(
                model_path, model_graph=model_graph, compile_mode=compile_mode, **config_dic["model_args"]
            )
            self.main_logger.info("Loaded model: cell_classifier (FlexResNet)")
        elif model_type == "focus-restorer-fl":
            model_graph = NAFNet(**model_configs["model_args"])
            self.fl_focus_restorer = FocusRestorer(
                model_path, model_graph=model_graph, compile_mode=compile_mode, **config_dic["inferer_args"]
            )
            self.main_logger.info("Loaded model: fl_focus (NAFNet)")
        elif model_type == "focus-restorer-bf":
            model_graph = NAFNet(**model_configs["model_args"])
            self.bf_focus_restorer = FocusRestorer(
                model_path, model_graph=model_graph, compile_mode=compile_mode, **config_dic["inferer_args"]
            )
            self.main_logger.info("Loaded model: bf_focus (NAFNet)")
        elif model_type == "pi-classifier":
            self.pi_classifier = PIClassifier(model_path)
            self.main_logger.info("Loaded model: pi_classification (scikit-learn)")
        elif model_type == "oof-detector":
            self.oof_detector = OofDetector(
                model_path,
                model_type=model_configs.get("model_type", "rfdetrbase"),
                compile_mode=compile_mode,
                **config_dic.get("inferer_args", {}),
            )
            self.oof_class_map = config_dic.get("inferer_args", {}).get("class_dict")
            self.main_logger.info("Loaded model: oof_detector (RF-DETR)")
        elif model_type == "sc-segmenter":
            detector_backend = model_configs.get("model_type", "rfdetrsegpreview")
            self.sc_segmenter = ScSegmenter(
                model_path,
                model_type=detector_backend,
                compile_mode=compile_mode,
                **config_dic.get("inferer_args", {}),
            )
            self.class_dict = config_dic.get("inferer_args", {}).get("class_dict")
            self.main_logger.info(f"Loaded model: sc_segmenter (backend={detector_backend})")
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    def load_model_bundle(self, path_to_bundle: str) -> None:
        """
        Load models and configurations from a model bundle.
        The model bundle must be a zip file with the following structure:
        model_bundle.zip
        ├── config.yml          # Main configuration file
        ├── models/             # Directory containing model weights
        │   ├── model_x.pth
        └── metadata/           # Directory containing model metadata
            ├── model_x.json

        Only models required by this pipeline (defined in required_models class attribute)
        will be loaded from the bundle.

        Args:
            path_to_bundle (str): Path to the model bundle zip file.

        Raises:
            FileNotFoundError: If the bundle path does not exist or is not a file.
            ValueError: If the bundle is not a zip file or has an invalid structure.
            AttributeError: If the pipeline does not define required_models.
        """

        if not os.path.isfile(path_to_bundle):
            raise FileNotFoundError(f"Model bundle not found at {path_to_bundle}")
        if not path_to_bundle.endswith(".zip"):
            raise ValueError("Model bundle must be a .zip file")

        # Get pipeline's required models - MANDATORY for selective loading
        required_models = getattr(self, 'required_models', None)
        if not required_models:
            raise AttributeError(
                f"Pipeline '{self.__class__.__name__}' does not have 'required_models' class attribute.\n"
                f"Please define required_models in the pipeline class as a set of model keys.\n"
            )

        self.main_logger.info(
            f"Selective loading enabled for {self.__class__.__name__}. "
            f"Required models: {', '.join(sorted(required_models))}"
        )

        # Define mapping between config keys and internal model types for proper loading
        model_type_mapping = {
            "bf_focus": "focus-restorer-bf",
            "fl_focus": "focus-restorer-fl",
            "segmentation": "segmentator",
            "cell_classifier": "cell-classifier",
            "pi_classification": "pi-classifier",
            "oof_detector": "oof-detector",
            "sc_segmenter": "sc-segmenter",
        }

        try:
            with zipfile.ZipFile(path_to_bundle, "r") as bundle_zip:
                namelist = bundle_zip.namelist()
                # Verify bundle structure has all required components
                required_items = ["config.yml", "models/", "metadata/"]
                for item in required_items:
                    if not any(name.startswith(item) for name in namelist):
                        raise ValueError(
                            f"Invalid model bundle structure: Missing {item}"
                        )

                with tempfile.TemporaryDirectory() as temp_dir:
                    _safe_extract_zip(bundle_zip, temp_dir)

                    # Load the main configuration file
                    config_path = os.path.join(temp_dir, "config.yml")
                    with open(config_path, "r") as config_file:
                        config = yaml.safe_load(config_file)

                    # Track loaded and skipped models for summary
                    loaded_models = []
                    skipped_models = []

                    # Process each model in the bundle
                    for model_key, model_config in config.items():
                        if model_key in model_type_mapping:
                            # Skip models not required by this pipeline (ALWAYS selective)
                            if model_key not in required_models:
                                self.main_logger.info(
                                    f"Skipping {model_key} (not required by {self.__class__.__name__})"
                                )
                                skipped_models.append(model_key)
                                continue

                            model_type = model_type_mapping[model_key]

                            # Update paths to point to files in the temporary directory
                            if model_key == "pi_classification":
                                model_config["model_path"] = os.path.join(
                                    temp_dir, model_config["model_path"]
                                )
                            else:
                                model_config["model_path"] = os.path.join(
                                    temp_dir, model_config["model_path"]
                                )
                                model_config["model_metadata"] = os.path.join(
                                    temp_dir, model_config["model_metadata"]
                                )

                            # Load the model
                            self.main_logger.info(f"Loading {model_key}...")
                            self.load_model_fromdict(model_type, model_config)
                            loaded_models.append(model_key)
                        else:
                            self.main_logger.warning(
                                f"Unknown model key in bundle: {model_key}"
                            )

        except zipfile.BadZipFile:
            raise ValueError(f"Invalid or corrupted zip file: {path_to_bundle}")
        except Exception as e:
            self.main_logger.error(f"Error loading model bundle: {e}")
            raise

        # Log summary of loaded models
        self.main_logger.info(
            f"Successfully loaded model bundle: {path_to_bundle}"
        )
        self.main_logger.info(
            f"Models loaded ({len(loaded_models)}): {', '.join(loaded_models)}"
        )
        if skipped_models:
            self.main_logger.info(
                f"Models skipped ({len(skipped_models)}): {', '.join(skipped_models)}"
            )

    def load_tracker(
        self,
        config_path: str = None,
        tracker_override_args: Optional[Dict[str, Any]] = None,
        tracker_backend: str = "btrack",
        tracker_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Load and configure cell tracker.

        Args:
            config_path: Path to config file (.yml/.json) or zip bundle (btrack only)
            tracker_override_args: Optional dict to override tracker parameters (btrack only)
            tracker_backend: Tracker backend - "btrack" or "hungarian"
            tracker_config: Backend-specific parameters (hungarian only). For the
                Hungarian backend, optional key "version" selects the
                implementation; omitted defaults to the legacy v2 tracker.
        """
        if tracker_backend == "btrack" and config_path and config_path.endswith(".zip"):
            with zipfile.ZipFile(config_path, "r") as _z:
                if "config_tracker.yml" not in _z.namelist():
                    yml_files = [n for n in _z.namelist() if n.endswith(".yml")]
                    raise FileNotFoundError(
                        f"backend: btrack requires config_tracker.yml in the bundle ZIP, "
                        f"but it was not found in {config_path}. "
                        f"Found YML files: {yml_files}. "
                        f"Use model_collection_semSegTrack.zip (not model_collection_semSeg.zip)."
                    )

        if tracker_backend == "hungarian":
            from HiTMicTools.tracking.hungarian_tracker import HungarianTracker
            from HiTMicTools.tracking.hungarian_tracker_v5 import HungarianTrackerV5

            params = dict(tracker_config or {})
            tracker_version = str(params.pop("version", "v2")).lower()
            tracker_classes = {
                "v1": HungarianTracker,
                "v2": HungarianTracker,
                "legacy": HungarianTracker,
                "v5": HungarianTrackerV5,
            }
            if tracker_version not in tracker_classes:
                valid_versions = ", ".join(sorted(tracker_classes))
                raise ValueError(
                    f"Unknown Hungarian tracker version: {tracker_version}. "
                    f"Expected one of: {valid_versions}"
                )

            tracker_class = tracker_classes[tracker_version]
            self.cell_tracker = tracker_class(**params)
            self.main_logger.info(
                f"Hungarian tracker {tracker_version} loaded "
                f"(max_distance={self.cell_tracker.max_distance})"
            )

        elif tracker_backend == "btrack":
            from HiTMicTools.tracking.cell_tracker import CellTracker

            if config_path is None:
                raise ValueError("config_path is required for btrack backend")

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Tracker config not found: {config_path}")

            if config_path.endswith(".zip"):
                config = self._load_tracker_config_from_zip(config_path)
                if tracker_override_args:
                    config = update_config(
                        config, tracker_override_args, logger=self.main_logger
                    )
                self.cell_tracker = CellTracker(config_dict=config)
            else:
                override_args = tracker_override_args or {}
                self.cell_tracker = CellTracker(
                    config_dict=config_path, override_args=override_args
                )

            self.main_logger.info(f"Cell tracker loaded from: {config_path}")

        else:
            raise ValueError(f"Unknown tracker_backend: {tracker_backend}")

    def load_learned_trackers(self, tracking_config: Dict[str, Any]) -> None:
        """
        Optionally load learned assignment scorer and division classifier.

        Reads two optional keys from the tracking config section:
          learned_cost_model:     path to assignment_scorer.pt
          learned_division_model: path to division_classifier.pt

        When absent or null, the pipeline falls back to Euclidean cost +
        reconcile_lineage (existing behaviour, fully backwards-compatible).
        """
        self.assignment_scorer = None
        self.division_classifier = None

        cost_path = tracking_config.get("learned_cost_model")
        if cost_path:
            from HiTMicTools.tracking.assignment_scorer import AssignmentScorer
            self.assignment_scorer = AssignmentScorer(cost_path)
            self.main_logger.info(
                f"Loaded learned assignment scorer from {cost_path} "
                f"(threshold={self.assignment_scorer.threshold:.2f})"
            )

        div_path = tracking_config.get("learned_division_model")
        if div_path:
            from HiTMicTools.tracking.division_classifier import DivisionClassifier
            self.division_classifier = DivisionClassifier(div_path)
            self.main_logger.info(
                f"Loaded learned division classifier from {div_path} "
                f"(threshold={self.division_classifier.threshold:.2f})"
            )

    def _load_tracker_config_from_zip(self, zip_path: str) -> Dict[str, Any]:
        """Load tracker config from zip file."""
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            if "config_tracker.yml" not in zip_ref.namelist():
                raise FileNotFoundError("config_tracker.yml not found in zip root")

            with zip_ref.open("config_tracker.yml") as config_file:
                return yaml.safe_load(config_file)

    def _load_species_config(
        self,
        species: str,
        custom_config_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load species-specific preprocessing parameters from a YAML config file.

        Looks up the species entry in the bundled ``species_defaults.yaml`` unless a
        custom path is provided via *custom_config_path*.  Species keys are
        case-insensitive.

        Args:
            species: Species identifier, e.g. ``"ecoli"``, ``"s_aureus"``.
            custom_config_path: Optional path to a user-supplied species YAML.
                If *None* or the file does not exist, the bundled defaults are used.

        Returns:
            Dict with operator configurations for the given species, or an empty
            dict if the species is not found.
        """
        bundled_path = str(
            Path(__file__).resolve().parent.parent
            / "img_processing"
            / "species_defaults.yaml"
        )

        config_path = bundled_path
        if custom_config_path and os.path.isfile(custom_config_path):
            config_path = custom_config_path
            self.main_logger.info(
                f"Species preprocessing: using custom config from {custom_config_path}"
            )
        else:
            self.main_logger.info(
                f"Species preprocessing: using bundled defaults ({bundled_path})"
            )

        with open(config_path, "r") as fh:
            all_species = yaml.safe_load(fh)

        species_key = species.lower()
        species_cfg = (all_species.get("species") or {}).get(species_key)

        if species_cfg is None:
            self.main_logger.warning(
                f"No preprocessing config found for species '{species}'. "
                "No species-specific operators will be applied."
            )
            return {}

        self.main_logger.info(
            f"Species preprocessing config loaded for '{species_key}': "
            + ", ".join(
                op for op, cfg in species_cfg.items() if isinstance(cfg, dict) and cfg.get("enabled")
            )
        )
        return species_cfg

    def _get_morphology_kwargs(self) -> dict:
        """Return species-specific kwargs for apply_instSeg_morphology_corrections.

        Reads the ``morphology_corrections`` block from the species config.
        Returns an empty dict if no species is set or no block is present,
        so the function falls back to its own defaults.
        """
        species = getattr(self, "species", None)
        if not species:
            return {}
        species_config_path = getattr(self, "species_config_path", None)
        species_cfg = self._load_species_config(species, species_config_path)
        morph = species_cfg.get("morphology_corrections", {})
        return dict(morph) if morph else {}

    def config_image_analysis(
        self,
        reference_channel: int,
        align_frames: bool = False,
        method: str = "basicpy_fl",
    ) -> None:
        """
        Configure the image analysis settings.

        Args:
            reference_channel (int): The reference channel for image analysis.
            align_frames (bool, optional): Whether to align frames. Defaults to False.
            method (str, optional): The method to use for image analysis. Defaults to "basicpy".
        """
        self.reference_channel = reference_channel
        self.align_frames = align_frames
        self.method = method

    # Valid compile modes for torch.compile
    VALID_COMPILE_MODES = {"default", "reduce-overhead", "max-autotune", False}

    def load_config_dict(self, config_dict: Dict) -> None:
        """Configure image analysis settings from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters
                - reference_channel (int): Reference channel index
                - align_frames (bool): Whether to align frames
                - method (str): Background correction method
                - focus_correction (bool): Whether to apply focus correction
                - compile_models (str or False, optional): Torch compile mode.
                    Options: "default", "reduce-overhead", "max-autotune", or False.
                    Defaults to "max-autotune".
                - debug_provenance (bool, optional): If True, include local source
                    and interpreter paths in analysis logs. Defaults to False.
                - species (str, optional): Species key for species-aware preprocessing.
                    E.g. "ecoli", "s_aureus", "m_tuberculosis". When set, the bundled
                    species_defaults.yaml is loaded and species-specific operators are
                    applied during preprocessing.
                - species_config_path (str, optional): Path to a custom species YAML
                    that overrides the bundled defaults.
                - pi_channel (int, optional): Fluorescence/PI channel index.

        Raises:
            ValueError: If required keys are missing or have invalid types
        """
        required_keys = {"reference_channel": int, "align_frames": bool, "method": str, "focus_correction": bool}

        # Validate required keys and types
        for key, expected_type in required_keys.items():
            if key not in config_dict:
                raise ValueError(f"Missing required key: {key}")
            if not isinstance(config_dict[key], expected_type):
                raise ValueError(f"Invalid type for {key}. Expected {expected_type}")

        # Validate compile_models if provided (default: False = no compilation)
        compile_mode = config_dict.get("compile_models", False)
        if compile_mode not in self.VALID_COMPILE_MODES:
            raise ValueError(
                f"Invalid compile_models: {compile_mode}. "
                f"Must be one of: {self.VALID_COMPILE_MODES}"
            )
        config_dict["compile_models"] = compile_mode

        # Set attributes
        for key, value in config_dict.items():
            setattr(self, key, value)

        # Log compile mode
        if compile_mode:
            self.main_logger.info(f"Models will be compiled with mode: {compile_mode}")
        else:
            self.main_logger.info("Models will not be compiled (compile_models: false)")

    def get_files(
        self,
        input_path: str,
        output_folder: str,
        pattern: str = None,
        no_reanalyse: bool = True,
    ) -> List[str]:
        """
        Retrieve a list of files from the specified input path, filtered by pattern and extension.

        Args:
            input_path (str): Path to the directory containing input files.
            output_folder (str): Path to the directory where output files will be saved.
            pattern (str, optional): File name pattern to match. Defaults to None.
            no_reanalyse (bool): If True, skip files that have already been analyzed. Defaults to True.

        Returns:
            List[str]: List of file basenames to be processed.
        """
        worklist_path = self.worklist_path
        if pattern is None:
            pattern = ""
        combined_pattern = f"{pattern}*{self.file_type}"

        # Initialize empty file list
        files_to_process = []

        # Case 1: Using a text file list
        if (
            worklist_path is not None
            and os.path.isfile(worklist_path)
            and worklist_path.endswith(".txt")
        ):
            with open(worklist_path, "r") as file:
                files_to_process = [line.strip() for line in file if line.strip()]
                # Ensure all files exist and match pattern
                files_to_process = [
                    f
                    for f in files_to_process
                    if os.path.exists(f)
                    and fnmatch.fnmatch(os.path.basename(f), combined_pattern)
                ]

        # Case 2: Using input directory
        elif os.path.isdir(input_path):
            files_to_process = glob.glob(os.path.join(input_path, combined_pattern))
        else:
            raise ValueError(
                f"Invalid input: {input_path}. Must be either a directory or a .txt file containing file paths."
            )

        if not files_to_process:
            self.main_logger.warning(
                f"No matching files found with pattern: {combined_pattern}"
            )
            return []

        # Remove already analyzed files if requested
        if no_reanalyse:
            filtered_files = []
            for file_path in files_to_process:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                full_output_path = os.path.join(output_folder, base_name)
                if not all(
                    os.path.exists(full_output_path + ext)
                    for ext in ["_summary.csv", "_fl.csv"]
                ):
                    filtered_files.append(file_path)
                else:
                    self.main_logger.info(
                        f"File {base_name} already analysed. Skipping."
                    )
            files_to_process = filtered_files

        # Return just the basenames for consistency
        return [os.path.basename(f) for f in files_to_process]

    def process_folder(
        self,
        files_pattern: Optional[str] = None,
        file_list: Optional[str] = None,
        export_labeled_mask: bool = False,
        export_aligned_image: bool = False,
        **kwargs,
    ) -> None:
        """
        Process all files with the matching pattern and file extension in the input folder.

        Args:
            files_pattern (str, optional): Glob pattern to match image files. Defaults to None.
            export_labeled_mask (bool): Whether to export labeled mask images. Defaults to False.
            export_aligned_image (bool): Whether to export aligned images. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the analyse_image method.

        Returns:
            None

        Notes:
            - Either files_pattern or file_list must be provided.
            - This method processes files sequentially, unlike process_folder_parallel.
            - The method will analyze each image file using the analyse_image method.
        """
        self.log_runtime_provenance()
        self.main_logger.info(f"Processing folder: {self.input_path}")
        self.main_logger.info(f"Output folder: {self.output_path}")
        self.main_logger.info(f"Files pattern: {files_pattern}")
        self.main_logger.info(f"File type: {self.file_type}")
        self.main_logger.info(
            get_system_info(include_paths=bool(getattr(self, "debug_provenance", False)))
        )

        file_list = self.get_files(
            self.input_path,
            self.output_path,
            files_pattern,
            no_reanalyse=True,
        )

        if not file_list:
            self.main_logger.warning("No files to process. Exiting.")
            return

        self.main_logger.info(
            f"{len(file_list)} files found with extension {self.file_type}"
        )

        start_time = time.time()
        for idx, name in enumerate(file_list, 1):
            self.main_logger.info(f"Processing file: {name}")
            self.main_logger.info(f"File number {idx} of {len(file_list)}")
            file_i = os.path.join(self.input_path, name)
            file_start_time = time.time()
            try:
                self.analyse_image(
                    file_i,
                    name,
                    export_labeled_mask=export_labeled_mask,
                    export_aligned_image=export_aligned_image,
                    **kwargs,
                )
                file_end_time = time.time()
                file_elapsed_time = file_end_time - file_start_time
                self.main_logger.info(
                    f"Job {name} has finished in time {file_elapsed_time:.2f} seconds"
                )
            except Exception as e:
                self.main_logger.error(f"Error processing file {name}: {str(e)}")

        end_time = time.time()
        total_elapsed_time = end_time - start_time
        self.main_logger.info(
            f"Total processing time for all files: {total_elapsed_time:.2f} seconds"
        )

    def process_folder_parallel(
        self,
        files_pattern: Optional[str] = None,
        file_list: Optional[str] = None,
        export_labeled_mask: bool = True,
        export_aligned_image: bool = True,
        num_workers: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Process multiple image files in parallel using multiprocessing.

        Args:
            files_pattern (str, optional): Glob pattern to match image files. Defaults to None.
            export_labeled_mask (bool): Whether to export labeled mask images. Defaults to True.
            export_aligned_image (bool): Whether to export aligned images. Defaults to True.
            num_workers (int, optional): Maximum number of worker processes. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the analyse_image method.

        Returns:
            None

        Notes:
            - Either files_pattern or file_list must be provided.
            - If num_workers is None, it defaults to half the number of CPU cores.
            - This method uses multiprocessing to analyze multiple images in parallel.
            - The analyse_image method is expected to handle its own return values.
        """
        file_list = self.get_files(
            self.input_path,
            self.output_path,
            files_pattern,
            no_reanalyse=True,
        )

        if not file_list:
            self.main_logger.warning("No files to process. Exiting.")
            return

        self.log_runtime_provenance()
        self.main_logger.info(f"Processing folder: {self.input_path}")
        self.main_logger.info(f"Output folder: {self.output_path}")
        self.main_logger.info(f"Files pattern: {files_pattern}")
        self.main_logger.info(f"File type: {self.file_type}")
        self.main_logger.info(
            get_system_info(include_paths=bool(getattr(self, "debug_provenance", False)))
        )
        self.main_logger.info(
            f"{len(file_list)} files found with extension {self.file_type}"
        )

        total_cpu_threads = os.cpu_count()
        if num_workers is None or num_workers == 0:
            num_workers = max(1, int(total_cpu_threads // 2))
        self.main_logger.info(f"Total CPU threads: {total_cpu_threads}")
        self.main_logger.info(f"Number of threads used: {num_workers}")
        self.main_logger.info(f"Total files to process: {len(file_list)}")
        start_time = time.time()  # Start timing the entire loop

        try:
            if get_device().type == "cuda":
                # IMPORTANT: spawn required for CUDA; ThreadPoolExecutor would only use
                # threads within individual cores, severely limiting parallelism.
                mp_context = multiprocessing.get_context("spawn")
                self.main_logger.info(
                    "Using spawn context with ProcessPoolExecutor for CUDA backend"
                )
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers, mp_context=mp_context
                )
            elif get_device().type == "mps":
                # IMPORTANT: macOS does not work well with ProcessPoolExecutor (deadlocks,
                # global state loss); ThreadPoolExecutor used instead (torch.compile disabled).
                self.main_logger.info("Using ThreadPoolExecutor for MPS backend")
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_workers
                )
            else:
                # IMPORTANT: fork required for CPU; spawn would cause global state loss.
                mp_context = multiprocessing.get_context("fork")
                self.main_logger.info(
                    "Using fork context with ProcessPoolExecutor for CPU backend"
                )
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=num_workers, mp_context=mp_context
                )

            with executor:
                futures = {}
                for index, name in enumerate(file_list, start=1):
                    file_i = os.path.join(self.input_path, name)
                    self.main_logger.info(
                        f"Submitting file number {index} of {len(file_list)}"
                    )
                    file_start_time = time.time()
                    future = executor.submit(
                        self.analyse_image,
                        file_i,
                        name,
                        export_labeled_mask,
                        export_aligned_image,
                        **kwargs,
                    )
                    futures[future] = (index, name, file_start_time)

                for future in concurrent.futures.as_completed(futures):
                    index, name, file_start_time = futures[future]
                    try:
                        with managed_resource(future):
                            future.result()
                        file_end_time = time.time()
                        file_elapsed_time = file_end_time - file_start_time
                        self.main_logger.info(
                            f"Job {name} has finished in time {file_elapsed_time:.2f} seconds. ({index}/{len(file_list)})"
                        )
                    except Exception as e:
                        self.main_logger.error(
                            f"Error processing file {index} ({name}): {str(e)}"
                        )
                    finally:
                        gc.collect()
        except Exception as e:
            self.main_logger.error(f"Error in parallel processing: {str(e)}")
        finally:
            end_time = time.time()
            total_elapsed_time = end_time - start_time
            self.main_logger.info(
                f"Total processing time for all files: {total_elapsed_time:.2f} seconds"
            )
            gc.collect()

    def _log_detection_summary(
        self,
        img_logger: MemoryLogger,
        n_frames: int,
        total_detections: int,
        total_instances: int,
        class_ids: List[np.ndarray],
        class_scores: List[np.ndarray],
    ) -> None:
        """Emit a human-friendly detection summary through the image-analysis logger."""
        segmenter_class_dict = self.sc_segmenter.class_dict
        class_counts = {class_name: 0 for class_name in segmenter_class_dict.values()}
        class_score_samples = {
            class_name: [] for class_name in segmenter_class_dict.values()
        }

        for frame_classes, frame_scores in zip(class_ids, class_scores):
            for cid, score in zip(frame_classes, frame_scores):
                class_name = segmenter_class_dict[int(cid)]
                class_counts[class_name] += 1
                class_score_samples[class_name].append(float(score))

        avg_detections = total_detections / n_frames if n_frames else 0.0
        summary_lines = [
            "[Pipeline] Detection summary:",
            f"  Total frames processed: {n_frames}",
            f"  Total bboxes detected: {total_detections}",
            f"  Total unique instances: {total_instances}",
            f"  Average detections per frame: {avg_detections:.1f}",
            "  Objects per class:",
        ]

        for class_name, count in class_counts.items():
            scores = class_score_samples.get(class_name, [])
            if scores:
                q05, q25, q50, q75, q95 = np.percentile(scores, [5, 25, 50, 75, 95])
                stats = (
                    f"conf_scores: q05={q05:.3f}, q25={q25:.3f}, "
                    f"q50={q50:.3f}, q75={q75:.3f}, q95={q95:.3f}"
                )
            else:
                stats = "conf_scores: q05=NA, q25=NA, q50=NA, q75=NA, q95=NA"
            summary_lines.append(f"    - {class_name}: {count}, {stats}")

        img_logger.info("\n".join(summary_lines))

    @abstractmethod
    def analyse_image(
        self,
        file_path: str,
        file_name: str,
        export_labeled_mask: bool = False,
        export_aligned_image: bool = False,
        **kwargs,
    ) -> None:
        """
        Analyze a single image file.

        This is an abstract method that must be implemented by subclasses.

        Args:
            file_path (str): Full path to the image file.
            file_name (str): Name of the image file.
            export_labeled_mask (bool): Whether to export labeled mask images.
            export_aligned_image (bool): Whether to export aligned images.
            **kwargs: Additional keyword arguments specific to the analysis method.

        Returns:
            None

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        pass
