import os
import sys

# Load local modules
from HiTMicTools.confreader import ConfReader
from HiTMicTools.pipelines import get_pipeline
from HiTMicTools.utils import check_btrack


def build_and_run_pipeline(config_file: str, worklist: str = None):
    """
    Build and run the image analysis pipeline based on configuration.

    Args:
        config_file (str): Path to the configuration file
        worklist (str, optional): Path to the worklist file. Defaults to None.
    """
    # Initialize worklist_id and file_list
    c_reader = ConfReader(config_file)
    configs = c_reader.opt

    if worklist is not None and worklist.strip():
        if not os.path.isfile(worklist):
            print(f"Error: Worklist file not found: {worklist}")
            sys.exit(1)
        print(f"Using worklist: {worklist}")
        # No need to extract worklist_id or update configs here
    else:
        # Set worklist to None explicitly when not provided
        worklist = None

    extra_args = configs.get("extra", {})
    num_workers = configs.pipeline_setup.get("num_workers", None)

    # Get pipeline class from registry
    pipeline_name = configs.pipeline_setup["name"]
    try:
        pipeline_metadata = get_pipeline(pipeline_name)
        analysis_pipeline = pipeline_metadata.cls
        print(f"Pipeline: {pipeline_name}")
        print(f"Required models: {', '.join(sorted(pipeline_metadata.required_models))}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    analysis_wf = analysis_pipeline(
        configs.input_data["input_folder"],
        configs.input_data["output_folder"],
        file_type=configs.input_data["file_type"],
        worklist_path=worklist,
    )

    analysis_wf.load_config_dict(configs.pipeline_setup)
    model_bundle = configs.get("models", {}).get("model_collection")
    if model_bundle:
        # Use the load_model_bundle method if a bundle is provided (always selective)
        analysis_wf.load_model_bundle(model_bundle)
    else:
        # Load models individually - ONLY those required by this pipeline
        for model_key in pipeline_metadata.required_models:
            if model_key in configs:
                analysis_wf.load_model_fromdict(model_key, configs[model_key])
            else:
                print(f"Warning: Required model '{model_key}' not found in config")

    # Load tracker if tracking is enabled and config is provided
    if configs.pipeline_setup.get("tracking", False):
        tracking_config = configs.get("tracking", {})
        tracker_backend = tracking_config.get("backend", "btrack")

        if tracker_backend == "hungarian":
            hungarian_params = tracking_config.get("parameters", {})
            analysis_wf.load_tracker(
                tracker_backend="hungarian",
                tracker_config=hungarian_params,
            )
            print(
                f"Tracking enabled: Hungarian "
                f"(max_distance={hungarian_params.get('max_distance', 25.0)})"
            )

        elif tracker_backend == "btrack":
            if not check_btrack():
                print(
                    "\033[1mError: btrack package is missing or not properly compiled. "
                    "Please check the installation README at "
                    "https://github.com/phisanti/HiTMicTools for proper btrack "
                    "compilation instructions.\033[0m"
                )
                sys.exit(1)

            tracker_override_args = tracking_config.get("parameters_override", None)
            config_path = tracking_config.get("config_path")

            if model_bundle:
                analysis_wf.load_tracker(
                    model_bundle, tracker_override_args=tracker_override_args
                )
                print(f"Tracking enabled with config from bundle: {model_bundle}")
            elif config_path:
                analysis_wf.load_tracker(
                    config_path, tracker_override_args=tracker_override_args
                )
                print(f"Tracking enabled with config: {config_path}")
            else:
                print("Warning: Tracking enabled but no config source provided")
                sys.exit(1)

        else:
            print(f"Error: Unknown tracker backend: {tracker_backend}")
            sys.exit(1)

        # Optionally load learned cost / division models (backwards-compatible)
        analysis_wf.load_learned_trackers(tracking_config)

    else:
        print("Tracking disabled")

    export_labeled_mask = configs.input_data["export_labelled_masks"]
    export_aligned_image = configs.input_data.get("export_aligned_image", False)
    export_training_crops = configs.input_data.get("export_training_crops", False)
    training_crop_size = configs.input_data.get("training_crop_size", 64)

    if configs.pipeline_setup["parallel_processing"]:
        analysis_wf.process_folder_parallel(
            files_pattern=configs.input_data["file_pattern"],
            export_labeled_mask=export_labeled_mask,
            export_aligned_image=export_aligned_image,
            num_workers=num_workers,
            export_training_crops=export_training_crops,
            training_crop_size=training_crop_size,
            **extra_args,
        )
    else:
        analysis_wf.process_folder(
            files_pattern=configs.input_data["file_pattern"],
            export_labeled_mask=export_labeled_mask,
            export_aligned_image=export_aligned_image,
            export_training_crops=export_training_crops,
            training_crop_size=training_crop_size,
            **extra_args,
        )
