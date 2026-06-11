import os
import unittest
import tempfile
import shutil

from HiTMicTools.confreader import ConfReader
from HiTMicTools.pipelines.ASCT_semSeg import ASCT_semSeg
from HiTMicTools.pipelines.ASCT_instSeg import ASCT_instSeg
from HiTMicTools.pipelines.base_pipeline import BasePipeline


class TestPipelineConfigLoading(unittest.TestCase):
    def setUp(self):
        self.test_config = "./config/templates/test_model_bundle.yml"
        self.pipeline_map = {
            "ASCT_semSeg": ASCT_semSeg,
            "ASCT_instSeg": ASCT_instSeg,
        }

    def test_pipeline_config_loading(self):
        # 1. Test config file exists
        self.assertTrue(
            os.path.exists(self.test_config),
            f"Config file not found: {self.test_config}",
        )

        # 2. Load configuration
        c_reader = ConfReader(self.test_config)
        configs = c_reader.opt

        # 3. Pipeline initialization
        pipeline_name = configs.pipeline_setup["name"]
        self.assertIn(
            pipeline_name, self.pipeline_map, f"Invalid pipeline name: {pipeline_name}"
        )
        analysis_pipeline = self.pipeline_map[pipeline_name]
        analysis_wf = analysis_pipeline(
            configs.input_data["input_folder"],
            configs.input_data["output_folder"],
            file_type=configs.input_data["file_type"],
        )
        analysis_wf.load_config_dict(configs.pipeline_setup)

        # 4. Model loading
        model_bundle = configs.get("models", {}).get("model_collection")
        if model_bundle and os.path.exists(model_bundle):
            analysis_wf.load_model_bundle(model_bundle)
        # else: skip if not present

        # 5. Tracker loading
        if configs.pipeline_setup.get("tracking", False):
            tracking_config = configs.get("tracking", {})
            tracker_override_args = tracking_config.get("parameters_override", None)
            config_path = tracking_config.get("config_path")
            if model_bundle and os.path.exists(model_bundle):
                try:
                    analysis_wf.load_tracker(
                        model_bundle, tracker_override_args=tracker_override_args
                    )
                except Exception:
                    if config_path and os.path.exists(config_path):
                        analysis_wf.load_tracker(
                            config_path, tracker_override_args=tracker_override_args
                        )
            elif config_path and os.path.exists(config_path):
                analysis_wf.load_tracker(
                    config_path, tracker_override_args=tracker_override_args
                )
            # else: skip if not present


class TestOofDetectorLoading(unittest.TestCase):
    """Test OofDetector loading through workflows."""

    def setUp(self):
        """Create temporary directories for test pipeline."""
        self.temp_input = tempfile.mkdtemp(prefix="test_input_")
        self.temp_output = tempfile.mkdtemp(prefix="test_output_")

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.temp_input, ignore_errors=True)
        shutil.rmtree(self.temp_output, ignore_errors=True)

    def test_oof_detector_loading_from_dict(self):
        """Test loading OofDetector via load_model_fromdict."""
        # Create minimal pipeline with required_models
        class MinimalPipeline(BasePipeline):
            required_models = {"oof_detector"}  # Define required models

            def analyse_image(self, *args, **kwargs):
                pass

        pipeline = MinimalPipeline(
            input_path=self.temp_input,
            output_path=self.temp_output
        )

        # Mock config for OofDetector
        config_dict = {
            "model_path": "./models/oof_detection/oof_baserfdetr.pth",
            "model_metadata": "./models/oof_detection/oof_baserfdetr_config.json",
            "inferer_args": {
                "patch_size": 560,
                "overlap_ratio": 0.25,
                "score_threshold": 0.5,
                "nms_iou": 0.5,
                "class_dict": {"oof": 0}
            }
        }

        # Skip test if model files don't exist
        if not os.path.exists(config_dict["model_path"]):
            self.skipTest("OOF detector model files not available")

        # Load OofDetector
        try:
            pipeline.load_model_fromdict("oof-detector", config_dict)
            self.assertIsNotNone(pipeline.oof_detector)
            self.assertIsNotNone(pipeline.oof_class_map)
            self.assertEqual(pipeline.oof_class_map, {"oof": 0})
        except Exception as e:
            self.fail(f"OofDetector loading failed: {e}")


if __name__ == "__main__":
    unittest.main()
