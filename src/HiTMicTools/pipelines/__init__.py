"""Pipeline registry for HiTMicTools image analysis workflows.

This module provides a centralized registry of available pipelines and their
model requirements, enabling smart model loading and validation.

The registry automatically discovers model requirements from pipeline classes
via their `required_models` class attribute.
"""

from typing import Dict, Type, Set

from HiTMicTools.pipelines.base_pipeline import BasePipeline
from HiTMicTools.pipelines.ASCT_semSeg import ASCT_semSeg
from HiTMicTools.pipelines.ASCT_singleFrame import ASCT_singleFrame
from HiTMicTools.pipelines.ASCT_instSegRod import ASCT_instSegRod
from HiTMicTools.pipelines.ASCT_instSegCoc import ASCT_instSegCoc
from HiTMicTools.pipelines.ASCT_cellasic import ASCT_cellasic
from HiTMicTools.pipelines.ASCT_zaslavier import ASCT_zaslavier
from HiTMicTools.pipelines.oof_detection import OOF_detection


class PipelineMetadata:
    """Metadata for a pipeline including its class and required models.

    The required_models are automatically extracted from the pipeline class's
    `required_models` attribute.

    Attributes:
        cls: The pipeline class
        required_models: Set of model keys required by this pipeline
    """

    def __init__(self, cls: Type[BasePipeline]):
        self.cls = cls
        # Extract required_models from the class attribute
        self.required_models = getattr(cls, 'required_models', set())

        # Validate that pipeline has required_models defined - MANDATORY
        if not self.required_models:
            raise AttributeError(
                f"Pipeline '{cls.__name__}' does not have 'required_models' class attribute.\n"
                f"Please define required_models in the pipeline class as a set of model keys.\n"
            )

    def __repr__(self) -> str:
        return f"PipelineMetadata(cls={self.cls.__name__}, required_models={self.required_models})"


# Pipeline registry mapping pipeline names to their classes
# Model requirements are automatically discovered from each class
PIPELINE_REGISTRY: Dict[str, PipelineMetadata] = {
    "ASCT_semSeg": PipelineMetadata(ASCT_semSeg),
    "ASCT_singleFrame": PipelineMetadata(ASCT_singleFrame),
    "ASCT_instSegRod": PipelineMetadata(ASCT_instSegRod),
    "ASCT_instSegCoc": PipelineMetadata(ASCT_instSegCoc),
    "ASCT_cellasic": PipelineMetadata(ASCT_cellasic),
    "ASCT_zaslavier": PipelineMetadata(ASCT_zaslavier),
    "oof_detection": PipelineMetadata(OOF_detection),
}


def get_pipeline(name: str) -> PipelineMetadata:
    """Get pipeline metadata by name (case-insensitive lookup).

    Args:
        name: Name of the pipeline to retrieve

    Returns:
        PipelineMetadata containing the pipeline class and required models

    Raises:
        ValueError: If pipeline name is not found in registry

    Example:
        >>> metadata = get_pipeline("ASCT_instSeg")
        >>> pipeline_cls = metadata.cls
        >>> required = metadata.required_models
    """
    # Try exact match first
    metadata = PIPELINE_REGISTRY.get(name)

    # Fall back to case-insensitive match
    if metadata is None:
        normalized = {k.lower(): v for k, v in PIPELINE_REGISTRY.items()}
        metadata = normalized.get(name.lower())

    if metadata is None:
        available = ", ".join(PIPELINE_REGISTRY.keys())
        raise ValueError(f"Invalid pipeline name: '{name}'. Available pipelines: {available}")

    return metadata


def list_pipelines() -> Dict[str, Set[str]]:
    """List all available pipelines and their required models.

    Returns:
        Dictionary mapping pipeline names to their required model sets
    """
    return {name: meta.required_models for name, meta in PIPELINE_REGISTRY.items()}


__all__ = [
    "BasePipeline",
    "ASCT_semSeg",
    "ASCT_singleFrame",
    "ASCT_instSegRod",
    "ASCT_instSegCoc",
    "ASCT_cellasic",
    "ASCT_zaslavier",
    "OOF_detection",
    "PipelineMetadata",
    "PIPELINE_REGISTRY",
    "get_pipeline",
    "list_pipelines",
]
