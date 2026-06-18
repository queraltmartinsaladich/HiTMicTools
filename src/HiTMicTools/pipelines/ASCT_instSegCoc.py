from HiTMicTools.pipelines.base_instSeg import BaseInstSeg


class ASCT_instSegCoc(BaseInstSeg):
    """instSeg pipeline for cocci (spherical) bacteria.

    Intended species: S. aureus.

    Uses a model bundle trained on coccal morphologies
    (model_collection_instSegCoc.zip).  All processing logic is
    inherited from BaseInstSeg; the model bundle is the only
    differentiator.
    """
