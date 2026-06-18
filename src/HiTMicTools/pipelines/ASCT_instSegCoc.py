from HiTMicTools.pipelines._instSeg_base import _InstSegBase


class ASCT_instSegCoc(_InstSegBase):
    """instSeg pipeline for cocci (spherical) bacteria.

    Intended species: S. aureus.

    Uses a model bundle trained on coccal morphologies
    (model_collection_instSegCoc.zip).  All processing logic is
    inherited from _InstSegBase; the model bundle is the only
    differentiator.
    """
