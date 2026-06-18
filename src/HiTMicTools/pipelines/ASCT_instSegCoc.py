from HiTMicTools.pipelines.ASCT_instSeg import ASCT_instSeg


class ASCT_instSegCoc(ASCT_instSeg):
    """instSeg pipeline for cocci (spherical) bacteria.

    Intended species: S. aureus.

    Uses a model bundle trained on coccal morphologies
    (model_collection_instSegCoc.zip).  All processing logic is
    inherited from ASCT_instSeg; the model bundle is the only
    differentiator.
    """
