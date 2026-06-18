from HiTMicTools.pipelines.ASCT_instSeg import ASCT_instSeg


class ASCT_instSegRod(ASCT_instSeg):
    """instSeg pipeline for rod-shaped bacteria.

    Intended species: E. coli, P. aeruginosa, M. tuberculosis,
    M. abscessus, M. chimaera.

    Uses a model bundle trained on rod morphologies
    (model_collection_instSegRod.zip).  All processing logic is
    inherited from ASCT_instSeg; the model bundle is the only
    differentiator.
    """
