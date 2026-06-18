from HiTMicTools.pipelines.base_instSeg import BaseInstSeg


class ASCT_instSegRod(BaseInstSeg):
    """instSeg pipeline for rod-shaped bacteria.

    Intended species: E. coli, P. aeruginosa, M. tuberculosis,
    M. abscessus, M. chimaera.

    Uses a model bundle trained on rod morphologies
    (model_collection_instSegRod.zip).  All processing logic is
    inherited from BaseInstSeg; the model bundle is the only
    differentiator.
    """
