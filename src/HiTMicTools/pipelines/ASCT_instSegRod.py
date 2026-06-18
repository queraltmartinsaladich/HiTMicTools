from HiTMicTools.pipelines._instSeg_base import _InstSegBase


class ASCT_instSegRod(_InstSegBase):
    """instSeg pipeline for rod-shaped bacteria.

    Intended species: E. coli, P. aeruginosa, M. tuberculosis,
    M. abscessus, M. chimaera.

    Uses a model bundle trained on rod morphologies
    (model_collection_instSegRod.zip).  All processing logic is
    inherited from _InstSegBase; the model bundle is the only
    differentiator.
    """
