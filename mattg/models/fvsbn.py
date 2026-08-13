import torch

from mattg.models.factory import ModelFactory
from mattg.models.masked_linear import MaskedLinear


class FVSBN(MaskedLinear):
    """Fully Visible Sigmoid Belief Network"""
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, mask=FVSBN.build_mask(in_features, out_features))

    @staticmethod
    def build_mask(in_features, out_features) -> torch.Tensor:
        mask = torch.ones(out_features, in_features)
        mask = torch.tril(mask, diagonal=-1)
        return mask


class FVSBNFactory(ModelFactory):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def _create(self):
        return FVSBN(self.in_features, self.out_features), []

    def _metadata(self):
        return {
            "name": "FVSBN",
            "description": "Fully Visible Sigmoid Belief Network",
        }