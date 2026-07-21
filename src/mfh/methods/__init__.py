"""Activation-steering method implementations."""

from mfh.methods.adaptive import AdaptiveController, AlphaController, RoutedVectorBank
from mfh.methods.composite import CompositePolicy, CompositePolicyConfig
from mfh.methods.extraction import CAAExtractor, CentroidExtractionMode, CentroidExtractor
from mfh.methods.probes import CalibratedProbe, ProbeTask
from mfh.methods.protected import ProtectedSubspace
from mfh.methods.sparse import SAEConfig, SparseAutoencoder
from mfh.methods.static import (
    CentroidVectorBuilder,
    OnlineMoments,
    PairedDifferenceBuilder,
    SteeringVector,
    VectorBank,
)

__all__ = [
    "AdaptiveController",
    "AlphaController",
    "CAAExtractor",
    "CalibratedProbe",
    "CentroidExtractionMode",
    "CentroidExtractor",
    "CentroidVectorBuilder",
    "CompositePolicy",
    "CompositePolicyConfig",
    "OnlineMoments",
    "PairedDifferenceBuilder",
    "ProbeTask",
    "ProtectedSubspace",
    "RoutedVectorBank",
    "SAEConfig",
    "SparseAutoencoder",
    "SteeringVector",
    "VectorBank",
]
