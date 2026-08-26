"""Federation with other internal LLM/AI systems."""

from .connectors import PeerAnswer, PeerConfig, PeerError, build_connector
from .orchestrator import FederationOrchestrator, get_orchestrator
from .registry import PeerRegistry, get_registry, load_yaml_peers

__all__ = [
    "FederationOrchestrator",
    "PeerAnswer",
    "PeerConfig",
    "PeerError",
    "PeerRegistry",
    "build_connector",
    "get_orchestrator",
    "get_registry",
    "load_yaml_peers",
]
