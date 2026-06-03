# path: src/juniorstock/engines/swarm/__init__.py
#!/usr/bin/env python3
"""
JuniorStock Swarm Package (V6.2)

Includes BitNet-MLX bridge, daemon, and supporting production features.
"""

from .consensus_graph import JCSwarmOrchestrator
try:
    from .bitnet_bridge import BitNetCognitiveBridge
except ImportError:
    BitNetCognitiveBridge = None

__all__ = ["JCSwarmOrchestrator", "BitNetCognitiveBridge"]
