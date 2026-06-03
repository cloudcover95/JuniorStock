# path: src/juniorstock/engines/swarm/__init__.py
#!/usr/bin/env python3
"""
JuniorStock Swarm Package (V6.4)

Includes consensus graph, BitNet bridge, and SovereignExecutionBus.
"""

from .consensus_graph import JCSwarmOrchestrator
try:
    from .bitnet_bridge import BitNetCognitiveBridge
except ImportError:
    BitNetCognitiveBridge = None
try:
    from .execution_bus import SovereignExecutionBus
except ImportError:
    SovereignExecutionBus = None

__all__ = ["JCSwarmOrchestrator", "BitNetCognitiveBridge", "SovereignExecutionBus"]
