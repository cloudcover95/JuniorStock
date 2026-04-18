# juniorstock/cli.py
import argparse
import sys
from juniorstock.engine.async_router import JuniorEngineLoop

def display_banner():
    print("==================================================")
    print("  JUNIORSTOCK SDK - Edge-Native Operations")
    print("  Target: M4/M1 SoC | Thermal Max: 45W")
    print("==================================================")

def run_ui_server(host: str, port: int):
    import uvicorn
    # Execute fastAPI payload
    print(f"[BOOT] Initializing iPad WS Audit Server at {host}:{port}")
    uvicorn.run("juniorstock.ui.dashboard_server:app", host=host, port=port, log_level="warning")

def run_engine_loop(hz: int):
    engine = JuniorEngineLoop(target_hz=hz)
    engine.ignite()

def main():
    parser = argparse.ArgumentParser(description="JuniorStock CLI Orchestrator")
    subparsers = parser.add_subparsers(dest="command")

    # Command: engine
    parser_engine = subparsers.add_parser("engine", help="Ignite async trading event loop")
    parser_engine.add_argument("--hz", type=int, default=100, help="Target loop frequency (Hz)")

    # Command: ui
    parser_ui = subparsers.add_parser("ui", help="Launch WebSocket Audit Dashboard")
    parser_ui.add_argument("--port", type=int, default=8000, help="Target port")

    args = parser.parse_args()
    display_banner()

    if args.command == "engine":
        run_engine_loop(args.hz)
    elif args.command == "ui":
        run_ui_server("0.0.0.0", args.port)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()

# --- Phase 14 Dispatch Override ---
def run_distributed_engine(node_id: str, hz: int):
    from juniorstock.engine.distributed_loop import DistributedJuniorEngine
    engine = DistributedJuniorEngine(node_id=node_id, target_hz=hz)
    engine.ignite()

# To utilize within argparse, append:
# parser_dist = subparsers.add_parser("mesh", help="Ignite distributed AX mesh loop")
# parser_dist.add_argument("--node", type=str, default="M4_PRIMARY")

# --- Phase 16 Dispatch Override ---
def run_zero_alloc_engine(node_id: str, hz: int):
    from juniorstock.engine.zero_alloc_loop import ZeroAllocEngine
    engine = ZeroAllocEngine(node_id=node_id, target_hz=hz)
    try:
        engine.ignite()
    finally:
        engine.shutdown()

# CLI logic append:
# parser_za = subparsers.add_parser("za-loop", help="Ignite Zero-Alloc execution loop")

# --- Phase 17 Dispatch Override ---
def run_deterministic_engine(node_id: str, hz: int):
    from juniorstock.engine.deterministic_loop import DeterministicExecutionEngine
    engine = DeterministicExecutionEngine(node_id=node_id, target_hz=hz)
    try:
        engine.ignite()
    finally:
        engine.shutdown()

# CLI logic append:
# parser_det = subparsers.add_parser("det-loop", help="Ignite Deterministic Mach-Pinned loop")

# --- Phase 18 Dispatch Override ---
def run_cognitive_engine(node_id: str, hz: int):
    from juniorstock.engine.cognitive_loop import CognitiveExecutionEngine
    engine = CognitiveExecutionEngine(node_id=node_id, target_hz=hz)
    try:
        engine.ignite()
    finally:
        engine.shutdown()

# CLI logic append:
# parser_cog = subparsers.add_parser("cog-loop", help="Ignite Cognitive LSH loop")

# --- Phase 19 Dispatch Override ---
def run_dynamic_engine(node_id: str, hz: int):
    from juniorstock.engine.dynamic_loop import DynamicExecutionEngine
    engine = DynamicExecutionEngine(node_id=node_id, target_hz=hz)
    try:
        engine.ignite()
    finally:
        engine.shutdown()

# CLI logic append:
# parser_dyn = subparsers.add_parser("dyn-loop", help="Ignite Dynamic Delta loop")

# --- Phase 20 Dispatch Override ---
def run_thermal_engine(node_id: str, hz: int):
    from juniorstock.engine.thermal_loop import ThermalExecutionEngine
    engine = ThermalExecutionEngine(node_id=node_id, base_target_hz=hz)
    try:
        engine.ignite()
    finally:
        engine.shutdown()

# CLI logic append:
# parser_therm = subparsers.add_parser("therm-loop", help="Ignite SMC Thermal-Governed loop")

# --- Phase 21 Dispatch Override ---
def run_adaptive_engine(node_id: str, hz: int):
    from juniorstock.engine.adaptive_loop import AdaptiveExecutionEngine
    engine = AdaptiveExecutionEngine(node_id=node_id, base_target_hz=hz)
    try:
        engine.ignite()
    finally:
        engine.shutdown()

# CLI logic append:
# parser_adapt = subparsers.add_parser("adapt-loop", help="Ignite Adaptive LFMP loop")

# --- Phase 22 Dispatch Override ---
def run_entropy_engine(node_id: str):
    from juniorstock.engine.entropy_loop import EntropyExecutionEngine
    engine = EntropyExecutionEngine(node_id=node_id)
    try:
        engine.ignite()
    finally:
        engine.shutdown()

# CLI logic append:
# parser_ent = subparsers.add_parser("ent-loop", help="Ignite Entropy-Bounded Execution loop")

# --- Phase 23 Dispatch Override ---
def run_hdam_engine(node_id: str):
    from juniorstock.engine.hdam_loop import HDAMExecutionEngine
    engine = HDAMExecutionEngine(node_id=node_id)
    try:
        engine.ignite()
    finally:
        engine.shutdown()

# CLI logic append:
# parser_hdam = subparsers.add_parser("hdam-loop", help="Ignite HDAM & PHY Bypass loop")

# --- Phase 24 Dispatch Override ---
def run_lut_engine(node_id: str):
    from juniorstock.engine.lut_loop import LUTExecutionEngine
    engine = LUTExecutionEngine(node_id=node_id)
    try:
        engine.ignite()
    finally:
        engine.shutdown()

# CLI logic append:
# parser_lut = subparsers.add_parser("lut-loop", help="Ignite Boolean LUT & PTP Engine")

# --- Phase 25 Dispatch Override ---
def run_interrupt_engine(node_id: str):
    from juniorstock.engine.interrupt_loop import InterruptExecutionEngine
    engine = InterruptExecutionEngine(node_id=node_id)
    try:
        engine.ignite()
    finally:
        engine.shutdown()

# CLI logic append:
# parser_irq = subparsers.add_parser("irq-loop", help="Ignite Hardware Interrupt loop (0W Idle)")

# --- Phase 26 Dispatch Override ---
def run_spiking_engine(node_id: str):
    from juniorstock.engine.spiking_loop import SpikingExecutionEngine
    engine = SpikingExecutionEngine(node_id=node_id)
    try:
        engine.ignite()
    finally:
        engine.shutdown()

# CLI logic append:
# parser_spike = subparsers.add_parser("spike-loop", help="Ignite Spiking Ternary Automata")

# --- Phase 27 Dispatch Override ---
def run_consensus_engine(node_id: str):
    from juniorstock.engine.consensus_loop import ConsensusExecutionEngine
    engine = ConsensusExecutionEngine(node_id=node_id)
    try:
        engine.ignite()
    finally:
        engine.shutdown()

# CLI logic append:
# parser_dag = subparsers.add_parser("dag-loop", help="Ignite DAG Consensus & Enclave Engine")

# --- Phase 28 Dispatch Override ---
@router.post("/api/devops/sync")
async def manual_sync():
    """
    API endpoint for the iPad Dashboard to trigger repository sync.
    """
    from juniorstock.devops.artifact_sync import SovereignRepoSync
    sync = SovereignRepoSync()
    sync.commit_and_push_delta()
    return {"status": "SUCCESS", "detail": "Repository pushed to cloudcover95"}

def run_sovereign_engine(node_id: str):
    from juniorstock.engine.sovereign_loop import SovereignExecutionEngine
    engine = SovereignExecutionEngine(node_id=node_id)
    try:
        engine.ignite()
    finally:
        engine.shutdown()

# CLI logic append:
# parser_sov = subparsers.add_parser("sov-loop", help="Ignite Sovereign T-NAS & Sync Engine")
