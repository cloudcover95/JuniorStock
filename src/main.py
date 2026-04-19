# src/main.py

import mlx.core as mx
import logging
from physiomanifold.tda_svd_core.koopman_market import KoopmanOperatorDMD
from physiomanifold.recursive_feedback.active_inference import FreeEnergyMinimizer
from physiomanifold.agi_primitives.hardware_execution import ATmegaHardwareLock

logging.basicConfig(level=logging.INFO, format='%(asctime)s - JUNIORSTOCK_HFFM - %(message)s')

def run_hffm_pipeline():
    logging.info("Bootstrapping JuniorStock Edge Node...")
    
    # 1. Initialize Hardware Lock
    hw_lock = ATmegaHardwareLock()
    hw_lock.engage_lock()
    
    # 2. Initialize Edge Mathematics
    koopman_dmd = KoopmanOperatorDMD(target_rank=16)
    fep_minimizer = FreeEnergyMinimizer(learning_rate=0.05)
    
    # Simulate streaming tick data (T and T+1 matrices)
    X_current = mx.random.normal((500, 200), dtype=mx.float32)
    X_next = X_current + mx.random.normal((500, 200), dtype=mx.float32) * 0.02
    
    logging.info("Extracting market regimes via Koopman DMD...")
    A_tilde, _ = koopman_dmd.extract_dynamics(X_current, X_next)
    
    # 3. Active Inference (Evaluating the Trade Manifold)
    internal_state = mx.random.normal((16, 16), dtype=mx.float32)
    generative_weights = mx.random.normal((16, 16), dtype=mx.float32)
    
    # Route the reduced Koopman dynamics into the Free Energy loop
    logging.info("Routing dynamics into Active Inference loop...")
    updated_state, free_energy = fep_minimizer.execute_perception_step(
        internal_state, 
        A_tilde, # Using the transition matrix as the sensory input
        generative_weights
    )
    
    # 4. Deterministic Execution
    logging.info(f"Variational Free Energy: {free_energy.item():.6f}")
    hw_lock.execute_trade_signal(A_tilde, free_energy.item())

if __name__ == "__main__":
    run_hffm_pipeline()
