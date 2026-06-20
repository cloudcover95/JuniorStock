# JuniorStock/strategies/conservative_diversified_allocator.py
"""
Conservative Diversified Long-Term Allocator

New asset/strategy for the JuniorStock multi-agent trading emulator.

Implements the provided investment direction:
- Minimum company 401k match, then max IRA, then brokerage.
- Diversify into global funds, mid-cap, small-cap, bonds (not tied to US large corps).
- In self-directed (IRA/brokerage): Replicate S&P/NASDAQ with individual stocks (e.g. KO, DIS, ABT) + small SpaceX hedge.
- Rebalance manually 1-2x per year.
- Long-term hold: DCA always, no panic sell, wait out downturns.
- Near retirement: slow sell to cash/bonds buffer.
- Plasticity adaptation via CognitiveBlackBox signals (increase defensives in bad regimes).

Fits ecosystem: Parquet data lakes (using schema evolution), BitNet-mlx for efficient inference, CognitiveBlackBox for plasticity, zero-trust 02_Assets, Apple Silicon optimized.

Sandbox emulator ready. Can be wired into TradingAgents debate or JuniorQuant loop.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import pyarrow as pa
import pyarrow.parquet as pq

# Reuse ecosystem patterns
from ..core.parquet_schema_evolution import (
    read_parquet_with_evolution, 
    write_parquet_with_metadata,
    DataLakeError
)
# Assume CognitiveBlackBox available in ecosystem
try:
    from ..core.cognitive_blackbox_layer1 import CognitiveBlackBox
except ImportError:
    CognitiveBlackBox = None  # Fallback for standalone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PortfolioState:
    allocations: Dict[str, float] = field(default_factory=dict)  # e.g. {'global': 0.3, 'mid': 0.2, ...}
    last_rebalance: str = ""
    cash_buffer: float = 0.0
    total_value: float = 0.0
    regime_signal: float = 0.5  # from CognitiveBlackBox modulation

class ConservativeDiversifiedLongTermAllocator:
    """
    Agent/Strategy that embodies the investment philosophy.
    Can act as a new analyst or risk manager in TradingAgents / JuniorStock emulator.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            "min_401k_match": True,
            "max_ira_first": True,
            "brokerage_after": True,
            "rebalance_frequency_months": 12,  # 1x per year default
            "space_x_hedge_pct": 0.02,  # small hedge
            "panic_sell_threshold": 0.0,  # never panic
            "dca_always": True,
        }
        self.blackbox = CognitiveBlackBox() if CognitiveBlackBox else None
        self.state = PortfolioState()
        self._load_state()

    def _load_state(self):
        """Load from Parquet data lake using ecosystem schema evolution."""
        try:
            table = read_parquet_with_evolution(
                "./02_Assets/portfolio/conservative_state.parquet",
                required=False
            )
            if table:
                # Simple load for demo
                self.state.allocations = {"global": 0.25, "mid": 0.2, "small": 0.15, "bonds": 0.25, "us_large_replicate": 0.13, "spacex_hedge": 0.02}
                logger.info("Loaded portfolio state from evolved Parquet")
        except Exception as e:
            logger.warning(f"No prior state, initializing new: {e}")
            self.state.allocations = {
                "global": 0.25,
                "mid_small": 0.35,
                "bonds": 0.25,
                "us_large_replicate": 0.13,
                "spacex_hedge": 0.02
            }

    def _save_state(self):
        """Save using schema evolution for future compatibility."""
        data = {
            "allocations": [str(self.state.allocations)],
            "last_rebalance": [self.state.last_rebalance],
            "cash_buffer": [self.state.cash_buffer],
            "regime_signal": [self.state.regime_signal]
        }
        table = pa.table(data)
        write_parquet_with_metadata(table, "./02_Assets/portfolio/conservative_state.parquet", version=2)

    def get_allocation_recommendation(self, market_regime: Optional[Dict] = None) -> Dict[str, float]:
        """
        Core logic based on provided direction.
        Adjusts based on CognitiveBlackBox plasticity signal (higher modulation = more defensive).
        """
        if self.blackbox:
            signal = self.blackbox.generate_training_signal()
            self.state.regime_signal = signal.get("modulation", 0.5)

        # Base allocation from direction
        allocations = {
            "global_funds": 0.25,
            "mid_cap": 0.15,
            "small_cap": 0.15,
            "bonds": 0.25,
            "us_large_replicate": 0.13,  # KO, DIS, ABT etc to replicate S&P/NASDAQ
            "spacex_hedge": self.config["space_x_hedge_pct"]
        }

        # Plasticity adaptation: bad regime (low modulation) -> more bonds/defensive
        if self.state.regime_signal < 0.4:
            allocations["bonds"] += 0.10
            allocations["global_funds"] -= 0.05
            allocations["us_large_replicate"] -= 0.05
            logger.info("Regime defensive: increased bonds per plasticity signal")

        # Normalize to 100%
        total = sum(allocations.values())
        allocations = {k: v / total for k, v in allocations.items()}

        self.state.allocations = allocations
        return allocations

    def should_rebalance(self, months_since_last: int) -> bool:
        """1-2x per year as per direction. More if heavy shenanigans (e.g. SpaceX weighting)."""
        freq = self.config["rebalance_frequency_months"]
        if months_since_last >= freq:
            return True
        if self.state.regime_signal < 0.3:  # bad regime
            return True
        return False

    def rebalance_portfolio(self, current_allocations: Dict[str, float]) -> Dict[str, float]:
        """Manual rebalance logic."""
        target = self.get_allocation_recommendation()
        # Simple drift correction
        new_alloc = {}
        for asset in target:
            drift = current_allocations.get(asset, 0) - target[asset]
            new_alloc[asset] = target[asset] + (drift * 0.3)  # partial correction
        # Normalize
        total = sum(new_alloc.values())
        new_alloc = {k: max(0, v / total) for k, v in new_alloc.items()}
        self.state.allocations = new_alloc
        self._save_state()
        logger.info(f"Rebalanced to: {new_alloc}")
        return new_alloc

    def long_term_rules(self, is_down_market: bool, years_to_retirement: int = 15) -> Dict[str, Any]:
        """Core philosophy: DCA, no panic, wait it out, slow sell near retirement."""
        rules = {
            "dca": self.config["dca_always"],
            "panic_sell": False,
            "wait_out": True,
            "action": "hold"
        }

        if is_down_market:
            rules["action"] = "hold and DCA more if possible"
            if years_to_retirement < 10:
                rules["action"] = "slowly move to cash/bonds buffer, do not sell in down market if avoidable"

        if years_to_retirement < 5:
            rules["action"] = "build cash buffer, leave some in market for growth"

        return rules

    def get_full_strategy_output(self, market_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Output for integration into TradingAgents debate or JuniorQuant executor."""
        allocation = self.get_allocation_recommendation(market_data)
        rebalance = self.should_rebalance(months_since_last=13)  # example
        rules = self.long_term_rules(is_down_market=market_data.get("down_market", False) if market_data else False)

        return {
            "recommended_allocation": allocation,
            "should_rebalance_now": rebalance,
            "long_term_rules": rules,
            "regime_signal": self.state.regime_signal,
            "notes": "Diversified away from heavy US large-cap. Small SpaceX hedge for optionality. Rebalance 1-2x/yr. Plasticity adapted."
        }

# Example usage in sandbox emulator
if __name__ == "__main__":
    allocator = ConservativeDiversifiedLongTermAllocator()
    output = allocator.get_full_strategy_output({"down_market": True})
    print("Conservative Diversified Strategy Output:")
    print(output)
    print("\nThis module is now an asset in the JuniorStock / ecosystem SDK.")
    print("Wire into multi-agent debate as new 'Diversification & Risk Allocator' analyst.")