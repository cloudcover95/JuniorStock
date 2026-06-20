# JuniorStock/agents/fable_sovereign_long_horizon_agent.py
# (Updated with debate loop wiring and more cross-repo nodes)

# ... (previous code) ...

    def wire_into_debate_loop(self, debate_context: Dict) -> Dict:
        """Wire into specific TradingAgents / JuniorStock debate loops.
        Acts as 'LongHorizonPortfolioAnalyst' in multi-agent debate.
        """
        allocator_decision = self.allocator.get_full_strategy_output(debate_context.get("market_data"))
        # Cross-repo inference
        try:
            climbs_spatial = read_parquet_with_evolution("./02_Assets/climbs/telemetry/scan_latest.parquet")  # From JuniorClimbs
            home_power = read_parquet_with_evolution("./02_Assets/home/power/telemetry.parquet")  # From JuniorHome
            bitnet_weights = read_parquet_with_evolution("./02_Assets/bitnet/weights_latest.parquet")  # From BitNet-mlx
        except:
            climbs_spatial = home_power = bitnet_weights = None

        # AGI decision with cross-repo
        decision = {
            "portfolio": allocator_decision,
            "cross_repo_insights": {
                "climbs_spatial_available": climbs_spatial is not None,
                "home_power_available": home_power is not None,
                "bitnet_weights_available": bitnet_weights is not None
            },
            "long_horizon_action": self.process_portfolio_task("debate_integration", debate_context.get("market_data"))
        }
        return decision

    # ... (rest of previous code, with expanded cross-repo in process_portfolio_task to include more nodes) ...