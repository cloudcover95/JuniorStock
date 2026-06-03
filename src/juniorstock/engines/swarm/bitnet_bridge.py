# path: src/juniorstock/engines/swarm/bitnet_bridge.py
#!/usr/bin/env python3
"""
V6.2: BitNet-MLX Cognitive Bridge

Sovereign local LLM reasoning node for Apple Silicon.
Translates deterministic tensor consensus into executive Obsidian logs.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import mlx.core as mx
    import mlx_lm
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False

logging.basicConfig(level=logging.INFO, format="[*] %(asctime)s - %(message)s")


class BitNetCognitiveBridge:
    """
    V6.2: Sovereign LLM Reasoning Node for Apple Silicon.
    Uses MLX for local ternary/quantized inference.
    """

    def __init__(self, model_path: str = "mlx-community/Qwen1.5-7B-Chat-4bit", vault_root: Optional[str] = None):
        if vault_root:
            self.vault_path = Path(vault_root)
        else:
            self.vault_path = Path.home() / "JuniorCloud" / "juniorstock" / "vault" / "obsidian_logs"
        self.vault_path.mkdir(parents=True, exist_ok=True)

        self.model_path = model_path
        self.model = None
        self.tokenizer = None

        if HAS_MLX_LM:
            logging.info(f"[+] Loading Sovereign MLX Model: {self.model_path}")
            try:
                self.model, self.tokenizer = mlx_lm.load(self.model_path)
            except Exception as e:
                logging.error(f"[!] MLX Model Load Failure: {e}")
                global HAS_MLX_LM
                HAS_MLX_LM = False

    def generate_debate_log(self, ticker: str, consensus_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        prompt = (
            f"<|im_start|>system\nYou are an elite quantitative analyst for JuniorCloud LLC. "
            f"Review the deterministic swarm consensus and output a strict 3-bullet executive summary "
            f"justifying the action. No fluff.<|im_end|>\n"
            f"<|im_start|>user\nTicker: {ticker}\nAction: {consensus_data.get('action_proposal', 'HOLD')}\n"
            f"Score: {consensus_data.get('consensus_score', 0.5)}\nContext: {context}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        reasoning = "N/A - MLX_LM offline. Defaulting to deterministic log."
        if HAS_MLX_LM and self.model and self.tokenizer:
            try:
                reasoning = mlx_lm.generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=150,
                    verbose=False
                )
            except Exception as e:
                logging.error(f"[!] Inference failure: {e}")

        md_artifact = f"""---
tags: [swarm_debate, junior_agents, {ticker}]
action: {consensus_data.get('action_proposal', 'HOLD')}
k_alpha: {context.get('k_alpha', 0.0):.4f}
date: {time.strftime('%Y-%m-%d %H:%M:%S')}
---

# Swarm Consensus Log: {ticker}

## Executive Reasoning (BitNet-MLX)
{reasoning.strip()}

## Deterministic Matrix
```json
{consensus_data}
```

## Context Tensor
```json
{context}
```
"""

        filepath = self.vault_path / f"debate_{ticker}_{int(time.time())}.md"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_artifact)

        logging.info(f"[+] Obsidian Log Generated: {filepath.name}")
        return reasoning
