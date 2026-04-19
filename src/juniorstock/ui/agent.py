import mlx.core as mx
from mlx_lm import load, generate
import logging

class QuantTerraformingAgent:
    def __init__(self, model_path="mlx-community/Meta-Llama-3-8B-Instruct-4bit"):
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        try:
            self.model, self.tokenizer = load(self.model_path)
        except Exception as e:
            logging.error(f"Quant Agent Offline: {e}")

    def generate_quant_board(self, request: str) -> str:
        if not self.model: return "<div class='text-red-500'>AGENT OFFLINE</div>"
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are the JuniorStock Web3 UI Harvester. Output ONLY raw HTML/Tailwind/JS.
Design for a dark-theme crypto/stocks dashboard (TradingView style).
Use monospace fonts for ticker data.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Build: {request}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        response = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=1024, verbose=False)
        return response.replace("```html", "").replace("```", "").strip()
