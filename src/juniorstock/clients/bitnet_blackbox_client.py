# path: src/juniorstock/clients/bitnet_blackbox_client.py
import requests
from typing import List, Dict, Any


class BitNetBlackboxClient:
    """
    Client for the isolated BitNet-mlx Proprietary Blackbox.
    Communicates over localhost:8001 (air-gapped).
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.base_url = base_url
        self.available = self._check_health()

    def _check_health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/docs", timeout=1.0)
            return r.status_code == 200
        except Exception:
            return False

    def infer(self, ticker: str, spatial_tensor: List[List[float]]) -> Dict[str, Any]:
        if not self.available:
            return {
                "status": "OFFLINE",
                "math_state": {},
                "bitnet_consensus": "Blackbox offline - using local fallback"
            }

        try:
            payload = {"ticker": ticker, "spatial_tensor": spatial_tensor}
            r = requests.post(f"{self.base_url}/v1/proprietary/infer", json=payload, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "math_state": {},
                "bitnet_consensus": "Request failed"
            }
