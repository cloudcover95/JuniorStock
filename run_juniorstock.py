import time, threading, numpy as np, pandas as pd, yfinance as yf, logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Future imports mapped to the JuniorStock namespace
from src.juniorstock.core.tensor_engine import UnifiedFinancialTensor
# NOTE: Assume telemetry, hunter, and profitability modules are copied into src/juniorstock/

logging.basicConfig(level=logging.INFO, format="[*] %(message)s")
app = FastAPI(title="JuniorStock V5.0 Flagship Node")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = UnifiedFinancialTensor()
ACTIVE_NET = ["BTC-USD", "ETH-USD", "SOL-USD", "SPY", "TSLA"] # Default fallback

@app.get("/")
async def serve_dashboard():
    # Requires the Omni-Flex HTML to be moved to src/juniorstock/dashboard/
    return FileResponse("src/juniorstock/dashboard/app.html")

def pipeline_loop():
    logging.info("V5.0 Flagship Pipeline Engaged. Connecting to KPZ manifold...")
    pulse_idx = 0
    while True:
        try:
            raw_data = yf.download(ACTIVE_NET, period="1d", interval="1m", group_by="ticker", progress=False)
            for t in ACTIVE_NET:
                try:
                    df = raw_data[t].dropna() if isinstance(raw_data.columns, pd.MultiIndex) else raw_data.dropna()
                    if len(df) < 60: continue
                    C, H, L = df['Close'].values, df['High'].values, df['Low'].values
                    metrics = engine.process_manifold(C, H, L)
                    
                    # Compute Delta Q
                    delta_q = engine.compute_delta_q(metrics["q_mark"])
                    
                    # Telemetry injection goes here (jcllc_registry.ingest)
                    logging.info(f"[Pulse {pulse_idx:04d}] {t} | Q: {metrics['q_mark'][0]:.4f} | K-alpha: {metrics['k_alpha'][0]:.4f}")
                except KeyError: continue
                
            pulse_idx += 1
            time.sleep(60)
        except Exception as e:
            logging.error(f"Root Node Failure: {e}")
            time.sleep(40)

if __name__ == "__main__":
    threading.Thread(target=pipeline_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8080)
