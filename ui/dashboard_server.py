# juniorstock/ui/dashboard_server.py
import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="JuniorStock Edge UI")

# Serve the TradingView-style dark theme template
@app.get("/")
async def get_dashboard():
    with open("juniorstock/ui/templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.websocket("/ws/audit")
async def audit_endpoint(websocket: WebSocket):
    """
    WebSocket feed tailored for M1 iPad Pro rendering.
    Pushes continuous Topo-SVD state updates and hardware macro triggers.
    """
    await websocket.accept()
    try:
        while True:
            # Dummy telemetry mirroring Omni Math Kernel states
            payload = {
                "system": "Apple M4",
                "power_draw": "38W",
                "tda_betti_state": [1, 0, 1, -1, 0],
                "active_bots": 1,
                "macro_automata_status": "LOCKED"
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.5)  # 2Hz telemetry limit for local network
    except Exception as e:
        print(f"[WS ERROR] Connection broken: {e}")

# --- Phase 6/7 Router Injections ---
try:
    from juniorstock.ui.audit_routes import router as audit_router
    from juniorstock.ui.metals_router import router as metals_router
    app.include_router(audit_router)
    app.include_router(metals_router)
except ImportError as e:
    print(f"[ROUTER FAULT] Failed to load expansion routes. ERR: {e}")

# --- Phase 11 Router Injections ---
try:
    from juniorstock.licensing.api_routes import router as licensing_router
    app.include_router(licensing_router)
except ImportError as e:
    print(f"[ROUTER FAULT] Failed to load licensing routes. ERR: {e}")
