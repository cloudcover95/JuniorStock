import json
import logging
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from juniorstock.ui.agent import QuantTerraformingAgent

app = FastAPI(title="JuniorStock Web3 Node")
app.mount("/public", StaticFiles(directory="src/juniorstock/ui/public"), name="public")
terraformer = QuantTerraformingAgent()

@app.get("/")
async def serve_canvas():
    return FileResponse("src/juniorstock/ui/public/index.html")

@app.websocket("/ws/quant")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            if payload.get("action") == "terraform":
                await websocket.send_json({"type": "status", "message": "Compiling Quant Component..."})
                html = terraformer.generate_quant_board(payload["request"])
                await websocket.send_json({"type": "board", "html": html})
    except WebSocketDisconnect:
        pass

def start_node():
    uvicorn.run("juniorstock.ui.server:app", host="127.0.0.1", port=8080, reload=True)

if __name__ == "__main__":
    start_node()
