# juniorstock/ui/metals_router.py
from fastapi import APIRouter
from pydantic import BaseModel
import mlx.core as mx
from juniorstock.math.fsd.kinematics import FSDKinematicsKernel
from juniorstock.metals.sentiment_node import MetalsSentimentNode

router = APIRouter()
fsd_kernel = FSDKinematicsKernel()

class MetalsTick(BaseModel):
    price: float
    volume: float
    dxy_correlation: float
    sentiment_index: float

# Initialize single node for in-memory pipeline routing
xau_node = MetalsSentimentNode(asset_id="XAU_USD")

@router.post("/api/metals/ingest")
async def ingest_metals_data(tick: MetalsTick):
    """
    Ingests metals stream. Tailored for the iPad M1 Dashboard.
    Passes data through FSD Kinematics to calculate safe execution states.
    """
    xau_node.ingest_metals_tick(tick.price, tick.volume, tick.dxy_correlation, tick.sentiment_index)
    
    # Extract and evaluate directly (assuming batch size of 1 for high-frequency stream)
    tensor_state = xau_node.extract_tensor()
    
    # Check execution boundary
    fsd_safe = fsd_kernel.evaluate_kinetic_boundary(tensor_state)
    
    return {
        "status": "INGESTED",
        "asset": xau_node.asset_id,
        "fsd_autonomous_safe": bool(fsd_safe)
    }
