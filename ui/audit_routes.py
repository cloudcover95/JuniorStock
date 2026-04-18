# juniorstock/ui/audit_routes.py
from fastapi import APIRouter, HTTPException
import pyarrow.parquet as pq
from juniorstock.security.path_isolation import EnvironmentIsolationGate

router = APIRouter()

@router.get("/api/ledger/history")
async def get_execution_history(limit: int = 100):
    """
    REST endpoint to query the Parquet execution ledger.
    Tailored for iPad dashboard integration.
    """
    ledger_path = "juniorstock/ledger/audit_trail.parquet"
    
    try:
        EnvironmentIsolationGate.verify_io_path(ledger_path)
        dataset = pq.ParquetDataset(ledger_path)
        table = dataset.read()
        df = table.to_pandas()
        
        # Sort by timestamp descending and apply limit
        df_sorted = df.sort_values(by="timestamp", ascending=False).head(limit)
        return df_sorted.to_dict(orient="records")
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Ledger not initialized or path unavailable.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ledger parse failure: {e}")
