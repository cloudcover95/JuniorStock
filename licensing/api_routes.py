# juniorstock/licensing/api_routes.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import os
from juniorstock.licensing.drm_verifier import EnterpriseLicenseGate
from juniorstock.licensing.hwid_lock import HardwareFingerprint

router = APIRouter()
drm_gate = EnterpriseLicenseGate()
LICENSE_PATH = "juniorstock/licensing/keys/node_license.jwt"

class LicensePayload(BaseModel):
    license_key: str

@router.get("/api/licensing/node_fingerprint")
async def get_node_fingerprint():
    """
    Returns the SHA-256 SoC hash.
    Enterprise clients send this hash to JuniorCloud LLC to generate their specific license key.
    """
    hwid = HardwareFingerprint.generate_soc_hash()
    if not hwid:
        raise HTTPException(status_code=500, detail="Hardware extraction failed. Ensure Apple Silicon target.")
    return {"status": "SUCCESS", "hwid": hwid}

@router.post("/api/licensing/apply")
async def apply_enterprise_license(payload: LicensePayload):
    """
    Receives the signed Ed25519 JWT and attempts local offline verification.
    If valid, persists to disk for immediate boot routing.
    """
    is_valid = drm_gate.verify_license_payload(payload.license_key)
    
    if not is_valid:
        raise HTTPException(status_code=403, detail="Cryptographic verification failed or HWID mismatch.")
        
    try:
        with open(LICENSE_PATH, "w") as f:
            f.write(payload.license_key)
        return {"status": "SUCCESS", "detail": "License locked to node. Engine ignition authorized."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disk I/O fault writing license: {e}")
