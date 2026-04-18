# juniorstock/engine/engine_patch.py
import os
from juniorstock.licensing.drm_verifier import EnterpriseLicenseGate

def assert_license_state():
    """
    Hard gate injected before Async Engine ignition.
    Reads cached license from disk and verifies HWID lock.
    """
    LICENSE_PATH = "juniorstock/licensing/keys/node_license.jwt"
    
    print("[DRM GATE] Verifying enterprise execution rights...")
    if not os.path.exists(LICENSE_PATH):
        raise PermissionError("[DRM FAULT] No enterprise license found. Node execution locked.")
        
    with open(LICENSE_PATH, "r") as f:
        jwt_payload = f.read().strip()
        
    gate = EnterpriseLicenseGate()
    if not gate.verify_license_payload(jwt_payload):
        raise PermissionError("[DRM FAULT] Local license invalid. Node execution locked.")
        
    print("[DRM GATE] Clear. Routing execution to Apple Silicon Neural Engine.")
