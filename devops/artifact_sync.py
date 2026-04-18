# juniorstock/devops/artifact_sync.py
import os
import subprocess
import time
from juniorstock.execution.phy.phy_injector import PHYNetworkInjector

class SovereignRepoSync:
    """
    Manages repository state. Attempts PHY hardware injection for off-grid 
    commits, but falls back to standard Darwin TCP/IP git push if the 
    ATmega32u4 hardware bridge is not detected.
    """
    def __init__(self, repo_url: str = "https://github.com/cloudcover95/JuniorStock.git"):
        self.repo_url = repo_url
        self.phy = PHYNetworkInjector()

    def ensure_git_initialized(self):
        if not os.path.exists(".git"):
            print("[SYNC GATE] Initializing missing Git tree...")
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "remote", "add", "origin", self.repo_url], check=False)
            subprocess.run(["git", "branch", "-M", "main"], check=True)

    def commit_and_push_delta(self):
        self.ensure_git_initialized()
        print(f"[SYNC GATE] Preparing delta for {self.repo_url}...")
        
        # Standard Git local staging
        try:
            subprocess.run(["git", "add", "."], check=True)
            # Check if there's anything to commit
            status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
            if status.stdout.strip():
                subprocess.run(["git", "commit", "-m", f"Iteration: {time.strftime('%Y%m%d-%H%M%S')} - Sovereign Scaffold"], check=True)
            else:
                print("[SYNC GATE] No changes to commit. Tree clean.")
        except subprocess.CalledProcessError as e:
            print(f"[SYNC FAULT] Local git staging failed: {e}")
            return

        # Dual-Path Push Routing
        if self.phy.fd is not None:
            print("[SYNC GATE] PHY Hardware active. Dispatching via UART...")
            self.phy.inject_raw_frame("COMMIT_PUSH_SIGNAL_CLOUDCOVER95")
            print("[SYNC SUCCESS] Iteration committed to sovereign upstream (PHY).")
        else:
            print("[SYNC GATE] PHY Hardware offline. Falling back to Darwin TCP Git Push...")
            try:
                subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
                print("[SYNC SUCCESS] Iteration pushed to cloudcover95 (TCP).")
            except subprocess.CalledProcessError as e:
                print(f"[SYNC FAULT] TCP Push failed. Verify SSH keys or GitHub credentials. {e}")
