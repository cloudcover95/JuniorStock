# path: src/juniorstock/engines/swarm/obsidian_indexer.py
#!/usr/bin/env python3
"""
Feature: Obsidian Vault Indexer

Generates daily summary markdown from debate logs.
"""

import glob
from datetime import datetime
from pathlib import Path


class ObsidianIndexer:
    def __init__(self, vault_path: str = None):
        if vault_path:
            self.vault = Path(vault_path)
        else:
            self.vault = Path.home() / "JuniorCloud" / "juniorstock" / "vault" / "obsidian_logs"

    def generate_daily_summary(self, date_str: str = None) -> str:
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        pattern = str(self.vault / f"debate_*_{date_str}*.md")
        files = glob.glob(pattern)

        summary = f"# Daily Swarm Summary - {date_str}\n\nTotal debates: {len(files)}\n"
        return summary
