# JuniorStock/second_brain/performance_tracker.py
# Performance measurement fed back to the main second brain.

from .sovereign_blackbox_brain import PerformanceMeasurer, SovereignBlackboxBrain

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTracker:
    def __init__(self, brain: SovereignBlackboxBrain = None):
        self.brain = brain or SovereignBlackboxBrain()
        self.measurer = PerformanceMeasurer()

    def track_and_adapt(self, task: str, result: Dict):
        latency = result.get("latency", 100)
        accuracy = result.get("accuracy", 0.95)
        metrics = self.measurer.measure(task, latency, accuracy)
        if hasattr(self.brain, 'performance_log'):
            self.brain.performance_log.append(metrics)
        logger.info(f"Tracked and adapted for {task}")
        return metrics