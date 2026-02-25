"""
Physics-Inspired Event Discovery in Long-Horizon Video.

A framework for discovering rare, important events in long video streams
using physics-inspired optimization techniques.

Quick start::

    from event_discovery.methods import HierarchicalEnergyMethod, EnergyConfig

    config = EnergyConfig(top_k=10)
    method = HierarchicalEnergyMethod(config)
    events = method.process_video("path/to/video.mp4")

    for event in events:
        print(f"{event.start_time:.1f}s - {event.end_time:.1f}s")
"""

__version__ = "0.1.0"
__author__ = "Meshal Alshammari"
__email__ = "meshal@berkeley.edu"
__license__ = "MIT"

__all__ = ["__version__", "__author__", "__email__", "__license__"]
