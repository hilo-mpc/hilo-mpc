"""
I/O plugins for industrial connectivity.

Currently includes an asyncio-based OPC UA client wrapper.
"""

from .opcua_async import AsyncOPCUAClient, IOMapping, build_mapping_from_model
from .opcua_loop import OPCUALoop

__all__ = [
    "AsyncOPCUAClient",
    "IOMapping",
    "build_mapping_from_model",
    "OPCUALoop",
]
