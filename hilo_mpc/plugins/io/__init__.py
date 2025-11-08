"""
I/O plugins for industrial connectivity.

Currently includes an asyncio-based OPC UA client wrapper.
"""

from .opcua_async import AsyncOPCUAClient, IOMapping, build_mapping_from_model
from .opcua_connector import OPCUAConnector
from .opcua_simple_control_loop import OPCUASimpleControlLoop

__all__ = [
    "AsyncOPCUAClient",
    "IOMapping",
    "build_mapping_from_model",
    "OPCUAConnector",
    "OPCUASimpleControlLoop",
]
