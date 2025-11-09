#
#   This file is part of HILO-MPC
#
#   HILO-MPC is a toolbox for easy, flexible and fast development of machine-learning-supported
#   optimal control and estimation problems
#
#   Copyright (c) 2025 Johannes Pohlodek, Bruno Morabito, Rolf Findeisen
#                      All rights reserved
#
#   HILO-MPC is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   HILO-MPC is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with HILO-MPC. If not, see <http://www.gnu.org/licenses/>.
#

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
