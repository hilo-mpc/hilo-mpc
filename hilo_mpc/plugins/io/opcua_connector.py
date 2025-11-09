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
from __future__ import annotations

import asyncio
from typing import Callable, Dict, Sequence, Optional, Tuple

from .opcua_async import AsyncOPCUAClient, IOMapping


ComputeFn = Callable[[Dict[str, float]], Dict[str, float]]


class OPCUAConnector:
    """Generic, minimal OPC UA polling loop.

    Reads a set of aliases, calls a user-provided compute function, and writes outputs.
    Provides safe shutdown behavior (writes zeros or user-provided values on exit).

    Parameters
    ----------
    endpoint: str
        OPC UA endpoint URL.
    mapping: IOMapping
        Aliases for read/write plus optional scale/bounds.
    period: float
        Loop period in seconds. The loop uses asyncio.sleep(period) between cycles.
    reconnect_backoff: (float, float)
        Initial and maximum backoff used by the underlying client on reconnect.
    safe_shutdown: dict[str, float] | None
        Values to write once on exit (e.g., zeros for torques). If None, nothing is written.
    """

    def __init__(
        self,
        endpoint: str,
        mapping: IOMapping,
        period: float,
        reconnect_backoff: Tuple[float, float] = (0.5, 5.0),
        safe_shutdown: Optional[Dict[str, float]] = None,
    ) -> None:
        self.endpoint = endpoint
        self.mapping = mapping
        self.period = float(period)
        self.safe_shutdown = dict(safe_shutdown or {})
        self.client = AsyncOPCUAClient(
            endpoint=endpoint,
            mapping=mapping,
            reconnect_backoff=reconnect_backoff,
            max_reconnect_attempts=None,
        )

    @property
    def read_keys(self) -> Sequence[str]:
        """Aliases read each cycle."""
        return list(self.mapping.reads.keys())

    @property
    def write_keys(self) -> Sequence[str]:
        """Aliases written each cycle."""
        return list(self.mapping.writes.keys())

    async def run(self, compute: ComputeFn, max_iters: Optional[int] = None) -> None:
        """Run the polling loop until cancelled or max_iters reached.

        Parameters
        ----------
        compute: Callable[[dict[str, float]], dict[str, float]]
            Function mapping read alias-values to write alias-values (engineering units).
        max_iters: int | None
            If provided, stop after this many cycles. When None, run indefinitely.
        """
        await self.client.connect()
        try:
            k = 0
            while True:
                data = await self.client.read(self.read_keys)
                inputs = {
                    k: float(data[k]["value"]) for k in self.read_keys if k in data and data[k].get("value") is not None
                }
                outputs = compute(inputs)
                await self.client.write(outputs)
                await asyncio.sleep(self.period)
                if max_iters is not None:
                    k += 1
                    if k >= max_iters:
                        break
        finally:
            try:
                if self.safe_shutdown:
                    await self.client.write(self.safe_shutdown)
            except Exception:
                pass
            await self.client.disconnect()
