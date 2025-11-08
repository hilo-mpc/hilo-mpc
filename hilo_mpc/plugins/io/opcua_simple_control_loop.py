from __future__ import annotations
"""OPC UA control loop (controller+estimator) interface similar to SimpleControlLoop.

Instead of simulating a local plant model, this loop reads the current state (and
optionally parameters/measurements) from an OPC UA server, invokes a controller (optionally an estimator)
and writes the computed control inputs back to the server.

Design goals
------------
- Mirror the convenience of SimpleControlLoop: automatic controller setup, flexible
  support for MPC/OCP, PID, learning-based and generic controllers.
- Keep plant-agnostic: we do not advance dynamics; we only exchange data with the server.
- Allow an optional estimator object whose update/estimate/predict method is called
  after each control action (e.g. for filtering noisy measurements).
- Provide safe shutdown: write zeros (or user-specified values) to all control outputs
  when the loop exits.

Minimal example
---------------

    mapping = IOMapping(
        reads={
            "theta": {"node": f"ns={ns};s=Pendulum/Angle_rad"},
        },
        writes={
            "u": {"node": f"ns={ns};s=Pendulum/Torque_Nm"},
        },
    )
    loop = OPCUASimpleControlLoop(
        endpoint=endpoint,
        mapping=mapping,
        controller=nmpc,
        state_aliases=["theta"],
        control_aliases=["u"],
        period=0.02,
    )
    await loop.run(max_iters=1000)

Controller interface detection (similar to SimpleControlLoop):
- MPC / OCP: must have .optimize(x, cp=None, **kwargs); we pass the current state subset and optional params.
- ANN / learning-based: must have .predict(x) returning control vector.
- PID controllers: must have .call(pv=x_current).
- Generic controller: fallback to .call(x=<state>, p=<params>)

Estimator (optional):
- If has .estimate(), call it after writing control.
- Else if has .predict(), call it.

Notes
-----
- The ordering of state_aliases defines the vector passed to the controller.
- If parameter_aliases are provided, their ordering defines cp for optimize().
- Safe shutdown values can be overridden by passing a dict mapping control_alias to value.
"""

import asyncio
from typing import Any, Dict, List, Optional, Sequence

from .opcua_async import AsyncOPCUAClient, IOMapping


class OPCUASimpleControlLoop:
    """Generic OPC UA feedback loop using an external controller and optional estimator."""

    def __init__(
        self,
        endpoint: str,
        mapping: IOMapping,
        controller: Any,
        estimator: Optional[Any] = None,
        state_aliases: Sequence[str] | None = None,
        control_aliases: Sequence[str] | None = None,
        parameter_aliases: Sequence[str] | None = None,
        period: float = 0.05,
        reconnect_backoff: tuple[float, float] = (0.5, 5.0),
        safe_shutdown: Optional[Dict[str, float]] = None,
    ) -> None:
        self.endpoint = endpoint
        self.mapping = mapping
        self.controller = controller
        self.estimator = estimator
        self.period = float(period)
        self.state_aliases = list(state_aliases or mapping.reads.keys())
        self.control_aliases = list(control_aliases or mapping.writes.keys())
        self.parameter_aliases = list(parameter_aliases or [])
        self.client = AsyncOPCUAClient(endpoint=endpoint, mapping=mapping, reconnect_backoff=reconnect_backoff)
        self.safe_shutdown = (
            dict(safe_shutdown)
            if safe_shutdown is not None
            else {alias: 0.0 for alias in self.control_aliases}
        )

        # Feature flags (similar to SimpleControlLoop)
        self._is_mpc = getattr(controller, "type", None) in {"NMPC", "LMPC"}
        self._is_ocp = getattr(controller, "type", None) == "OCP"
        self._is_pid = getattr(controller, "type", None) == "PID"
        self._has_optimize = hasattr(controller, "optimize") and (self._is_mpc or self._is_ocp)
        self._has_predict = hasattr(controller, "predict")
        self._has_call = hasattr(controller, "call")

        self._est_has_estimate = hasattr(estimator, "estimate") if estimator else False
        self._est_has_predict = hasattr(estimator, "predict") if estimator else False

        # Ensure setup() executed if available
        if hasattr(controller, "is_setup") and not controller.is_setup():
            controller.setup()
        if estimator and hasattr(estimator, "is_setup") and not estimator.is_setup():
            estimator.setup()

    async def _read_values(self) -> Dict[str, float]:
        data = await self.client.read(self.state_aliases + self.parameter_aliases)
        out: Dict[str, float] = {}
        for k, v in data.items():
            try:
                out[k] = float(v["value"])
            except Exception:
                # Non-numeric -> skip
                pass
        return out

    def _build_vectors(self, values: Dict[str, float]) -> tuple[List[float], Optional[List[float]]]:
        x_vec = [values.get(a, 0.0) for a in self.state_aliases]
        cp_vec = [values.get(a, 0.0) for a in self.parameter_aliases] if self.parameter_aliases else None
        return x_vec, cp_vec

    def _compute_control(self, x_vec: List[float], cp_vec: Optional[List[float]]) -> List[float]:
        def _tolist(u: Any) -> List[float]:
            """Assume controller returns CasADi DM/MX; convert via toarray() and flatten."""
            arr = u.toarray()  # Expect DM/MX; if not, this will raise and reveal misuse early.
            return [float(v) for v in arr.flatten()]

        if self._has_optimize:
            return _tolist(self.controller.optimize(x_vec, cp=cp_vec))
        if self._has_predict:
            return _tolist(self.controller.predict(x_vec))
        if self._is_pid and self._has_call:
            return _tolist(self.controller.call(pv=x_vec))
        if self._has_call:
            return _tolist(self.controller.call(x=x_vec, p=cp_vec))
        raise RuntimeError("Controller object does not expose optimize/predict/call interface")

    async def _estimator_step(self) -> None:
        if not self.estimator:
            return
        try:
            if self._est_has_estimate:
                self.estimator.estimate()
            elif self._est_has_predict:
                self.estimator.predict()
        except Exception:
            # Non-critical; ignore estimator errors for robustness
            pass

    async def run(self, max_iters: Optional[int] = None) -> None:
        await self.client.connect()
        try:
            k = 0
            while True:
                values = await self._read_values()
                x_vec, cp_vec = self._build_vectors(values)
                u_vec = self._compute_control(x_vec, cp_vec)
                out: Dict[str, float] = {}
                for i, alias in enumerate(self.control_aliases):
                    out[alias] = float(u_vec[i]) if i < len(u_vec) else 0.0
                await self.client.write(out)
                await self._estimator_step()
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

    def run_sync(self, max_iters: Optional[int] = None) -> None:
        """Synchronous convenience wrapper around the async run method."""
        asyncio.run(self.run(max_iters=max_iters))
