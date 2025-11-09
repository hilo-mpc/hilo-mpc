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
from typing import Any, Dict, List, Optional, Sequence

from .opcua_async import AsyncOPCUAClient, IOMapping


class OPCUASimpleControlLoop:
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
        # AsyncOPCUAClient connection/security options (passthrough)
        timeout: float = 2.0,
        security_mode: Optional[str] = None,
        security_policy: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        cert_path: Optional[str] = None,
        key_path: Optional[str] = None,
        max_reconnect_attempts: Optional[int] = None,
    ) -> None:
        """Initialize the OPC UA control loop.

        Parameters
        ----------
        endpoint : str
            OPC UA server endpoint URL, e.g., "opc.tcp://127.0.0.1:4840/server/".
        mapping : IOMapping
            Mapping between aliases and OPC UA node IDs for reads and writes.
        controller : Any
            Controller object (NMPC, LMPC, OCP, PID, or machine learning model) with optimize/predict/call interface.
        estimator : Optional[Any], (MHE, KF, UKF or machine learning model) default=None
            Optional estimator object with estimate() or predict() methods.
        state_aliases : Sequence[str] | None, default=None
            List of state variable aliases to read and pass to controller. If None, uses all read aliases.
        control_aliases : Sequence[str] | None, default=None
            List of control variable aliases to write. If None, uses all write aliases.
        parameter_aliases : Sequence[str] | None, default=None
            List of parameter aliases to read and pass to controller as cp argument.
        period : float, default=0.05
            Control loop sampling period in seconds.
        reconnect_backoff : tuple[float, float], default=(0.5, 5.0)
            Exponential backoff bounds (min, max) in seconds for reconnection attempts.
        safe_shutdown : Optional[Dict[str, float]], default=None
            Dictionary mapping control aliases to safe shutdown values. If None, all controls set to 0.0.
        timeout : float, default=2.0
            Timeout in seconds for OPC UA read/write operations.
        security_mode : Optional[str], default=None
            OPC UA security mode: "None", "Sign", or "SignAndEncrypt".
        security_policy : Optional[str], default=None
            OPC UA security policy: "Basic256Sha256", "Aes128_Sha256_RsaOaep", etc.
        username : Optional[str], default=None
            Username for OPC UA authentication.
        password : Optional[str], default=None
            Password for OPC UA authentication.
        cert_path : Optional[str], default=None
            Path to client certificate file (DER or PEM format).
        key_path : Optional[str], default=None
            Path to client private key file (PEM format).
        max_reconnect_attempts : Optional[int], default=None
            Maximum number of consecutive reconnection attempts. If None, retry indefinitely.
        """
        self.endpoint = endpoint
        self.mapping = mapping
        self.controller = controller
        self.estimator = estimator
        self.period = float(period)
        self.state_aliases = list(state_aliases or mapping.reads.keys())
        self.control_aliases = list(control_aliases or mapping.writes.keys())
        self.parameter_aliases = list(parameter_aliases or [])
        self.client = AsyncOPCUAClient(
            endpoint=endpoint,
            mapping=mapping,
            timeout=timeout,
            security_mode=security_mode,
            security_policy=security_policy,
            username=username,
            password=password,
            cert_path=cert_path,
            key_path=key_path,
            reconnect_backoff=reconnect_backoff,
            max_reconnect_attempts=max_reconnect_attempts,
        )
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
        """Read state and parameter values from the OPC UA server.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping aliases to their float values. Non-numeric values are skipped.
        """
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
        """Build state and parameter vectors from raw values dictionary.

        Parameters
        ----------
        values : Dict[str, float]
            Dictionary of alias-value pairs read from the server.

        Returns
        -------
        tuple[List[float], Optional[List[float]]]
            Tuple containing:
            - x_vec: State vector ordered according to state_aliases
            - cp_vec: Parameter vector ordered according to parameter_aliases, or None if no parameters
        """
        x_vec = [values.get(a, 0.0) for a in self.state_aliases]
        cp_vec = [values.get(a, 0.0) for a in self.parameter_aliases] if self.parameter_aliases else None
        return x_vec, cp_vec

    def _compute_control(self, x_vec: List[float], cp_vec: Optional[List[float]]) -> List[float]:
        """Compute control action using the controller.

        Parameters
        ----------
        x_vec : List[float]
            Current state vector.
        cp_vec : Optional[List[float]]
            Optional parameter vector for the controller.

        Returns
        -------
        List[float]
            Computed control action vector.

        Raises
        ------
        RuntimeError
            If the controller does not expose a recognized interface (optimize/predict/call).
        """
        def _tolist(u: Any) -> List[float]:
            """Assume controller returns CasADi DM/MX; convert via toarray() and flatten.
            
            Parameters
            ----------
            u : Any
                Controller output (expected to be CasADi DM/MX or array-like).
            
            Returns
            -------
            List[float]
                Flattened list of control values.
            """
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
        """Execute one estimator step if an estimator is configured.

        Calls estimate() if available, otherwise predict(). Errors are silently caught
        for robustness, as estimator failures should not crash the control loop.
        """
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
        """Run the control loop asynchronously.

        Executes the sense-compute-actuate cycle at the configured period until interrupted
        or max_iters is reached. On exit, writes safe shutdown values to all control outputs.

        Parameters
        ----------
        max_iters : Optional[int], default=None
            Maximum number of iterations to run. If None, runs indefinitely until interrupted.

        Notes
        -----
        The control loop performs the following steps each iteration:
        1. Read state and parameter values from OPC UA server
        2. Compute control action using the controller
        3. Write control action to OPC UA server
        4. Execute estimator step (if configured)
        5. Sleep for the configured period

        On exit (normal or via exception), safe shutdown values are written to ensure
        the system is left in a safe state.
        """
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
        """Synchronous convenience wrapper around the async run method.

        Parameters
        ----------
        max_iters : Optional[int], default=None
            Maximum number of iterations to run. If None, runs indefinitely until interrupted.

        Notes
        -----
        This method creates a new event loop, runs the async control loop, and cleans up.
        Use this method when calling from synchronous code. For async contexts, use run() directly.
        """
        asyncio.run(self.run(max_iters=max_iters))
