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

from dataclasses import dataclass, field
from typing import Dict, Any, Iterable, Optional, Tuple, Sequence, TypedDict, Union

# Optional dependency: asyncua (opcua-asyncio)
try:
    from asyncua import Client as UAClient
    from asyncua import ua
except Exception:  # pragma: no cover - handled at runtime with clear error
    UAClient = None  # type: ignore
    ua = None  # type: ignore


class ReadItem(TypedDict, total=False):
    """Structured item returned by reads.

    Keys:
    - value: The decoded value (possibly scaled to engineering units)
    - status: UA status code object (if asyncua available) or None
    - ts: Server/source timestamp if available; None in minimal version
    """
    value: Any
    status: Any
    ts: Any


Number = Union[int, float]


@dataclass
class IOMapping:
    """
    A simple dictionary of nicknames (aliases) to real OPC UA addresses (NodeIds).

    Why aliases? Because typing a long NodeId like "ns=2;s=Plant/Speed" everywhere is hard to read
    and easy to mistype. So we map:
        alias -> NodeId string

    We also store two optional things per alias:
    - scale: clever way to convert raw numbers to pretty numbers (engineering units).
             If scale is {a: 2, b: 3}, then pretty = 2*raw + 3.
    - bounds: minimum and maximum allowed values when we write, to keep things safe.

    Example mapping (YAML-like):
        reads:
          y_speed: { node: 'ns=2;s=Plant/Speed',  scale: {a: 1.0, b: 0.0} }
        writes:
          u_pitch: { node: 'ns=2;s=Plant/BladePitch_deg', bounds: [0, 30] }
    """

    reads: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    writes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def node_of(self, key: str) -> str:
        """Return the NodeId string for an alias.

        Parameters
        ----------
        key: str
            Alias as defined in reads/writes.

        Returns
        -------
        str
            The configured NodeId string, e.g. "ns=2;s=Plant/Speed".

        Raises
        ------
        KeyError
            If the alias is not present in either reads or writes.
        """
        if key in self.reads:
            return self.reads[key]["node"]
        if key in self.writes:
            return self.writes[key]["node"]
        raise KeyError(f"Alias '{key}' not found in mapping")

    def get_scale(self, key: str) -> Optional[Tuple[float, float]]:
        """Return (a, b) for linear scale pretty = a*raw + b, if configured.

        Returns None when no scale exists for this alias.
        """
        cfg = self.reads.get(key) or self.writes.get(key)
        if not cfg:
            return None
        scale = cfg.get("scale")
        if not scale:
            return None
        a = float(scale.get("a", 1.0))
        b = float(scale.get("b", 0.0))
        return a, b

    def get_bounds(self, key: str) -> Optional[Tuple[float, float]]:
        """Return (min, max) bounds if defined for a write alias; else None."""
        cfg = self.writes.get(key)
        if not cfg:
            return None
        bnd = cfg.get("bounds")
        if not bnd:
            return None
        return float(bnd[0]), float(bnd[1])


def build_mapping_from_model(
    model: Any,
    ns_idx: int,
    base_path: str,
    read_from: Sequence[str] | str = ("y",),
    write_from: Sequence[str] | str = ("u",),
    write_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    scales: Optional[Dict[str, Tuple[float, float]]] = None,
    node_overrides: Optional[Dict[str, str]] = None,
    include_read_names: Optional[Sequence[str]] = None,
    include_write_names: Optional[Sequence[str]] = None,
) -> IOMapping:
    """Build an IOMapping from a HILO model using variable names and a path template.

    - NodeId format: f"ns={ns_idx};s={base_path}/{name}"
    - read_from: which variable groups to expose as reads: any of 'x','y','z','u','p'.
    - write_from: variable groups to expose as writes (commonly 'u').
    - write_bounds: optional per-name (lo, hi) bounds applied on write.
    - scales: optional per-name (a, b) linear scaling for read/write.

    This makes it nearly zero-boilerplate to connect a controller to an OPC UA server
    when the server exposes nodes named exactly after the model variable names under a
    common base path.
    Parameters
    ----------
    model: Any
        HILO model instance; must expose name lists like `input_names`, `dynamical_state_names`, etc.
    ns_idx: int
        Namespace index where the server publishes the nodes.
    base_path: str
        Base path segment for nodes, e.g. "Pendulum" so nodes become "Pendulum/theta".
    read_from: Sequence[str] | str
        Model groups to expose as readable aliases; subset of {"x","y","z","u","p"}.
    write_from: Sequence[str] | str
        Model groups to expose as writable aliases (commonly {"u"}).
    write_bounds: dict | None
        Optional per-name bounds applied before write, as {name: (lo, hi)}.
    scales: dict | None
        Optional per-name linear scaling (a, b) applied on read and inversely on write.
    node_overrides: dict | None
        Optional explicit NodeId per name; takes precedence over the template.
    include_read_names: Sequence[str] | None
        If provided, limit read aliases to this exact subset of names.
    include_write_names: Sequence[str] | None
        If provided, limit write aliases to this exact subset of names.

    Returns
    -------
    IOMapping
        Mapping ready to use with AsyncOPCUAClient/OPCUAConnector.

    Notes
    -----
    - NodeId format used here is string-based: f"ns={ns_idx};s={base_path}/{name}".
    - This is convenience only; for complex addressing schemes, pass explicit overrides.
    """
    def _names(kind: str) -> Sequence[str]:
        kind = kind.lower()
        if kind == "x":
            return getattr(model, "dynamical_state_names", [])
        if kind == "y":
            return getattr(model, "measurement_names", [])
        if kind == "z":
            return getattr(model, "algebraic_state_names", [])
        if kind == "u":
            return getattr(model, "input_names", [])
        if kind == "p":
            return getattr(model, "parameter_names", [])
        return []

    if isinstance(read_from, str):
        read_from = (read_from,)
    if isinstance(write_from, str):
        write_from = (write_from,)

    reads: Dict[str, Dict[str, Any]] = {}
    writes: Dict[str, Dict[str, Any]] = {}

    overrides = node_overrides or {}

    include_read: Optional[set] = set(include_read_names) if include_read_names else None
    include_write: Optional[set] = set(include_write_names) if include_write_names else None

    for grp in read_from:
        for name in _names(grp):
            if include_read is not None and name not in include_read:
                continue
            node = overrides.get(name, f"ns={ns_idx};s={base_path}/{name}")
            cfg: Dict[str, Any] = {"node": node}
            if scales and name in scales:
                a, b = scales[name]
                cfg["scale"] = {"a": float(a), "b": float(b)}
            reads[name] = cfg

    for grp in write_from:
        for name in _names(grp):
            if include_write is not None and name not in include_write:
                continue
            node = overrides.get(name, f"ns={ns_idx};s={base_path}/{name}")
            cfg2: Dict[str, Any] = {"node": node}
            if write_bounds and name in write_bounds:
                lo, hi = write_bounds[name]
                cfg2["bounds"] = [float(lo), float(hi)]
            if scales and name in scales:
                a, b = scales[name]
                cfg2["scale"] = {"a": float(a), "b": float(b)}
            writes[name] = cfg2

    return IOMapping(reads=reads, writes=writes)


class AsyncOPCUAClient:
    """
    Tiny async OPC UA client that speaks in aliases, not NodeIds.

    Simple mental picture:
    - endpoint is the doorbell address of the machine box, like
      "opc.tcp://127.0.0.1:4840/freeopcua/server/".
    - connect() rings the bell and enters.
    - read([aliases]) asks for the current numbers by their nicknames.
    - write({alias: value}) sends a new number to the machine.
    - disconnect() politely leaves.

    Usage example:
        client = AsyncOPCUAClient(endpoint, mapping)
        await client.connect()
        data = await client.read(["y_speed"])  # {"y_speed": {"value": 123.4, "status": ..., "ts": ...}}
        await client.write({"u_pitch": 5.0})
        await client.disconnect()
    """

    def __init__(
        self,
        endpoint: str,
        mapping: IOMapping,
        timeout: float = 2.0,
        security_mode: Optional[str] = None,
        security_policy: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        cert_path: Optional[str] = None,
        key_path: Optional[str] = None,
        reconnect_backoff: Tuple[float, float] = (0.5, 5.0),
        max_reconnect_attempts: Optional[int] = None,
    ) -> None:
        """Store connection settings and the alias mapping.

        Parameters
        ----------
        endpoint: str
            OPC UA endpoint URL, e.g. "opc.tcp://127.0.0.1:4840/freeopcua/demo/".
        mapping: IOMapping
            Alias-to-NodeId mapping including optional scale and bounds.
        timeout: float
            Request timeout (seconds).
        security_mode, security_policy, username, password, cert_path, key_path
            Optional security/authentication parameters. Not used in the minimal demo.
        reconnect_backoff: (float, float)
            Initial and maximum delays (s) for exponential backoff during reconnect.
        max_reconnect_attempts: int | None
            If set, stop retrying after this many consecutive reconnect attempts.

        Raises
        ------
        ImportError
            If asyncua is not installed in the current environment.
        """
        self._endpoint = endpoint
        self._mapping = mapping
        self._timeout = timeout
        self._security_mode = security_mode
        self._security_policy = security_policy
        self._username = username
        self._password = password
        self._cert_path = cert_path
        self._key_path = key_path
        self._client: Optional[Any] = None
        # Reconnection config/state
        self._reconnect_backoff = reconnect_backoff
        self._max_reconnect_attempts = max_reconnect_attempts
        self._consec_failures = 0

        if UAClient is None:
            raise ImportError(
                "asyncua (opcua-asyncio) is not installed. Install with 'poetry install -E connectivity' or 'pip install asyncua'."
            )

    async def connect(self) -> None:
        """Open a session with the server.

        Applies minimal security/user configuration if provided. For local demos this
        typically uses an unsecured connection.
        """
        client = UAClient(self._endpoint, timeout=self._timeout)  # type: ignore[arg-type]

        # Security configuration (minimal; extend as needed)
        if self._security_mode and self._security_policy:
            # Map simple strings to ua.SecurityPolicyURIs when provided
            policy_uri = getattr(ua.SecurityPolicyURIS, self._security_policy, None) if ua else None
            mode = getattr(ua.MessageSecurityMode, self._security_mode, None) if ua else None
            if policy_uri and mode:
                await client.set_security(policy_uri, certificate=self._cert_path, private_key=self._key_path, mode=mode)  # type: ignore[func-returns-value]

        if self._username and self._password:
            client.set_user(self._username)
            client.set_password(self._password)

        await client.connect()
        self._client = client

    async def disconnect(self) -> None:
        """Close the session when finished."""
        if self._client is not None:
            await self._client.disconnect()
            self._client = None

    def _require_client(self) -> Any:
        """Return the internal asyncua client; require an active connection.

        Raises
        ------
        RuntimeError
            If connect() has not been called successfully yet.
        """
        if self._client is None:
            raise RuntimeError("Client is not connected. Call connect() first.")
        return self._client

    async def ensure_connected(self) -> None:
        """Ensure there is an active session; connect if needed."""
        if self._client is None:
            await self.connect()

    async def _attempt_reconnect(self) -> None:
        """Reconnect with exponential backoff.

        Raises after exceeding max_reconnect_attempts when configured.
        """
        import asyncio
        lo, hi = self._reconnect_backoff
        delay = max(0.0, float(lo))
        attempts = 0
        while True:
            try:
                await self.connect()
                self._consec_failures = 0
                return
            except Exception:
                attempts += 1
                self._consec_failures += 1
                if self._max_reconnect_attempts is not None and attempts >= self._max_reconnect_attempts:
                    raise
                # backoff
                await asyncio.sleep(delay)
                delay = min(delay * 2 if delay > 0 else lo, hi)

    async def read(self, keys: Iterable[str]) -> Dict[str, ReadItem]:
        """Read current values for a list of aliases.

        Steps for each alias:
        - Look up the real NodeId string from the mapping.
        - Ask the server for the value.
        - If a scale is defined, apply pretty = a*raw + b.
        - Return a small dict with value, status, and timestamp (ts is None in this minimal version).
        Parameters
        ----------
        keys: Iterable[str]
            Aliases to read (must be present in mapping.reads or writes).

        Returns
        -------
        dict[str, ReadItem]
            A dictionary keyed by alias with value/status/ts entries.
        """
        await self.ensure_connected()
        client = self._require_client()
        out: Dict[str, ReadItem] = {}
        for k in keys:
            try:
                nodeid = self._mapping.node_of(k)
                node = client.get_node(nodeid)
                value = await node.read_value()
            except Exception:
                # one reconnect attempt then retry once
                await self._attempt_reconnect()
                client = self._require_client()
                node = client.get_node(self._mapping.node_of(k))
                value = await node.read_value()
            status = ua.StatusCode(ua.StatusCodes.Good) if ua else None
            tsrv = None
            # Basic scaling: eng = a*raw + b if provided
            scale = self._mapping.get_scale(k)
            if scale:
                a, b = scale
                try:
                    value = a * float(value) + b
                except Exception:
                    # leave as-is if not numeric
                    pass
            out[k] = {"value": value, "status": status, "ts": tsrv}
        return out

    async def write(self, data: Dict[str, Number]) -> None:
        """Write values by alias.

        Steps for each alias:
        - If scale exists, reverse it: raw = (pretty - b)/a before sending to the server.
        - If bounds exist, clamp the value inside [min, max] for safety.
        - Send the value to the NodeId.
        Parameters
        ----------
        data: dict[str, Number]
            Mapping from write alias to numeric value in engineering units.
        """
        await self.ensure_connected()
        client = self._require_client()
        for k, v in data.items():
            nodeid = self._mapping.node_of(k)
            # Apply inverse scaling if provided: raw = (eng - b)/a
            scale = self._mapping.get_scale(k)
            if scale:
                a, b = scale
                try:
                    v = (float(v) - b) / a
                except Exception:
                    # If scaling fails (e.g., non-numeric value), use original value
                    pass
            bounds = self._mapping.get_bounds(k)
            if bounds:
                lo, hi = bounds
                try:
                    v = min(max(float(v), lo), hi)
                except Exception:
                    pass
            try:
                node = client.get_node(nodeid)
                await node.write_value(v)
            except Exception:
                await self._attempt_reconnect()
                client = self._require_client()
                node = client.get_node(nodeid)
                await node.write_value(v)
