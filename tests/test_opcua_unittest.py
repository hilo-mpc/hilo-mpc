import asyncio
import socket
import unittest

from hilo_mpc.plugins.io.opcua_async import IOMapping, AsyncOPCUAClient

try:
    from asyncua import Server, ua  # type: ignore
    ASYNCUA_AVAILABLE = True
except Exception:
    ASYNCUA_AVAILABLE = False


def _find_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


async def _start_server():
    port = _find_free_port()
    endpoint = f"opc.tcp://127.0.0.1:{port}/unittest/opcua/"
    server = Server()
    await server.init()
    server.set_endpoint(endpoint)
    idx = await server.register_namespace("http://hilo-mpc.test")
    objects = await server.nodes.objects.add_object(idx, "Test")
    angle = await objects.add_variable(ua.NodeId("Test/Angle", idx), "Angle", 0.0)
    torque = await objects.add_variable(ua.NodeId("Test/Torque", idx), "Torque", 0.0)
    await torque.set_writable()
    await server.start()
    return server, angle, torque, idx, endpoint


class TestIOMapping(unittest.TestCase):
    def test_basics(self):
        m = IOMapping(
            reads={"y": {"node": "ns=2;s=Test/Angle", "scale": {"a": 2.0, "b": 1.0}}},
            writes={"u": {"node": "ns=2;s=Test/Torque", "bounds": [-1.0, 1.0]}},
        )
        self.assertEqual(m.node_of("y"), "ns=2;s=Test/Angle")
        self.assertEqual(m.get_scale("y"), (2.0, 1.0))
        self.assertEqual(m.get_bounds("u"), (-1.0, 1.0))
        with self.assertRaises(KeyError):
            _ = m.node_of("missing")


@unittest.skipUnless(ASYNCUA_AVAILABLE, "asyncua not installed")
class TestAsyncIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server, self.angle, self.torque, self.idx, self.endpoint = await _start_server()

    async def asyncTearDown(self):
        await self.server.stop()

    async def test_client_read_write_scale_bounds(self):
        mapping = IOMapping(
            reads={"y": {"node": f"ns={self.idx};s=Test/Angle", "scale": {"a": 2.0, "b": -1.0}}},
            writes={"u": {"node": f"ns={self.idx};s=Test/Torque", "bounds": [-0.5, 0.5]}},
        )
        client = AsyncOPCUAClient(endpoint=self.endpoint, mapping=mapping)
        await client.connect()
        try:
            await self.angle.write_value(1.25)  # raw
            data = await client.read(["y"])  # 2*1.25 - 1 = 1.5
            self.assertAlmostEqual(float(data["y"]["value"]), 1.5, places=6)

            await client.write({"u": 0.9})  # expect clamp to 0.5
            raw = await self.torque.read_value()
            self.assertAlmostEqual(float(raw), 0.5, places=6)
        finally:
            await client.disconnect()

    async def test_opcua_loop_safe_shutdown(self):
        from hilo_mpc.plugins.io.opcua_loop import OPCUALoop

        mapping = IOMapping(
            reads={"y": {"node": f"ns={self.idx};s=Test/Angle"}},
            writes={"u": {"node": f"ns={self.idx};s=Test/Torque"}},
        )

        async def runner():
            loop = OPCUALoop(
                endpoint=self.endpoint,
                mapping=mapping,
                period=0.01,
                reconnect_backoff=(0.01, 0.05),
                safe_shutdown={"u": 0.0},
            )

            def compute(_inputs):
                return {"u": 0.3}

            await loop.run(compute, max_iters=2)

        await runner()
        final_u = await self.torque.read_value()
        self.assertAlmostEqual(float(final_u), 0.0, places=9)

    async def test_simple_control_loop_dm_output(self):
        from hilo_mpc.plugins.io.opcua_simple_control_loop import OPCUASimpleControlLoop
        import numpy as np

        class DummyDM:
            def __init__(self, arr):
                self._arr = np.array(arr, dtype=float)
            def toarray(self):
                return self._arr

        class DummyCtrl:
            type = "NMPC"
            def is_setup(self):
                return True
            def optimize(self, x, cp=None):
                return DummyDM([0.42])

        mapping = IOMapping(
            reads={"theta": {"node": f"ns={self.idx};s=Test/Angle"}},
            writes={"u": {"node": f"ns={self.idx};s=Test/Torque"}},
        )

        async def runner():
            loop = OPCUASimpleControlLoop(
                endpoint=self.endpoint,
                mapping=mapping,
                controller=DummyCtrl(),
                estimator=None,
                state_aliases=["theta"],
                control_aliases=["u"],
                period=0.01,
                reconnect_backoff=(0.01, 0.05),
                safe_shutdown={"u": 0.0},
            )
            await loop.run(max_iters=1)

        await runner()
        final_u = await self.torque.read_value()
        self.assertAlmostEqual(float(final_u), 0.0, places=9)
