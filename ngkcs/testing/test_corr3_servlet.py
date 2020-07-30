"""Unit test for the Corr3Servlet class."""

import pytest
import asyncio
from ngkcs.corr3_servlet import Corr3Servlet
import aiokatcp
from fake_node import FakeNode

CORR3_SERVLET_PORT = 7404
ARRAY_SIZE = 4  # TODO: Need to parameterise the unit tests. Only works for 4 at the moment.


@pytest.fixture(scope="function")
async def corr3_servlet_test_fixture(unused_tcp_port_factory):
    """Provide an instance of Corr3Servlet to test.

    This fixture readies a Corr3Servlet instance in a running state, that can be queried and the results
    compared with the desired output to test for correctness of operation.

    At the moment, the parameters of the Corr3Servlet are hard-coded. Ideally we'd like to parameterise this,
    but I'm not quite sure how to do that yet.

    I'd also like to scope the fixture for the class instead of the function. It makes more sense to me, rather than
    tearing down and rebuilding the servlet for every single little test. I'll defer to those with more wisdom if this
    is not a good idea. The main obstacle at the moment is that this involves changing the scope of the default
    `event_loop` fixture as well. I'm not sure how to do that yet either.
    """
    ports = [unused_tcp_port_factory() for _ in range(ARRAY_SIZE)]
    fake_device_servers = [FakeNode("localhost", port) for port in ports]
    for fake_ds in fake_device_servers:
        # TODO: figure out how to do this concurrently. Should be fine as-is for small tests.
        await fake_ds.start()
    servlet = Corr3Servlet(
        name="stephen",
        n_antennas=ARRAY_SIZE,
        host="0.0.0.0",
        port=CORR3_SERVLET_PORT,
        x_engine_endpoints=[("localhost", port) for port in ports],
    )
    await servlet.start()

    # Using `yield` in a fixture allows execution to come back to the fixture once the test is done with it, so that
    # necessary cleanup can happen afterwards.
    yield fake_device_servers
    await servlet.stop()
    await servlet.join()
    for fake_ds in fake_device_servers:
        await fake_ds.stop()
        await fake_ds.join()


async def make_a_request(*args):
    """Connect to the servlet and send a request."""
    client = await aiokatcp.Client.connect("localhost", CORR3_SERVLET_PORT)
    try:
        # We put this inside a `try/finally` block in order to make sure that everything gets cleaned up properly.
        # In some tests, we may expect an exception from deliberately sending a bad request.
        reply, informs = await client.request(*args)
    except aiokatcp.connection.FailReply:
        reply, informs = None, None
        raise
    finally:
        # So we use this to ensure that we don't leave the client dangling.
        client.close()
        await client.wait_closed()
    return reply, informs


class TestCorr3Servlet:
    """Class for grouping Corr3Servlet tests."""

    def test_beam_weights(self, corr3_servlet_test_fixture, event_loop):
        """Test a properly-formed ?beam-weights request."""
        event_loop.run_until_complete(
            make_a_request("beam-weights", "tied-array-channelised-voltage", "1", "2", "3", "4")
        )
        for fake_device_server in corr3_servlet_test_fixture:
            assert fake_device_server.beam_weights_set

    def test_beam_weights_wrong_numbers(self, corr3_servlet_test_fixture, event_loop):
        """Test a ?beam-weights request with the wrong number of weights being passed."""
        # This context manager tells `pytest` that we expect an exception to be raised.
        with pytest.raises(aiokatcp.connection.FailReply):
            event_loop.run_until_complete(
                make_a_request("beam-weights", "tied-array-channelised-voltage", "1", "2", "3")
            )  # Only 3 weights, 4 expected.
        for fake_device_server in corr3_servlet_test_fixture:
            assert not fake_device_server.beam_weights_set  # i.e. verify that the message didn't get through.

    @pytest.mark.xfail  # We expect this one to fail because I haven't built the logic to ignore wrong streams yet.
    def test_beam_weights_wrong_stream(self, corr3_servlet_test_fixture, event_loop):
        """Test a ?beam-weights request with the wrong data-stream being passed."""
        event_loop.run_until_complete(
            make_a_request("beam-weights", "baseline-correlation-products", "1", "2", "3", "4")
        )  # i.e. the stream is not tied-array-channelised-voltage.
        for fake_device_server in corr3_servlet_test_fixture:
            assert not fake_device_server.beam_weights_set

    def test_sensor_propagation(self, corr3_servlet_test_fixture, event_loop):
        """Test that the sensor values presented by the nodes are propagated to the servlet.
        
        I'm not sure that this function-within-a-function approach is the best way to go about this
        test, but in my experience, multiple run_until_complete() calls are a recipe for frustration.
        So I decided just to wrap it all up so that a single call would work.
        """

        async def changing_sensor_check():
            """Check a sensor, change it, then check it again.

            We are hard-coding this for the time being just to check a single sensor on a single node. This will
            at least verify that the mechanism works. Ideally it should be parameterised, but I haven't gotten
            around to understanding properly how to do that with pytest yet.
            """
            # I'm going to assume that we are testing just node1 (there should be from node0 to node3 available.)
            _reply, informs = await make_a_request("sensor-value", "node1.device-status")
            initial_timestamp = float(informs[0].arguments[0])  # First field in the inform is a UNIX timestamp
            assert informs[0].arguments[2] == b"node1.device-status"  # sensor-name
            assert informs[0].arguments[3] == b"unknown"  # sensor status. This should be "unknown" to start with.
            assert informs[0].arguments[4] == b"default"  # The actual sensor value.

            # Let a little bit of time pass so that we can test the timestamp mechanism
            await asyncio.sleep(0.1)

            # I'm not quite sure of the best way to address specific fake-nodes so we will do them all.
            for fake_device_server in corr3_servlet_test_fixture:
                fake_device_server.modify_device_status_sensor("changed")

            _reply, informs = await make_a_request("sensor-value", "node1.device-status")
            final_timestamp = float(informs[0].arguments[0])
            assert final_timestamp > initial_timestamp
            assert informs[0].arguments[2] == b"node1.device-status"
            assert informs[0].arguments[3] == b"nominal"
            assert informs[0].arguments[4] == b"changed"

        event_loop.run_until_complete(changing_sensor_check())
