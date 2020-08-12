"""Unit test for the FakeNode class."""

import pytest
import aiokatcp
from fake_node import FakeNode
from ngkcs.data_processor import ProcessorState


LOCALHOST = "127.0.0.1"
DEFAULT_PORT = 5678
ARRAY_SIZE = 4


@pytest.fixture(scope="function")
async def fake_node_test_fixture():
    """Provide an instance of FakeNode to test.

    This fixture readies a FakeNode instance in a running state, that can be queried and the results
    compared with the desired output to test for correctness of operation.
    """
    fake_node_instances = [FakeNode(host=LOCALHOST, port=DEFAULT_PORT)]
    for fake_node_instance in fake_node_instances:
        await fake_node_instance.start()

    yield fake_node_instances
    for fake_node_instance in fake_node_instances:
        await fake_node_instance.stop()
        await fake_node_instance.join()


async def make_a_request(*args):
    """Connect to the servlet and send a request."""
    client = await aiokatcp.Client.connect(host=LOCALHOST, port=DEFAULT_PORT)
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


class TestFakeNode:
    """Class for grouping FakeNode tests."""

    def test_configure(self, fake_node_test_fixture, event_loop):
        """Test a properly-formed ?configure request."""
        event_loop.run_until_complete(make_a_request("configure", "test", "testing/test_4k_corr.ini"))
        for fake_device_server in fake_node_test_fixture:
            assert fake_device_server.data_processor is not None

    def test_configure_invalid_config_file(self, fake_node_test_fixture, event_loop):
        """Test a ?configure request with an invalid config file."""
        # This context manager tells `pytest` that we expect an exception to be raised.
        with pytest.raises(aiokatcp.connection.FailReply):
            event_loop.run_until_complete(make_a_request("configure", "test", "amish.ini"))

    def test_deconfigure(self, fake_node_test_fixture, event_loop):
        """Test a ?deconfigure command."""
        event_loop.run_until_complete(make_a_request("configure", "test", "testing/test_4k_corr.ini"))

        event_loop.run_until_complete(make_a_request("deconfigure"))
        for fake_device_server in fake_node_test_fixture:
            assert fake_device_server.data_processor.state == ProcessorState.DEAD

    def test_server_halt(self, fake_node_test_fixture, event_loop):
        """Test a ?halt command with an already-running DataProcessor."""
        # First, (re)configure a DataProcessor
        event_loop.run_until_complete(make_a_request("configure", "test", "testing/test_4k_corr.ini"))

        event_loop.run_until_complete(make_a_request("halt"))
