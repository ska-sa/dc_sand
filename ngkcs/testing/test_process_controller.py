"""Unit test for the FakeNode class."""

import pytest
import aiokatcp
from ngkcs.process_controller import ProcessController
from ngkcs.data_processor import (
    DeviceServer,
    ProcessorState,
)


LOCALHOST = "127.0.0.1"
DEFAULT_PORT = 5678
NUM_NODES = 2


@pytest.fixture(scope="function")
async def process_controller_test_fixture(unused_tcp_port_factory):
    """Provide an instance of ProcessController to test."""
    ports = [unused_tcp_port_factory() for _ in range(NUM_NODES)]
    data_processor_instances = [DeviceServer(host=LOCALHOST, port=port) for port in ports]

    for data_processor_instance in data_processor_instances:
        await data_processor_instance.start()

    # Or perhaps:
    # asyncio.gather([data_proc_instance.start() for data_proc_instance in data_processor_instances])

    process_controller_inst = ProcessController(
        name="test", processor_endpoints=[(LOCALHOST, port) for port in ports], host=LOCALHOST, port=DEFAULT_PORT,
    )

    await process_controller_inst.start()

    yield data_processor_instances
    await process_controller_inst.stop()
    await process_controller_inst.join()
    for data_processor_instance in data_processor_instances:
        await data_processor_instance.stop()
        await data_processor_instance.join()


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


class TestProcessController:
    """Class for grouping ProcessController tests."""

    def test_configure(self, process_controller_test_fixture, event_loop):
        """Test a properly-formed ?configure-processors request."""
        event_loop.run_until_complete(make_a_request("configure-processors", "test", "testing/test_4k_corr.ini"))
        for data_proc_instance in process_controller_test_fixture:
            assert data_proc_instance.data_processor.state == ProcessorState.IDLE

    def test_configure_invalid_config_file(self, process_controller_test_fixture, event_loop):
        """Test a ?configure-processors request with an invalid config file."""
        # This context manager tells `pytest` that we expect an exception to be raised.
        with pytest.raises(aiokatcp.connection.FailReply):
            event_loop.run_until_complete(make_a_request("configure-processors", "test", "amish.ini"))

    def test_deconfigure(self, process_controller_test_fixture, event_loop):
        """Test a ?deconfigure-processors command."""
        event_loop.run_until_complete(make_a_request("configure-processors", "test", "testing/test_4k_corr.ini"))

        event_loop.run_until_complete(make_a_request("deconfigure-processors"))
        for data_proc_instance in process_controller_test_fixture:
            assert data_proc_instance.data_processor.state == ProcessorState.DEAD

    def test_server_halt(self, process_controller_test_fixture, event_loop):
        """Test a ?halt command with an already-running DataProcessor."""
        # First, (re)configure a DataProcessor
        event_loop.run_until_complete(make_a_request("configure-processors", "test", "testing/test_4k_corr.ini"))

        event_loop.run_until_complete(make_a_request("halt"))
