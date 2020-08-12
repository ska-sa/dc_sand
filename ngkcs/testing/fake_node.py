"""A fake node.

This was used in manual tests of the corr3-proto-servlet, in order to demonstrate whether or not the servlet was
actually connecting to another katcp device server and passing messages forward.

Pass the port to run the server on as an argument. Or don't, and it'll default to 1234.
"""
import asyncio
import aiokatcp
import logging
import os
import sys
from configparser import ConfigParser
from typing import Type

from aiokatcp import (
    FailReply,
    Sensor,
)
from ngkcs.data_processor import (
    DeviceStatus,
    ProcessorState,
    DataProcessorBase,
    DataProcessorInterface,
    device_status_to_sensor_status,
)

LOCALHOST = "127.0.0.1"
DEFAULT_PORT = 5678
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def parse_config_file(config_file=""):
    """
    Parse an config file into a dictionary. No checking done at all.
    
    Placing it here for now - will eventually be moved into a utils.py.

    :param config_file: the config file to process
    :param required_sections: sections that MUST be included
    :return: a dictionary containing the configuration
    """
    parser = ConfigParser()
    files = parser.read(config_file)
    if len(files) == 0:
        raise IOError("Could not read the config file, %s" % config_file)

    config_dict = {}

    for section in parser.sections():
        config_dict[section] = {}
        for items in parser.items(section):
            config_dict[section][items[0]] = items[1]

    return config_dict


class FakeNode(aiokatcp.DeviceServer):
    """A simple DeviceServer masquerading as a hypothetical DSP node.

    There are some `logging' statements included for debugging purposes,  it's not anticipated that this
    code will be used for anything more than that.
    """

    VERSION = "version"
    BUILD_STATE = "build-state"

    def __init__(
        self,
        host: str = LOCALHOST,
        port: int = DEFAULT_PORT,
        cbf_servlet: aiokatcp.Client = None,  # Defaulting to None, for now
        shutdown_delay: float = 7.0,  # Delay before completing ?halt
        *args,
        **kwargs,
    ):
        """Initialise the FakeNode DeviceServer with the necessary properties."""
        self.data_processor = None
        self.cbf_servlet = cbf_servlet
        self.shutdown_delay = shutdown_delay

        super(FakeNode, self).__init__(host, port=port, *args, **kwargs)

        self.beam_weights_set = False

        self.sensors.add(
            Sensor(
                DeviceStatus,
                "device-status",
                "Devices status of the data processor",
                default=DeviceStatus.OK,
                status_func=device_status_to_sensor_status,
            )
        )

    async def start(self, *args, **kwargs):
        """Override base method in order to log the port we're on. For debug."""
        logger.debug(f"Starting FakeNode server on port {self._port}")
        await super(FakeNode, self).start(*args, **kwargs)

    async def on_stop(self) -> None:
        """Add extra clean-up before finally halting the server."""
        # await self._consul_deregister()
        # self._prometheus_watcher.close()
        if self.data_processor is not None and self.data_processor.state != ProcessorState.DEAD:
            logger.warning("Data Processor interrupted - deconfiguring running Data Processor.")
            try:
                await self.data_processor.deconfigure(force=True)
            except Exception:
                logger.warning("Failed to deconfigure %s during shutdown", exc_info=True)

    async def _client_connected_cb(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Provide debug info concerning new connections that the base doesn't give.

        I want to be able to see clearly in my terminal when a client connects, so that I can be sure that the
        clients are actually doing their work.
        """
        # None of this is really necessary, but I want to be able to see the addr:port from which new clients connect.
        old_connections = self._connections.copy()  # Get a set of the old connections.
        await super(FakeNode, self)._client_connected_cb(reader, writer)  # Let the DeviceServer add the new one.
        new_connection = self._connections.difference(old_connections)  # The new connection will be the only one.
        logger.debug(f"Client connected from {new_connection.pop().address}")
        # This all just goes to show that Python doesn't really have proper encapsulating and data hiding.
        # So as a result any cowboy like me can plunder base classes for things that really should be left to do their own thing.

    async def request_beam_weights(self, ctx, data_stream, *weights):
        """Load weights for all inputs on a specified beam data-stream."""
        logger.debug("Received the beam-weights request.")
        self.beam_weights_set = True  # Obiously in a production version, we'd check that the request was correct.

    async def request_configure(self, ctx, data_proc_name: str, config_filename: str) -> None:
        """Configure a data processor instance.

        Parameters
        ----------
        data_proc_name : str
            Name of the data processor.
        config_filename : str
            Traditional corr2 config-file filename, for now
        """
        logger.info(f"?configure called with: {ctx.req}")

        config_dict = None

        if self.data_processor is not None and self.data_processor.state != ProcessorState.DEAD:
            raise FailReply("Already configured or configuring")
        try:
            logger.debug(f"Trying to create and configure product {data_proc_name.lower()}")

            # Check if the config-file exists
            abs_path = os.path.abspath(config_filename)
            if not os.path.exists(abs_path):
                # Problem
                errmsg = "Config-file {} is not valid".format(config_filename)
                logger.error(errmsg)
                raise FailReply(errmsg)
            # else: Continue!
            config_dict = parse_config_file(abs_path)
            # Now, pass it on to the actual configure_product command!
            await self._configure_data_processor(name=data_proc_name.lower(), config_dict=config_dict)

        except Exception as exc:
            retmsg = f"Failed to process config: {exc}"
            logger.error(retmsg)
            raise FailReply(retmsg) from exc

    async def _configure_data_processor(self, name: str, config_dict: dict) -> None:
        """Configure a data processor in response to a request."""

        def dead_callback(product):
            if self.shutdown_delay > 0:
                logger.info("Sleeping %.1f seconds to give time for final tasks to complete", self.shutdown_delay)
                asyncio.get_event_loop().call_later(self.shutdown_delay, self.halt, False)
            else:
                self.halt(False)

        logger.debug(f"Received config data: {config_dict}")

        asyncio.sleep(0.5)

        # Create DataProcessor in 'interface mode'
        data_processor_cls: Type[DataProcessorBase] = DataProcessorInterface
        data_processor = data_processor_cls(data_proc_id=name, config=config_dict, proc_controller=self)
        self.data_processor = data_processor
        self.data_processor.dead_callbacks.append(dead_callback)

        try:
            await self.data_processor.configure()
        except Exception:
            self.data_processor = None
            raise

    def _get_data_processor(self):
        """Check that self.data_processor exists (i.e. ?product-configure has been called).

        If it has not, raises a :exc:`FailReply`.
        """
        if self.data_processor is None:
            raise FailReply("?product-configure has not been called yet. It must be called before other requests.")
        return self.data_processor

    async def request_deconfigure(self, ctx, force: bool = False) -> None:
        """Deconfigure the data_processor and shut down the server."""
        await self._get_data_processor().deconfigure(force=force)


async def main():
    """Execute the program. Go on, hop to."""
    port = DEFAULT_PORT
    if len(sys.argv) >= 2:  # Crude primitive arg parsing.
        port = sys.argv[1]
    server = FakeNode(host=LOCALHOST, port=port)
    await server.start()
    await server.join()  # Technically not really needed, as there's no cleanup afterwards as things currently stand.


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
