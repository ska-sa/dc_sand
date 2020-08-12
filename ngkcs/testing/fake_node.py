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
from ngkcs.cbf_subarray_product import (
    DeviceStatus,
    ProductState,
    CBFSubarrayProductBase,
    CBFSubarrayProductInterface,
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
        # product_id: str = None,
        cbf_servlet: aiokatcp.Client = None,  # Defaulting to None, for now
        shutdown_delay: float = 7.0,  # Delay before completing ?halt
        *args,
        **kwargs,
    ):
        """Initialise the FakeNode DeviceServer with the necessary properties."""
        self.product = None
        self.cbf_servlet = cbf_servlet
        self.shutdown_delay = shutdown_delay
        # self.logger = logging.getLogger(name=self.product_id)
        # logging.basicConfig()  # For now

        super(FakeNode, self).__init__(host, port=port, *args, **kwargs)

        self.beam_weights_set = False

        self.sensors.add(
            Sensor(
                DeviceStatus,
                "device-status",
                "Devices status of the subarray product controller",
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
        if self.product is not None and self.product.state != ProductState.DEAD:
            logger.warning("Product controller interrupted - deconfiguring running product")
            try:
                await self.product.deconfigure(force=True)
            except Exception:
                logger.warning("Failed to deconfigure product %s during shutdown", exc_info=True)

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

    async def request_product_configure(self, ctx, product_name: str, config_filename: str) -> None:
        """Configure a CBF Subarray product instance.

        Parameters
        ----------
        name : str
            Name of the subarray product.
        config_filename : str
            Traditional corr2 config-file filename, for now
        """
        logger.info(f"?product-configure called with: {ctx.req}")

        config_dict = None

        if self.product is not None and self.product.state != ProductState.DEAD:
            raise FailReply("Already configured or configuring")
        try:
            # self.product_id = "product1" if product_id is None else product_id.lower()
            logger.debug(f"Trying to create and configure product {product_name.lower()}")

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
            await self.configure_product(name=product_name.lower(), config_dict=config_dict)

        except Exception as exc:
            retmsg = f"Failed to process config: {exc}"
            logger.error(retmsg)
            raise FailReply(retmsg) from exc

    async def configure_product(self, name: str, config_dict: dict) -> None:
        """
        Configure a subarray product in response to a request.

        Raises
        ------
        FailReply
            if a configure/deconfigure is in progress
        FailReply
            If any of the following occur
            - The specified subarray product id already exists, but the config
              differs from that specified
            - If docker python libraries are not installed and we are not using interface mode
            - There are insufficient resources to launch
            - A docker image could not be found
            - If one or more nodes fail to launch (e.g. container not found)
            - If one or more nodes fail to become alive
            - If we fail to establish katcp connection to all nodes requiring them.

        Returns
        -------
        str
            Final name of the subarray-product.

        """

        def dead_callback(product):
            if self.shutdown_delay > 0:
                logger.info("Sleeping %.1f seconds to give time for final tasks to complete", self.shutdown_delay)
                asyncio.get_event_loop().call_later(self.shutdown_delay, self.halt, False)
            else:
                self.halt(False)

        logger.debug(f"Received config data: {config_dict}")

        asyncio.sleep(0.5)

        # Create CBFSubarrayProduct in 'interface mode'
        product_cls: Type[CBFSubarrayProductBase] = CBFSubarrayProductInterface
        product = product_cls(subarray_product_id=name, config=config_dict, product_controller=self)
        self.product = product
        self.product.dead_callbacks.append(dead_callback)

        try:
            await self.product.configure()
        except Exception:
            self.product = None
            raise

    def _get_product(self):
        """Check that self.product exists (i.e. ?product-configure has been called).

        If it has not, raises a :exc:`FailReply`.
        """
        if self.product is None:
            raise FailReply("?product-configure has not been called yet. " "It must be called before other requests.")
        return self.product

    async def request_product_deconfigure(self, ctx, force: bool = False) -> None:
        """Deconfigure the product and shut down the server."""
        await self._get_product().deconfigure(force=force)


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
