"""A fake node.

This was used in manual tests of the corr3-proto-servlet, in order to demonstrate whether or not the servlet was
actually connecting to another katcp device server and passing messages forward.

Pass the port to run the server on as an argument. Or don't, and it'll default to 1234.
"""
import logging
import sys
import asyncio
import aiokatcp

from aiokatcp import (
    FailReply,
    Sensor,
)
from ngkcs.cbf_subarray_product import (
    DeviceStatus,
    ProductState,
    device_status_to_sensor_status,
)

LOCALHOST = "127.0.0.1"
DEFAULT_PORT = 5678


class FakeNode(aiokatcp.DeviceServer):
    """A simple DeviceServer masquerading as a hypothetical DSP node.

    There are some `print()` statements included for debugging purposes,  it's not anticipated that this
    code will be used for anything more than that.
    """

    VERSION = "version"
    BUILD_STATE = "build-state"

    def __init__(
        self,
        host: str = LOCALHOST,
        port: int = DEFAULT_PORT,
        product_id: str = None,
        cbf_servlet: aiokatcp.Client = None,  # Defaulting to None, for now
        shutdown_delay: float = 7.0,  # Delay before completing ?halt
        *args,
        **kwargs,
    ):
        """Override the default to set up some values hopefully useful for unit-testing."""
        self.product_id = "product1" if product_id is None else product_id.lower()
        self.product = None
        self.cbf_servlet = cbf_servlet
        self.shutdown_delay = shutdown_delay
        self.logger = logging.getLogger(name=self.product_id)
        logging.basicConfig()  # For now

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
        """Override base method in order to print the port we're on. For debug."""
        print(f"Starting FakeNode server on port {self._port}")
        await super(FakeNode, self).start(*args, **kwargs)

    async def on_stop(self) -> None:
        """Add extra clean-up before finally halting the server."""
        # await self._consul_deregister()
        # self._prometheus_watcher.close()
        if self.product is not None and self.product.state != ProductState.DEAD:
            self.logger.warning("Product controller interrupted - deconfiguring running product")
            try:
                await self.product.deconfigure(force=True)
            except Exception:
                self.logger.warning("Failed to deconfigure product %s during shutdown", exc_info=True)

    async def _client_connected_cb(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Provide debug info concerning new connections that the base doesn't give.

        I want to be able to see clearly in my terminal when a client connects, so that I can be sure that the
        clients are actually doing their work.
        """
        # None of this is really necessary, but I want to be able to see the addr:port from which new clients connect.
        old_connections = self._connections.copy()  # Get a set of the old connections.
        await super(FakeNode, self)._client_connected_cb(reader, writer)  # Let the DeviceServer add the new one.
        new_connection = self._connections.difference(old_connections)  # The new connection will be the only one.
        print(f"Client connected from {new_connection.pop().address}")
        # This all just goes to show that Python doesn't really have proper encapsulating and data hiding.
        # So as a result any cowboy like me can plunder base classes for things that really should be left to do their own thing.

    async def request_beam_weights(self, ctx, data_stream, *weights):
        """Load weights for all inputs on a specified beam data-stream."""
        print("Received the beam-weights request.")
        self.beam_weights_set = True  # Obiously in a production version, we'd check that the request was correct.

    async def request_product_configure(self, ctx, name: str, config: str) -> None:
        """
        Configure a CBF Subarray product instance.

        Parameters
        ----------
        name : str
            Name of the subarray product.
        config : str
            A JSON-encoded dictionary of configuration data.
        """
        print(f"?product-configure called with: {ctx.req}")

        if self.product is not None:
            raise FailReply("Already configured or configuring")
        try:
            self.logger.debug(f"Trying to create and configure product {self.product_id}")
        except Exception as exc:
            retmsg = f"Failed to process config: {exc}"
            self.logger.error(retmsg)
            raise FailReply(retmsg) from exc

        await self.configure_product(name, config)

    async def configure_product(self, name: str, config: str) -> None:
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
        self.logger.debug(f"Received config data: {config}")

        asyncio.sleep(0.5)

        # Create CBFSubarrayProduct in 'interface mode'

    def _get_product(self):  # -> CBFSubarrayProductBase:
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
    await server.join()  # Technically not really needed,


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
