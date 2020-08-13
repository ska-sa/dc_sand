"""A fake node.

This was used in manual tests of the corr3-proto-servlet, in order to demonstrate whether or not the servlet was
actually connecting to another katcp device server and passing messages forward.

Pass the port to run the server on as an argument. Or don't, and it'll default to 1234.
"""
import asyncio
import aiokatcp
import logging
import sys

from aiokatcp import Sensor
from ngkcs.data_processor import (
    DeviceStatus,
    device_status_to_sensor_status,
)

LOCALHOST = "127.0.0.1"
DEFAULT_PORT = 5678
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class FakeNode(aiokatcp.DeviceServer):
    """A simple DeviceServer masquerading as a hypothetical DSP node.

    There are some `logging' statements included for debugging purposes,  it's not anticipated that this
    code will be used for anything more than that.
    """

    VERSION = "version"
    BUILD_STATE = "build-state"

    def __init__(
        self, host: str = LOCALHOST, port: int = DEFAULT_PORT, **kwargs,
    ):
        """Initialise the FakeNode DeviceServer with the necessary properties."""
        super(FakeNode, self).__init__(host, port=port, **kwargs)

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
