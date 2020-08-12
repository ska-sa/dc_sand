"""
A Corr3 proto-servlet.

A Katcp device server whose purpose in life is to interpret between old-CAM and NGC.
At the moment this consists only of forwarding a beam-weights request from a client to
a bunch of other DeviceServers, which represent hypothetical processing nodes.
"""

import aiokatcp
import logging
from typing import (
    List,
    Tuple,
)
from ngkcs.data_processor import (
    DeviceStatus,
    device_status_to_sensor_status,
)


class SensorMirror(aiokatcp.SensorWatcher):
    """Proof-of-concept SensorWatcher to demonstrate that we can watch and aggregate node sensors centrally.

    Shouln't actually need to change that much on shift to a production version, turns out the operation
    is relatively simple.
    """

    def __init__(self, client: aiokatcp.Client, server: aiokatcp.DeviceServer, node_name: str):
        """Create an instance of the SensorMirror.
        
        Parameters
        ----------
        client
            An aiokatcp.Client object. The SensorMirror rides on this connection and uses it to get sensor
            information.
        server
            The Corr3Servlet to which the SensorMirror will send its mirrored sensor information.
        node_name
            The name of the node that we are connecting to. This is prepended to the names of the sensors
            coming in to distinguish them from each other.
        """
        super().__init__(client)
        self.server = server
        self._interface_stale = False
        self.node_name = node_name
        logging.debug(f"SensorMirror for {self.node_name} created, connected to {client.host}:{client.port}.")

    def rewrite_name(self, name: str):
        """Prepend the name of the node to the name of the incoming sensors."""
        return f"{self.node_name}.{name}"  # Simple enough. Add the node name in front.

    def sensor_added(self, name: str, description: str, units: str, type_name: str, *args: bytes) -> None:
        """Add a sensor to the server in response to detection of a new sensor downstream."""
        super().sensor_added(name, description, units, type_name, *args)
        self.server.sensors.add(self.sensors[f"{self.node_name}.{name}"])
        self._interface_stale = True
        logging.info(f"Added sensor {name} on {self.node_name}")

    def sensor_removed(self, name: str) -> None:
        """Remove a sensor in response to removal of the corresponding downstream one."""
        del self.server.sensors[f"{self.node_name}.{name}"]
        self._interface_stale = True
        super().sensor_removed(name)
        logging.info(f"Removed sensor {name} on {self.node_name}")

    def batch_stop(self) -> None:
        """Call at the end of a batch of back-to-back updates."""
        if self._interface_stale:
            self.server.mass_inform("interface-changed", "sensor-list")
            self._interface_stale = False
            logging.info(f"Sent interface-changed on {self.node_name}")


class Corr3Servlet(aiokatcp.DeviceServer):
    """Proof-of-concept DeviceServlet to demonstrate how Corr3 could work.
    
    This servlet in its current guise does not do anything to instantiate any other nodes. It presumes that
    these have been handled by someone else, and that they're ready to receive connections.
    """

    VERSION = "corr3_servlet-0.1"
    BUILD_STATE = "corr3_servlet-0.1.0"  # What is the philosophical difference between VERSION and BUILD_STATE?

    def __init__(
        self,
        *,  # I'm forcing all the arguments to be named, I find it helps with readability.
        name: str,
        n_antennas: int,
        host: str,
        port: int,
        x_engine_endpoints: List[Tuple[str, int]],
        **kwargs,
    ):
        """Create an instance of the Corr3Servlet.

        Parameters
        ----------
        name
            A text descriptor of the correlator. Required for historical reasons.
        n_antennas
            The number of antennas in the subarry.
        host
            The interface on which the server should listen for katcp connections. Not really a host, since
            it's running on this host, but we might have multiple ethernet interfaces perhaps.
        port
            The TCP port on which to listen for katcp connections.
        x_engine_endpoints
            A list of tuples describing endpoints, consisting of a (resolvable) hostname or IP address and port. These
            are the nodes on which X-engines are running, so that we can forward requests and aggregate sensors.
        """
        self.name = name
        self.n_antennas = n_antennas
        self.x_engine_endpoints = x_engine_endpoints  # Since this is POC code, we are not doing any data validation.
        self.x_engine_clients: List[aiokatcp.Client] = []
        self.sensor_mirrors: List[SensorMirror] = []
        super(Corr3Servlet, self).__init__(host=host, port=port, **kwargs)
        logging.info(f'Corr3Servlet "{self.name}" created.')

        self.sensors.add(
            aiokatcp.Sensor(
                DeviceStatus,
                "device-status",
                "Devices status of the Corr3Servlet",
                default=DeviceStatus.OK,
                status_func=device_status_to_sensor_status,
            )
        )

    async def start(self):
        """Do the usual startup stuff plus initiating connections to DSP nodes.
        
        The __init__ function (and in general other "magic" Python funtions") cannot handle asynchronous stuff.
        This start() function is called manually by the user after creating the object, usually by some main
        function which runs in a `loop.run_until_complete(main())` kind of way.

        I thought it would be simplest to tack the functionality of starting the member katcp clients onto the
        `start()` function, which is going to run anyway, rather than having a separate one which the user would
        need to remember to run. This way a Corr3Servlet looks very much like a normal DeviceServer.
        """
        # Call the parent start() function
        await super(Corr3Servlet, self).start()
        logging.info(f'Corr3Servlet "{self.name}" started.')
        # Pin on a little bit of functionality of our own...
        for n, (host, port) in enumerate(self.x_engine_endpoints):
            client = aiokatcp.Client(host=host, port=port)
            sensor_mirror = SensorMirror(client, self, f"node{n}")  # More thought needs to be given to naming.
            client.add_sensor_watcher(sensor_mirror)
            self.x_engine_clients.append(client)
            self.sensor_mirrors.append(sensor_mirror)

    async def on_stop(self):
        """Tidy up the client connections before quitting completely."""
        for client in self.x_engine_clients:
            client.close()
            await client.wait_closed()

    async def request_beam_weights(self, ctx, data_stream, *weights):
        """Load weights for all inputs on a specified beam data-stream."""
        if len(weights) != self.n_antennas:
            raise aiokatcp.connection.FailReply(f"{len(weights)} weights received, expected {self.n_antennas}")

        # For the time being, we are assuming that the B-engines understand pretty much the same request.
        for client_no, client in enumerate(self.x_engine_clients):
            logging.debug(f"Forwarding the ?beam-weight message to client no {client_no}")
            _reply, _informs = await client.request("beam-weights", data_stream, *weights)

        # TODO: The ICD says "the explanation describes the current weights applied to the inputs of a specific beam".
        #      I should probably figure out what that looks like, and return appropriately.
        #      The test should probably also assert this.
        return "Beam weights set correctly."
