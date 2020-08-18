"""Control of a single subarray product."""

import asyncio
from configparser import ConfigParser
import logging
import time
import docker
import sys

from typing import Dict, List, Optional, Set

import aiokatcp
from aiokatcp import FailReply, Sensor, Address

from enum import Enum
from ipaddress import IPv4Address

LOCALHOST = "127.0.0.1"  # Unlike 'localhost', guaranteed to be IPv4
DEFAULT_PORT = 5680
logger = logging.getLogger(__name__)


# region From katsdpcontroller


class OrderedEnum(Enum):
    """Ordered enumeration from Python 3.x Enum documentation."""

    def __ge__(self, other):
        """Greater-than or Equal-to comparison."""
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        """Greater-than comparison."""
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        """Less-than or Equal-to comparison."""
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        """Less-than comparison."""
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class ProcessorState(OrderedEnum):
    """State of a subarray.

    Only the following transitions can occur:
    - CONFIGURING -> IDLE (via product-configure)
    - IDLE -> CAPTURING (via capture-init)
    - CAPTURING -> IDLE (via capture-done)
    - CONFIGURING/IDLE/CAPTURING/ERROR -> DECONFIGURING -> POSTPROCESSING -> DEAD
      (via product-deconfigure)
    - IDLE/CAPTURING/DECONFIGURING/POSTPROCESSING -> ERROR (via an internal error)
    """

    CONFIGURING = 0
    IDLE = 1
    CAPTURING = 2
    DECONFIGURING = 3
    DEAD = 4
    ERROR = 5
    POSTPROCESSING = 6


class DeviceStatus(OrderedEnum):
    """DeviceStatus Enum object."""

    OK = 1
    DEGRADED = 2
    FAIL = 3


def device_status_to_sensor_status(status: DeviceStatus) -> Sensor.Status:
    """Map DeviceStatus discrete Enum to aiokatcp.Sensor Statuses."""
    mapping = {
        DeviceStatus.OK: Sensor.Status.NOMINAL,
        DeviceStatus.DEGRADED: Sensor.Status.WARN,
        DeviceStatus.FAIL: Sensor.Status.ERROR,
    }
    return mapping[status]


def _error_on_error(state: ProcessorState) -> Sensor.Status:
    """Status function callback method for aiokatcp.Sensor."""
    return Sensor.Status.ERROR if state == ProcessorState.ERROR else Sensor.Status.NOMINAL


# endregion


class DataProcessorBase:
    """Data Processor Base.

    Represents an instance of a CBF Data Processor.

    In general each Data Processor is handled in a completely parallel fashion by the CBF.
    This class encapsulates these instances, handling control input and sensor feedback to CAM.

    This is a base class that is intended to be subclassed. The methods whose
    names end in ``_impl`` are extension points that should be implemented in
    subclasses to do the real work. These methods are run as part of the
    asynchronous operations.
    """

    def __init__(self, data_proc_id: str, config: dict, proc_controller: aiokatcp.DeviceServer):
        """
        Initialise the Base class with the minimum-required arguments.

        Parameters
        ----------
        data_proc_id : str
            Name of the data processor
        config : dict
            Configuration data as a (python) dictionary
        proc_controller : aiokatcp.DeviceServer
            The parent server that interfaces with CAM
        """
        self._async_task: Optional[asyncio.Task] = None  #: Current background task (can only be one)
        self.docker_client = docker.from_env()

        self.config = config
        self.data_proc_id = data_proc_id
        self.proc_controller = proc_controller

        self._state: ProcessorState = ProcessorState.CONFIGURING

        # Set of sensors to remove when the product is removed
        self.sensors: Set[aiokatcp.Sensor] = set()
        self._state_sensor = Sensor(
            ProcessorState, "state", "State of the data processor state machine", status_func=_error_on_error,
        )
        self._device_status_sensor = proc_controller.sensors["device-status"]

        self.state = ProcessorState.CONFIGURING  # This sets the sensor
        self.add_sensor(self._state_sensor)
        logger.info("Created: %r", self)

    @property
    def state(self) -> ProcessorState:
        """Property of the DataProcessor."""
        return self._state

    @state.setter
    def state(self, value: ProcessorState) -> None:
        """State-setter method for the DataProcessor."""
        if self._state == ProcessorState.ERROR and value not in (ProcessorState.DECONFIGURING, ProcessorState.DEAD):
            return  # Never leave error state other than by deconfiguring
        now = time.time()
        if value == ProcessorState.ERROR and self._state != value:
            self._device_status_sensor.set_value(DeviceStatus.FAIL, timestamp=now)
        self._state = value
        self._state_sensor.set_value(value, timestamp=now)

    def add_sensor(self, sensor: Sensor) -> None:
        """Add the supplied sensor to the top-level device and track it locally."""
        self.sensors.add(sensor)
        self.proc_controller.sensors.add(sensor)

    def remove_sensors(self):
        """Remove all sensors added via :meth:`add_sensor`.

        It does *not* send an ``interface-changed`` inform; that is left to the
        caller.
        """
        for sensor in self.sensors:
            self.proc_controller.sensors.discard(sensor)
        self.sensors.clear()

    @property
    def async_busy(self) -> bool:
        """Whether there is an asynchronous state-change operation in progress."""
        return self._async_task is not None and not self._async_task.done()

    def _fail_if_busy(self) -> None:
        """Raise a FailReply if there is an asynchronous operation in progress."""
        if self.async_busy:
            raise FailReply(
                "Subarray product {} is busy with an operation. "
                "Please wait for it to complete first.".format(self.data_proc_id)
            )

    async def configure_impl(self) -> None:
        """Extension point to configure the DataProcessor."""
        pass

    async def deconfigure_impl(self, force: bool) -> None:
        """Extension point to deconfigure the DataProcessor.

        Parameters
        ----------
        force
            Force deconfiguration, regardless of state.
        """
        pass

    async def _configure(self) -> None:
        """Asynchronous task that does the configuration."""
        await self.configure_impl()
        self.state = ProcessorState.IDLE

    async def _deconfigure(self, force: bool) -> None:
        """Asynchronous task that does the deconfiguration."""
        self.state = ProcessorState.DECONFIGURING
        await self.deconfigure_impl(force=force)

        self.state = ProcessorState.DEAD

    def _clear_async_task(self, future: asyncio.Task) -> None:
        """Clear the current async task.

        Parameters
        ----------
        future
            The expected value of :attr:`_async_task`. If it does not match,
            it is not cleared (this can happen if another task replaced it
            already).
        """
        if self._async_task is future:
            self._async_task = None

    async def configure(self) -> None:
        """Configure method exposed by the Base class."""
        assert not self.async_busy, "configure should be the first thing to happen"
        assert self.state == ProcessorState.CONFIGURING, "configure should be the first thing to happen"
        task = asyncio.get_event_loop().create_task(self._configure())
        self._async_task = task
        try:
            await task
        finally:
            self._clear_async_task(task)
        logger.info("Data Processor %s successfully configured", self.data_proc_id)

    async def deconfigure(self, force: bool = False):
        """Deconfigure method exposed to tear down the DataProcessor."""
        # Obviously need to call _deconfigure, which further calls deconfigure_impl
        if self.state == ProcessorState.DEAD:
            # Deed has already been done
            return

        task = asyncio.get_event_loop().create_task(self._deconfigure(force))
        self._async_task = task
        try:
            await task
        finally:
            self._clear_async_task(task)
        logger.info("Subarray product %s successfully deconfigured", self.data_proc_id)

        return True

    def __repr__(self) -> str:
        """Format string representation when the object is queried."""
        return "Subarray product {} (State: {})".format(self.data_proc_id, self.state.name)


class InterfaceModeSensors:
    """aiokatcp.Sensors created for interfacing with a DataProcessorInterface instance."""

    def __init__(self, data_proc_id: str):
        """Manage dummy data processor sensors on a DeviceServer instance.

        Parameters
        ----------
        data_proc_id
           Data Processor ID, e.g. `array_1_c856M4k`
        """
        self.data_proc_id = data_proc_id
        self.sensors: Dict[str, Sensor] = {}

    def add_sensors(self, server: aiokatcp.DeviceServer) -> None:
        """Add dummy subarray product sensors and issue #interface-changed."""
        interface_sensors: List[Sensor] = [
            Sensor(
                Address,
                "bf_ingest.beamformer.1.port",
                "IP endpoint for port",
                default=Address(IPv4Address("1.2.3.4"), 31048),
                initial_status=Sensor.Status.NOMINAL,
            ),
            Sensor(
                bool,
                "ingest.sdp_l0.1.capture-active",
                "Is there a currently active capture session.",
                default=False,
                initial_status=Sensor.Status.NOMINAL,
            ),
            Sensor(
                str,
                "cal.1.capture-block-state",
                "JSON dict with the state of each capture block",
                default="{}",
                initial_status=Sensor.Status.NOMINAL,
            ),
        ]

        sensors_added = False
        try:
            for sensor in interface_sensors:
                if sensor.name in self.sensors:
                    logger.info("Simulated sensor %r already exists, skipping", sensor.name)
                    continue
                self.sensors[sensor.name] = sensor
                server.sensors.add(sensor)
                sensors_added = True
        finally:
            if sensors_added:
                server.mass_inform("interface-changed", "sensor-list")

    def remove_sensors(self, server: aiokatcp.DeviceServer) -> None:
        """Remove dummy subarray product sensors and issue #interface-changed."""
        sensors_removed = False
        try:
            for sensor_name, sensor in list(self.sensors.items()):
                server.sensors.discard(sensor)
                del self.sensors[sensor_name]
                sensors_removed = True
        finally:
            if sensors_removed:
                server.mass_inform("interface-changed", "sensor-list")


class DataProcessorInterface(DataProcessorBase):
    """Dummy implementation of DataProcessorBase interface that does not actually run anything."""

    def __init__(self, *args, **kwargs):
        """Create object in interface-mode."""
        super().__init__(*args, **kwargs)
        self._interface_mode_sensors = InterfaceModeSensors(self.data_proc_id)
        sensors = self._interface_mode_sensors.sensors
        self._capture_block_states = [
            sensor for sensor in sensors.values() if sensor.name.endswith(".capture-block-state")
        ]

    async def configure_impl(self) -> None:
        """Update parent DeviceServer with Interface-mode Sensors."""
        # Add dummy sensors for this product
        self._interface_mode_sensors.add_sensors(self.proc_controller)
        # Start an example docker image in detached mode
        # - Will need to pass through a name and config details (eventually)
        # - This example constantly logs the message 'Reticulating Spine {i}', where 'i' increments
        self.container = self.docker_client.containers.run("bfirsh/reticulate-splines", detach=True)

    async def deconfigure_impl(self, force: bool) -> None:
        """Initiate Deconfigure in Interface-mode."""
        self._interface_mode_sensors.remove_sensors(self.proc_controller)
        # Stop the docker container
        # - Unfortunately the container.status doesn't really change from 'created'
        self.container.stop()


def parse_config_file(config_filename=""):
    """
    Parse a config file into a dictionary. No checking done at all.

    Placing it here for now - will eventually be moved into a utils.py.

    Parameters
    ----------
    config_filename
        The config file to process

    Returned
    --------
        A dictionary containing the configuration data
    """
    parser = ConfigParser()
    files = parser.read(config_filename)
    if len(files) == 0:
        raise IOError(f"Could not read the config file: {config_filename}")

    config_dict = {}

    for section in parser.sections():
        config_dict[section] = {}
        for items in parser.items(section):
            config_dict[section][items[0]] = items[1]

    return config_dict


class DeviceServer(aiokatcp.DeviceServer):
    """A simple aiokatcp.DeviceServer exposing control of a data processing node."""

    VERSION = "process-controller-0.1"
    BUILD_STATE = "build-state"

    def __init__(
        self, *, host: str = LOCALHOST, port: int = DEFAULT_PORT, **kwargs,
    ):
        """Initialise the DataProcessor DeviceServer with the necessary properties."""
        self.data_processor = None

        super(DeviceServer, self).__init__(host, port=port, **kwargs)

        self.sensors.add(
            Sensor(
                DeviceStatus,
                "device-status",
                "Devices status of the DataProcessor",
                default=DeviceStatus.OK,
                status_func=device_status_to_sensor_status,
            )
        )

    async def on_stop(self) -> None:
        """Add extra clean-up before finally halting the server."""
        if self.data_processor is not None and self.data_processor.state != ProcessorState.DEAD:
            logger.warning("Data Processor interrupted - deconfiguring running Data Processor.")
            try:
                await self.data_processor.deconfigure(force=True)
            except Exception:
                logger.warning("Failed to deconfigure %s during shutdown", exc_info=True)

    async def _client_connected_cb(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Provide debug info concerning new connections that the base doesn't give.

        Taken from testing/fake_node.py.
        """
        old_connections = self._connections.copy()  # Get a set of the old connections.
        await super(DeviceServer, self)._client_connected_cb(reader, writer)  # Let the DeviceServer add the new one.
        new_connection = self._connections.difference(old_connections)  # The new connection will be the only one.
        logger.debug(f"Client connected from {new_connection.pop().address}")

    async def request_configure(self, ctx, data_proc_name: str, config_filename: str) -> None:
        """Configure a data processor instance.

        Parameters
        ----------
        data_proc_name : str
            Name of the data processor.
        config_filename : str
            Traditional corr2 config-file filename, for now
            Should be error-checked before being passed here
        """
        logger.info(f"?configure called with: {ctx.req}")

        config_dict = None

        if self.data_processor is not None and self.data_processor.state != ProcessorState.DEAD:
            raise FailReply("Already configured or configuring")
        try:

            config_dict = parse_config_file(config_filename)
            # Now, pass it on to the actual configure_product command!
            await self._configure_data_processor(name=data_proc_name.lower(), config_dict=config_dict)

        except Exception as exc:
            retmsg = f"Failed to process config: {exc}"
            logger.error(retmsg)
            raise FailReply(retmsg)

    async def _configure_data_processor(self, name: str, config_dict: dict) -> None:
        """Configure a data processor in response to a request."""
        logger.debug(f"Received config data: {config_dict}")

        data_processor = DataProcessorInterface(data_proc_id=name, config=config_dict, proc_controller=self)
        self.data_processor = data_processor

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
    """Asynchronous function that exposes ability to run the data_processor.DeviceServer manually."""
    port = DEFAULT_PORT
    if len(sys.argv) >= 2:  # Crude primitive arg parsing.
        port = sys.argv[1]
    server = DeviceServer(host=LOCALHOST, port=port)
    await server.start()
    await server.join()  # Technically not really needed, as there's no cleanup afterwards as things currently stand.


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
