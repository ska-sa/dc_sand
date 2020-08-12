"""Control of a single subarray product."""

import asyncio
import logging
import time
import docker

from typing import Dict, Set, List, Optional

import aiokatcp
from aiokatcp import FailReply, Sensor, Address

from enum import Enum
from ipaddress import IPv4Address

LOCALHOST = "127.0.0.1"  # Unlike 'localhost', guaranteed to be IPv4
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


class ProductState(OrderedEnum):
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


def _error_on_error(state: ProductState) -> Sensor.Status:
    """Status function callback method for aiokatcp.Sensor."""
    return Sensor.Status.ERROR if state == ProductState.ERROR else Sensor.Status.NOMINAL


# endregion


class CBFSubarrayProductBase:
    """CBF Subarray Product Base.

    Represents an instance of an CBF Subarray product. This includes ingest, an
    appropriate telescope model, and any required post-processing.

    In general each telescope subarray product is handled in a completely
    parallel fashion by the CBF. This class encapsulates these instances,
    handling control input and sensor feedback to CAM.

    State changes are asynchronous operations. There can only be one
    asynchronous operation at a time. Attempting a second one will either
    fail, or in some cases will cancel the prior operation. To avoid race
    conditions, changes to :attr:`state` should generally only be made from
    inside the asynchronous tasks.

    This is a base class that is intended to be subclassed. The methods whose
    names end in ``_impl`` are extension points that should be implemented in
    subclasses to do the real work. These methods are run as part of the
    asynchronous operations. They need to be cancellation-safe, to allow for
    forced deconfiguration to abort them.
    """

    def __init__(self, subarray_product_id: str, config: dict, product_controller: aiokatcp.DeviceServer):
        """
        Initialise the Base class with the minimum-required arguments.
        
        Parameters
        ----------
        subarray_product_id : str
            Name of the subarray product
        config : dict
            Configuration data as a (python) dictionary
        product_controller : aiokatcp.DeviceServer
            The parent server that interfaces with CAM
        """
        self._async_task: Optional[asyncio.Task] = None  #: Current background task (can only be one)
        self.docker_client = docker.from_env()

        self.config = config
        self.subarray_product_id = subarray_product_id
        self.product_controller = product_controller
        self.dead_event = asyncio.Event()  # Set when reached state DEAD

        # Callbacks that are called when we reach state DEAD. These are
        # provided in addition to dead_event, because sometimes it's
        # necessary to react immediately rather than waiting for next time
        # around the event loop. Each callback takes self as the argument.
        self.dead_callbacks = [lambda product: product.dead_event.set()]
        self._state: ProductState = ProductState.CONFIGURING

        # Set of sensors to remove when the product is removed
        self.sensors: Set[aiokatcp.Sensor] = set()
        self._state_sensor = Sensor(
            ProductState,
            "state",
            "State of the subarray product state machine (prometheus: gauge)",
            status_func=_error_on_error,
        )
        self._device_status_sensor = product_controller.sensors["device-status"]

        self.state = ProductState.CONFIGURING  # This sets the sensor
        self.add_sensor(self._state_sensor)
        logger.info("Created: %r", self)

    @property
    def state(self) -> ProductState:
        """Property of the CBFSubarrayProduct."""
        return self._state

    @state.setter
    def state(self, value: ProductState) -> None:
        """State-setter method for the CBFSubarrayProduct."""
        if self._state == ProductState.ERROR and value not in (ProductState.DECONFIGURING, ProductState.DEAD):
            return  # Never leave error state other than by deconfiguring
        now = time.time()
        if value == ProductState.ERROR and self._state != value:
            self._device_status_sensor.set_value(DeviceStatus.FAIL, timestamp=now)
        self._state = value
        self._state_sensor.set_value(value, timestamp=now)

    def add_sensor(self, sensor: Sensor) -> None:
        """Add the supplied sensor to the top-level device and track it locally."""
        self.sensors.add(sensor)
        self.product_controller.sensors.add(sensor)

    def remove_sensors(self):
        """Remove all sensors added via :meth:`add_sensor`.

        It does *not* send an ``interface-changed`` inform; that is left to the
        caller.
        """
        for sensor in self.sensors:
            self.product_controller.sensors.discard(sensor)
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
                "Please wait for it to complete first.".format(self.subarray_product_id)
            )

    async def configure_impl(self) -> None:
        """Extension point to configure the subarray."""
        pass

    async def deconfigure_impl(self, force: bool, ready: asyncio.Event = None) -> None:
        """Extension point to deconfigure the subarray.

        Parameters
        ----------
        force
            Whether to do an abrupt deconfiguration without waiting for
            postprocessing.
        ready
            If the ?product-deconfigure command should return before
            deconfiguration is complete, this event can be set at that point.
        """
        pass

    async def _configure(self) -> None:
        """Asynchronous task that does the configuration."""
        await self.configure_impl()
        self.state = ProductState.IDLE

    async def _deconfigure(self, force: bool, ready: asyncio.Event = None) -> None:
        """Asynchronous task that does the deconfiguration."""
        self.state = ProductState.DECONFIGURING
        await self.deconfigure_impl(force=force, ready=ready)

        self.state = ProductState.DEAD

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
        assert self.state == ProductState.CONFIGURING, "configure should be the first thing to happen"
        task = asyncio.get_event_loop().create_task(self._configure())
        self._async_task = task
        try:
            await task
        finally:
            self._clear_async_task(task)
        logger.info("Subarray product %s successfully configured", self.subarray_product_id)

    async def deconfigure(self, force: bool = False, ready: asyncio.Event = None):
        """Deconfigure command to satisfy the parent DeviceServer on_stop() command."""
        # Obviously need to call _deconfigure, which further calls deconfigure_impl
        if self.state == ProductState.DEAD:
            # Deed has already been done
            return

        task = asyncio.get_event_loop().create_task(self._deconfigure(force, ready))
        self._async_task = task
        try:
            await task
        finally:
            self._clear_async_task(task)
        logger.info("Subarray product %s successfully deconfigured", self.subarray_product_id)

        return True

        # if self.async_busy:
        #     if not force:
        #         self._fail_if_busy()
        #     else:
        #         logger.warning('Subarray product %s is busy with an operation, '
        #                        'but deconfiguring anyway', self.subarray_product_id)

        # ready = asyncio.Event()
        # task = asyncio.get_event_loop().create_task(self._deconfigure(force, ready))

    def __repr__(self) -> str:
        """Format string representation when the object is queried."""
        return "Subarray product {} (State: {})".format(self.subarray_product_id, self.state.name)


class InterfaceModeSensors:
    """aiokatcp.Sensors created for interfacing with a CBFSubarrayProductInterface instance."""

    def __init__(self, subarray_product_id: str):
        """Manage dummy subarray product sensors on a DeviceServer instance.

        Parameters
        ----------
        subarray_product_id
            Subarray product id, e.g. `array_1_c856M4k`
        """
        self.subarray_product_id = subarray_product_id
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


class CBFSubarrayProductInterface(CBFSubarrayProductBase):
    """Dummy implementation of CBFSubarrayProductBase interface that does not actually run anything."""

    def __init__(self, *args, **kwargs):
        """Create object in interface-mode."""
        super().__init__(*args, **kwargs)
        self._interface_mode_sensors = InterfaceModeSensors(self.subarray_product_id)
        sensors = self._interface_mode_sensors.sensors
        self._capture_block_states = [
            sensor for sensor in sensors.values() if sensor.name.endswith(".capture-block-state")
        ]

    async def configure_impl(self) -> None:
        """Update parent DeviceServer with Interface-mode Sensors."""
        logger.warning("No components will be started - running in interface mode")
        # Add dummy sensors for this product
        self._interface_mode_sensors.add_sensors(self.product_controller)
        # Start an example docker image in detached mode
        # - Will need to pass through a name and config details (eventually)
        # - This example constantly logs the message 'Reticulating Spine {i}', where 'i' increments
        self.container = self.docker_client.containers.run("bfirsh/reticulate-splines", detach=True)

    async def deconfigure_impl(self, force: bool, ready: asyncio.Event = None) -> None:
        """Initiate Deconfigure in Interface-mode."""
        self._interface_mode_sensors.remove_sensors(self.product_controller)
        # Stop the docker container
        # - Unfortunately the container.status doesn't really change from 'created'
        self.container.stop()
