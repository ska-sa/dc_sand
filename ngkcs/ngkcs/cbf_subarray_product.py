"""Control of a single subarray product."""

import asyncio
import logging
import json
import time
import docker

from typing import Dict, Set, List, Callable, Optional

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


class CaptureBlockState(OrderedEnum):
    """State of a single capture block."""

    INITIALISING = 0  # Only occurs briefly on construction
    CAPTURING = 1  # capture-init called, capture-done not yet called
    BURNDOWN = 2  # capture-done returned, but real-time processing still happening
    POSTPROCESSING = 3  # real-time processing complete, running batch processing
    DEAD = 4  # fully complete


class CaptureBlock:
    """
    A capture block is book-ended by a capture-init and a capture-done.

    However, note that processing on it continues after the capture-done.
    """

    def __init__(self, name: str, config: dict):
        """Init method for the CaptureBlock."""
        self.name = name
        self.config = config
        self._state = CaptureBlockState.INITIALISING
        self.postprocess_task: Optional[asyncio.Task] = None
        self.dead_event = asyncio.Event()
        self.state_change_callback: Optional[Callable[[], None]] = None
        # Time each state is reached
        self.state_time: Dict[CaptureBlockState, float] = {}

    @property
    def state(self) -> CaptureBlockState:
        """Property of the CaptureBlock."""
        return self._state

    @state.setter
    def state(self, value: CaptureBlockState) -> None:
        """State-setter method for the CaptureBlock."""
        if self._state != value:
            self._state = value
            if value not in self.state_time:
                self.state_time[value] = time.time()
            if value == CaptureBlockState.DEAD:
                self.dead_event.set()
            if self.state_change_callback is not None:
                self.state_change_callback()


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

    There are some invariants that must hold at yield points:
    - There is at most one capture block in state CAPTURING.
    - :attr:`current_capture_block` is the capture block in state
      CAPTURING, or ``None`` if there isn't one.
    - :attr:`current_capture_block` is set if and only if the subarray state
      is CAPTURING.
    - Elements of :attr:`capture_blocks` are not in state DEAD.

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
        self.capture_blocks: Dict[str, CaptureBlock] = {}  # live capture blocks, indexed by name
        # set between capture_init and capture_done
        self.current_capture_block: Optional[CaptureBlock] = None
        self.dead_event = asyncio.Event()  # Set when reached state DEAD

        # Callbacks that are called when we reach state DEAD. These are
        # provided in addition to dead_event, because sometimes it's
        # necessary to react immediately rather than waiting for next time
        # around the event loop. Each callback takes self as the argument.
        self.dead_callbacks = [lambda product: product.dead_event.set()]
        self._state: ProductState = ProductState.CONFIGURING

        # Set of sensors to remove when the product is removed
        self.sensors: Set[aiokatcp.Sensor] = set()
        self._capture_block_sensor = Sensor(
            str,
            "capture-block-state",
            "JSON dictionary of capture block states for active capture blocks",
            default="{}",
            initial_status=Sensor.Status.NOMINAL,
        )
        self._state_sensor = Sensor(
            ProductState,
            "state",
            "State of the subarray product state machine (prometheus: gauge)",
            status_func=_error_on_error,
        )
        self._device_status_sensor = product_controller.sensors["device-status"]

        self.state = ProductState.CONFIGURING  # This sets the sensor
        self.add_sensor(self._capture_block_sensor)
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

    async def capture_init_impl(self, capture_block: CaptureBlock) -> None:
        """Extension point to start a capture block.

        If it raises an exception, the capture block is assumed to not have
        been started, and the subarray product goes into state ERROR.
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

    def _capture_block_dead(self, capture_block: CaptureBlock) -> None:
        """Mark a capture block as dead and remove it from the list."""
        try:
            del self.capture_blocks[capture_block.name]
        except KeyError:
            pass  # Allows this function to be called twice
        # Setting the state will trigger _update_capture_block_sensor, which
        # will update the sensor with the value removed
        capture_block.state = CaptureBlockState.DEAD

    def _update_capture_block_sensor(self) -> None:
        """Update aiokatcp.Sensor.value for all CaptureBlocks.
        
        Writes value as a JSON-string.
        """
        value = {name: capture_block.state.name.lower() for name, capture_block in self.capture_blocks.items()}
        self._capture_block_sensor.set_value(json.dumps(value, sort_keys=True))

    async def _capture_init(self, capture_block: CaptureBlock) -> None:
        self.capture_blocks[capture_block.name] = capture_block
        capture_block.state_change_callback = self._update_capture_block_sensor
        # Update the sensor with the INITIALISING state
        self._update_capture_block_sensor()
        try:
            await self.capture_init_impl(capture_block)
            if self.state == ProductState.ERROR:
                raise FailReply("Subarray product went into ERROR while starting capture")
        except asyncio.CancelledError:
            self._capture_block_dead(capture_block)
            raise
        except Exception:
            self.state = ProductState.ERROR
            self._capture_block_dead(capture_block)
            raise
        assert self.current_capture_block is None
        self.state = ProductState.CAPTURING
        self.current_capture_block = capture_block
        capture_block.state = CaptureBlockState.CAPTURING

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

    async def capture_init(self, capture_block_id: str, config: dict) -> str:
        """Initiate the data-capture sequence for this CBFSubarrayProduct."""
        self._fail_if_busy()
        if self.state != ProductState.IDLE:
            raise FailReply(
                "Subarray product {} is currently in state {}, not IDLE as expected. "
                "Cannot be inited.".format(self.subarray_product_id, self.state.name)
            )
        logger.info("Using capture block ID %s", capture_block_id)

        capture_block = CaptureBlock(capture_block_id, config)
        task = asyncio.get_event_loop().create_task(self._capture_init(capture_block))
        self._async_task = task
        try:
            await task
        finally:
            self._clear_async_task(task)
        logger.info("Started capture block %s on subarray product %s", capture_block_id, self.subarray_product_id)
        return capture_block_id

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

    def _update_capture_block_state(self, capture_block_id: str, state: Optional[CaptureBlockState]) -> None:
        """Update the simulated *.capture-block-state sensors.

        The dictionary that is JSON-encoded in the sensor value is updated to
        set the value associated with the key `capture_block_id`. If `state` is
        `None`, the key is removed instead.
        """
        for name, sensor in self._interface_mode_sensors.sensors.items():
            if name.endswith(".capture-block-state"):
                states = json.loads(sensor.value)
                if state is None:
                    states.pop(capture_block_id, None)
                else:
                    states[capture_block_id] = state.name.lower()
                sensor.set_value(json.dumps(states))

    async def configure_impl(self) -> None:
        """Update parent DeviceServer with Interface-mode Sensors."""
        logger.warning("No components will be started - running in interface mode")
        # Add dummy sensors for this product
        self._interface_mode_sensors.add_sensors(self.product_controller)
        # Start an example docker image in detached mode
        # - Will need to pass through a name and config details (eventually)
        # - This example constantly logs the message 'Reticulating Spine {i}', where 'i' increments
        self.container = self.docker_client.containers.run("bfirsh/reticulate-splines", detach=True)

    async def capture_init_impl(self, capture_block: CaptureBlock) -> None:
        """Update CaptureBlock in Interface-mode."""
        self._update_capture_block_state(capture_block.name, CaptureBlockState.CAPTURING)

    async def deconfigure_impl(self, force: bool, ready: asyncio.Event = None) -> None:
        """Initiate Deconfigure in Interface-mode."""
        self._interface_mode_sensors.remove_sensors(self.product_controller)
        # Stop the docker container
        # - Unfortunately the container.status doesn't really change from 'created'
        self.container.stop()
