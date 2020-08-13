"""
A first attempt at a 'product controller' - renamed here as ProcessController.

This Controller is based on an aiokatcp.DeviceServer that is operated via katcp requests.
The main reason for this implementation is to launch and interact with multiple nodes, i.e. DataProcessors.
These individual nodes are responsible for their own docker containers, which will eventually house GPU-based DSP.
"""

import aiokatcp
import logging
import os
from typing import (
    List,
    Tuple,
)
from ngkcs.data_processor import (
    DeviceStatus,
    device_status_to_sensor_status,
)

LOCALHOST = "127.0.0.1"
DEFAULT_PORT = 5678
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class ProcessController(aiokatcp.DeviceServer):
    """The main point of contact when attempting to control multiple processing nodes.

    That is, this Process Controller will create one or more DataProcessor entities.
    """

    VERSION = "process_controller-0.1"
    BUILD_STATE = "process_controller-0.1.0"

    def __init__(
        self,
        *,  # Forces all arguments to be named
        name: str,
        processor_endpoints: List[Tuple[str, int]],
        host: str = LOCALHOST,
        port: int = DEFAULT_PORT,
        **kwargs,
    ):
        """Create an instance of the ProcessController.

        Parameters
        ----------
        name
            The name of this ProcessController.
        processor_endpoints
            The physical processing nodes that will be running as DataProcessors.
            Passed as a list of tuples as per Corr3Servlet's design - [(host1, port1), (host2, port2)].
        host
            The interface that listens for katcp client connections, e.g. via telnet.
        port
            The port on which to listen for katcp connections, e.g. via telnet.
        """
        self.name = name.lower()
        self.num_endpoints = len(processor_endpoints)
        self.proc_endpoints = processor_endpoints
        self.processor_clients: List[aiokatcp.Client] = []

        super(ProcessController, self).__init__(host=host, port=port, **kwargs)

        logger.info(f"Created ProcessController: {self.name}")

        self.sensors.add(
            aiokatcp.Sensor(
                DeviceStatus,
                "device-status",
                "Devices status of the ProcessController",
                default=DeviceStatus.OK,
                status_func=device_status_to_sensor_status,
            )
        )

    async def start(self):
        """Additional functionality to the usual startup routine of the DeviceServer."""
        # Call the parent start() function
        await super(ProcessController, self).start()
        logger.info(f"Started ProcessController {self.name}.")

        for n, (host, port) in enumerate(self.proc_endpoints):
            this_client = aiokatcp.Client(host=host, port=port)
            self.processor_clients.append(this_client)

    async def on_stop(self):
        """Additional functionality upon receiving a ?halt request."""
        for this_client in self.processor_clients:
            # Ensure all the clients are halted
            _reply, _informs = await this_client.request("halt")
            this_client.close()
            await this_client.wait_closed()

    async def request_configure_processors(self, ctx, name: str, config_filename: str):
        """Configure command to ready the DataProcessors for docker container instantiation.
        
        Parameters
        ----------
        name
            The name of this configuration mode, e.g. array0-bc8n856M1k.
        config_filename
            The configuration file that would contain e.g. GPU host info.
            Defaulting to the corr2 config-file format for now.
        """
        abs_path = None  # The absolute path of the config-file

        try:
            # Check if the config-file exists
            abs_path = os.path.abspath(config_filename)
            if not os.path.exists(abs_path):
                # Problem
                errmsg = "Config-file {} is not valid".format(config_filename)
                logger.error(errmsg)
                raise aiokatcp.FailReply(errmsg)
            # else: Continue!

            for client_counter, this_client in enumerate(self.processor_clients):
                this_processor_name = f"{self.name}-node{client_counter}"  # For now
                logger.debug(f"Configuring DataProcessor: {this_processor_name}")
                # This is the blocking method
                _reply, _informs = await this_client.request("configure", this_processor_name, abs_path)
                # Might be able to do something like:
                # - configure_tasks = [client.request("", ...) for client in self.processor_clients]
                # - asyncio.wait(configure_tasks)

        except Exception as exc:
            retmsg = f"Failed to process config: {exc}"
            logger.error(retmsg)
            raise aiokatcp.FailReply(retmsg)

    async def request_deconfigure_processors(self, ctx, force: bool = False):
        """Deconfigure command to tear down Data Processors and their respective docker containers."""
        for this_client in self.processor_clients:
            logger.debug(f"Deconfiguring DataProcessor @ {this_client.host}:{this_client.port}")
            _reply, _informs = await this_client.request("deconfigure", force)
