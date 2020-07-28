"""
A Corr3 proto-servlet.

A Katcp device server whose purpose in life is to interpret between old-CAM and NGC.
At the moment this consists only of forwarding a beam-weights request from a client to
a bunch of other DeviceServers, which represent hypothetical processing nodes.
"""

import aiokatcp
from typing import List, Tuple


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
        super(Corr3Servlet, self).__init__(host=host, port=port, **kwargs)

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
        # Pin on a little bit of functionality of our own...
        for host, port in self.x_engine_endpoints:
            # So it's worth noting that this is not the most efficient way to run this particular loop. The connections do not
            # happen concurrently, but one after the other. For this simple, small use-case, it's fine. But if we end up having
            # dozens of processing nodes, we may have to optimise this somewhat.
            client = await aiokatcp.Client.connect(host=host, port=port)
            self.x_engine_clients.append(client)

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
            print(f"Forwarding the ?beam-weight message to client no {client_no}")
            _reply, _informs = await client.request("beam-weights", data_stream, *weights)

        # TODO: The ICD says "the explanation describes the current weights applied to the inputs of a specific beam".
        #      I should probably figure out what that looks like, and return appropriately.
        #      The test should probably also assert this.
        return "Beam weights set correctly."
