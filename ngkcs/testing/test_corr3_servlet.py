import asyncio
import pytest
import pytest_asyncio
from ngkcs.corr3_servlet import Corr3Servlet
import aiokatcp

CORR3_SERVLET_PORT = 7404

@pytest.fixture(scope="function")
async def corr3_servlet():
    """Provide an instance of Corr3Servlet to test.

    This fixture readies a Corr3Servlet instance in a running state, that can be queried and the results
    compared with the desired output to test for correctness of operation.

    At the moment, the parameters of the Corr3Servlet are hard-coded. Ideally we'd like to parameterise this,
    but I'm not quite sure how to do that yet.

    I'd also like to scope the fixture for the class instead of the function. It makes more sense to me, rather than
    tearing down and rebuilding the servlet for every single little test. I'll defer to those with more wisdom if this
    is not a good idea. The main obstacle at the moment is that this involves changing the scope of the default
    `event_loop` fixture as well. I'm not sure how to do that yet either.
    """
    servlet = Corr3Servlet(name="bob",
                           n_antennas=4,
                           n_channels=4096,
                           host="0.0.0.0",
                           port=CORR3_SERVLET_PORT)
    await servlet.start()
    # Using `yield` in a fixture allows execution to come back to the fixture once the test is done with it, so that
    # necessary cleanup can happen afterwards.
    yield servlet
    await servlet.stop()
    await servlet.join()


async def make_a_request(*args):
    """Connect to the servlet and send a request."""
    client = await aiokatcp.Client.connect('localhost', CORR3_SERVLET_PORT)
    try:
        # We put this inside a `try/finally` block in order to make sure that everything gets cleaned up properly.
        # In some tests, we may expect an exception from deliberately sending a bad request, so we want to make sure
        # that we don't leave the client dangling.
        reply, informs = await client.request(*args)
    finally:
        client.close()
        await client.wait_closed()
    #TODO: In principle, we could `return` the reply, or parse it or something. I'm not quite sure what we need.


class TestCorr3Servlet:
    """Class for grouping Corr3Servlet tests."""
    def test_beam_weights(self, corr3_servlet, event_loop):
        """Test a properly-formed ?beam-weights request."""
        reply = event_loop.run_until_complete(make_a_request("beam-weights",
                                                             "tied-array-channelised-voltage",
                                                             "1", "2", "3", "4"))
        #TODO: test whether the servlet passes the info on to the individual engines.
        #      This will require more infrastructure from the test fixture I think. Or another fixture.


    def test_beam_weights_wrong_numbers(self, corr3_servlet, event_loop):
        """Test a ?beam-weights request with the wrong number of weights being passed."""
        # This context manager tells `pytest` that we expect an exception to be raised.
        with pytest.raises(aiokatcp.connection.FailReply):
            reply = event_loop.run_until_complete(make_a_request("beam-weights",
                                                                 "tied-array-channelised-voltage",
                                                                 "1", "2", "3")) # Only 3 weights, 4 expected.


    def test_beam_weights_wrong_stream(self, corr3_servlet, event_loop):
        """Test a ?beam-weights request with the wrong data-stream being passed."""
        reply = event_loop.run_until_complete(make_a_request("beam-weights",
                                                             "baseline-correlation-products", # i.e. not t-a-c-v.
                                                             "1", "2", "3", "4"))
