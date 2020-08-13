"""
Example of how to start a corr3 servlet.

This is very rough-and-ready, proof-of-concept kind of code. Don't deploy into production.
Dragons will appear from the void and breathe fire all over your baby seals.
"""
import asyncio
import logging
from ngkcs.corr3_servlet import Corr3Servlet


async def main():
    """Execute the program asynchronously if you would be so kind, good sir."""
    servlet = Corr3Servlet(
        name="stephen",  # for historical reasons
        n_antennas=4,
        x_engine_endpoints=[("localhost", 1234), ("localhost", 1235), ("localhost", 1236), ("localhost", 1237)],
        host="0.0.0.0",
        port=7404,
    )
    await servlet.start()
    await servlet.join()  # Turns out it's as simple as that.


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
