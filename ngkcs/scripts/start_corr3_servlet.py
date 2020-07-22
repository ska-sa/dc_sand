"""
Example of how to start a corr3 servlet.
This is very rough-and-ready, proof-of-concept kind of code. Don't deploy into production.
Dragons will appear from the void and breathe fire all over your baby seals.
"""
import asyncio
from ngkcs.corr3_servlet import Corr3Servlet

async def main():
    servlet = Corr3Servlet(name="stephen", # for historical reasons
                           n_antennas=4,
                           n_channels=4096,
                           host="0.0.0.0",
                           port=7404)
    await servlet.start()
    yield
    await servlet.stop()
    await servlet.join()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
