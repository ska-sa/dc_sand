"""Example of how to start a ProcessController."""
import asyncio
from ngkcs.process_controller import ProcessController

LOCALHOST = "127.0.0.1"
DEFAULT_PORT = 5678


async def main():
    """Start a ProcessController instance asynchronously."""
    process_controller_inst = ProcessController(
        name="test", processor_endpoints=[(LOCALHOST, 5680)], host=LOCALHOST, port=DEFAULT_PORT
    )

    await process_controller_inst.start()
    await process_controller_inst.join()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
