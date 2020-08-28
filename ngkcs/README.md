This module is intended as a proof-of-concept to demonstrate that it is possible to roll out the
next-generation correlator (NGC) for MeerKAT (and MeerKAT+) without modifying the interface between
the correlator and the Control and Monitoring (CAM) subsystems.

Currently implemented:
----------------------
- Corr3Servlet:
    - Parsing of incoming KATCP requests (as though by old-CAM) and forwarding (possibly modified) onwards to
      hypothetical processing nodes.
    - Mirroring (and name modification) of the sensors on those hypothetical nodes onto a single server.
    - For convenience and testing, a FakeNode class is included in the `testing` directory.
- DataProcessor:
    - An object designed to mimic a physical processing node, e.g. GPU server.
    - Control of the processing node is facilitated via `katcp` messages to its `aiokatcp.DeviceServer`.
    - This currently exposes `configure` and `deconfigure` commands, which create and destroy a docker container.
    - The docker container associated with the DataProcessor instance is a placeholder for the GPU kernel call, to be executed on the processing node.
- ProcessController:
    - Acting in a similar fashion to the `Corr3Servlet`, but in this case controlling the instance(s) of a DataProcessor.
    - Again, facilitates interfacing and control via `katcp` messages to its `aiokatcp.DeviceServer`.

If you wish to use the `start_corr3_servlet.py` or `start_process_controller.py` scripts, please ensure you have started the corresponding child entities (`FakeNode(s)` or `DataProcessor(s)`) before attempting to interact with them via their parent controllers.

Prerequisite knowledge:
-----------------------
Someone attempting to work on this code should at least be familiar with `aiokatcp`.
A handy tutorial can be found [here](https://aiokatcp.readthedocs.io/en/latest/server/tutorial.html)

For developing:
---------------

If you want to develop with this module, do it in a virtual environment.
Make sure that you've got your venv active, cd into the directory with `setup.py` in it, and execute
```
pip install -r requirements.txt
pip install -e .
```
Then you can `cd` into the `testing` direcory for example, or anywhere else, and you'll be able to use the module
by such means as `from ngkcs.corr3_servlet import Corr3Servlet` (see test_corr3_servlet.py as an example).

Furthermore, testing and developing of the Data Processor requires a Docker Engine to be running on your machine - installation instructions can be found [here](https://docs.docker.com/engine/install/ubuntu/)

Unfortunately, due to the way that Python just is, there's not a clean way to develop using the module without having
it installed in some way. (You can leave out the `-e` flag, but then you'll need to re-install every time you
make a change.)

Unit tests:
-----------
This module is (ostensibly) ready for unit-testing with `pytest`. Executing `pytest` in this direcory will run all the tests in the `testing` directory. A TDD-style approach was followed, so if everything passes, you should be good to go.

The file `test_4k_corr.ini` is included in the repository in order to make running the tests a bit easier.
- The file itself isn't much more than a placeholder at the moment.
- It is used mainly to illustrate where and how configuration data might be parsed when starting up a ProcessController and configuring DataProcessors.
