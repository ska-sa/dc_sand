This module is intended as a proof-of-concept to demonstrate that it is possible to roll out the
next-generation correlator (NGC) for MeerKAT (and MeerKAT+) without modifying the interface between
the correlator and the Control and Monitoring (CAM) subsystems.

Currently implemented:
----------------------
- Corr3Servlet
    - Parsing of incoming KATCP requests (as though by old-CAM) and forwarding (possibly modified) onwards to
      hypothetical processing nodes.
    - Mirroring (and name modification) of the sensors on those hypothetical nodes onto a single server.
    - For convenience and testing, a FakeNode class is included in the `testing` directory.

Prerequisite knowledge:
-----------------------
Someone attempting to work on this code should at least be familiar with `aiokatcp`.
A handy tutorial can be found here:
https://aiokatcp.readthedocs.io/en/latest/server/tutorial.html


For developing:
---------------

If you want to develop with this module, do it in a virtual environment.
Make sure that you've got your venv active, cd into the directory with `setup.py` in it, and execute
```
pip install -e .
```

Then you can cd into the `testing` direcory for example, or anywhere else, and you'll be able to use the module
by such means as `from ngkcs.corr3_servlet import Corr3Servlet` (see test_corr3_servlet.py as an example).

Unfortunately, due to the way that Python just is, there's not a clean way to develop using the module without having
it installed in some way. (You can leave out the `-e` flag, but then you'll need to re-install every time you
make a change.)

Unit tests:
-----------
This module is (ostensibly) ready for unit-testing with `pytest`. Executing `pytest` in this direcory will run all
the tests in the `testing` directory. A TDD-style approach was followed, so if everything passes, you should be good
to go.
