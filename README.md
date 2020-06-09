# dc_sand
DSP team's CUDA sandbox.
For edification and entertainment.

An example can be found in the "example" subfolder. To compile, simply cd into the directory and "make", and the resulting executable will produce an output.

Execute doxygen in this directory in order to produce HTML-browseable documentation about each class.

## Python pre-commit Workflow
This repo includes configuration files for a Python pre-commit workflow that does the following when attempting to commit Python files to the repo:
* auto-formats all .py and .pyi files
* checks all .py and .pyi files for compliance to PEP8 style and standards

Commits will only succeed if all checks are passed.

Contributors to this repo are encouraged to install and use this this workflow.

### Setup
#### Requirements
[pre-commit]([https://pre-commit.com/](https://pre-commit.com/)) is required to install the pre-commit hooks. To install these development requirments, run:

`pip install -r requirements-dev.txt`

#### Installing pre-commit hooks
Then, to install the pre-commit hooks, run the following inside the project repo:

`pre-commit install`

All Python source files staged for commit will now be checked upon running `git commit`.
