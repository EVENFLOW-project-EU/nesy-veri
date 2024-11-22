## Installation
We use [Poetry](https://python-poetry.org/) for package management. Simply run:
```
poetry install
```
to install all packages. The Python version requirement is >=3.11,<=3.12.


You can then create a virtual environment through:
```
poetry shell
```
in which you can run everything as usual. Otherwise, if you want you can run a single file without setting up an environment via:
```
poetry run python your_script.py
```

A full example can be found in ```nesy_veri/examples/mnist_addition/verification.py```.
