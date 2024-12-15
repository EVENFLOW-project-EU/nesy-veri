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

## Examples

### Multi-Digit MNIST Addition
Run ```nesy_veri/examples/mnist_addition/verification.py```. This downloads the original MNIST dataset, creates the multi-digit dataset, loads the trained CNN network, loads the circuits, and performs verification for the true positive examples. 

You can also compare with Gurobi and Marabou by running the scripts in ```nesy_veri/examples/mnist_addition/gurobi_comparison``` and ```nesy_veri/examples/mnist_addition/marabou``` respectively.