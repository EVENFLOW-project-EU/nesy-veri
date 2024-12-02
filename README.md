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


### BDD-OIA
This is an autonomous driving example based on the [BDD-OIA](https://twizwei.github.io/bddoia_project/) dataset. To get the data download the .zip file from [here](https://drive.google.com/file/d/1WFiwRi_sMA_McZnkbEjh8Rnl-Im7_9Mk/view) and extract (unzip) the contents into the ```nesy_veri/examples/BDD_OIA/unprocessed_data``` directory.

BDD-OIA dataset is an extension of [BDD100K](https://bair.berkeley.edu/blog/2018/05/30/bdd/). The creators collected the complicated scenes (> 5 pedestrians or >5 vehicles) in the original BDD100K dataset, and then annotated them with 4 action categories and 21 explanation categories.

<img src="https://twizwei.github.io/bddoia_project/figs/bddoia_stat.jpg" class="center" width="500">

