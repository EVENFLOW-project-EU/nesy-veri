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

For detailed explanations of each dataset, read the paper, the code, or both.

### Multi-Digit MNIST Addition
Run ```nesy_veri/examples/mnist_addition/verification.py```. This downloads the original MNIST dataset, creates the multi-digit dataset, loads the trained CNN network, loads the circuits, and performs robustness verification for the true positive examples. 

To compare our end-to-end abstract method with a baseline performing approximate verification for the neural part of the NeSy system and exact bound propagation through the symbolic part run:
```
python nesy_veri/examples/mnist_addition/comparisons/gurobi_comparison.py --digits 2 3 4 5 6 --num_epochs 2 --epsilon 0.001
```
If you wish to ensure it works, run it for 2 and 3 digits. The full set will take a long time (minimum 4 days, larger epsilons require even longer) and a lot of RAM. In the paper we repeat the command above for three values of epsilon: 0.01, 0.001, 0.0001.

You can also compare with Marabou performing robustness verification on just the CNN of the NeSy architecture by running ```nesy_veri/examples/mnist_addition/comparisons/marabou_cnn_robustness.py```. Finally, for our failed attempt in performing NeSy verification with Marabou, see the ```nesy_veri/examples/mnist_addition/comparisons/marabou_nesy``` directory and the README inside it.

### ROAD-R
This is an autonomous driving task based on the [ROAD-R dataset](https://sites.google.com/view/road-r/home).

For instructions on downloading the data click [here](https://sites.google.com/view/road-r/dataset#h.9jzfyrwvkt7j). For the scripts below to run without modification we require that, after cloning this repo, you create a directory named "dataset" at the project root. This directory should contain the ```road_trainval_v1.0.json``` file as well as the ```rgb-images``` directory. We don't need the videos or the .txt files.

After you have the data, you can run ```nesy_veri/examples/ROAD_R/verification.py```. This will load the two pretrained networks, create our custom dataset, create the NeSy system, and perform robustness verification for the true positive examples.
