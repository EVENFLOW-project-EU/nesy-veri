## Installation
We use [Poetry](https://python-poetry.org/) for package management. After you install Poetry itself, simply run:
```
poetry install
```
to install all packages. The Python version requirement is >3.11 .


You can then create a virtual environment through:
```
poetry shell
```
in which you can run everything as usual. Otherwise, if you want you can run a single file without setting up an environment via:
```
poetry run python your_script.py
```

## Data Exploration
All work is in: 
```
dfki/data.py
```
This traverses the directories and creates a Pytorch dataset. It also plots action distributions per trajectory per robot.