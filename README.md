# my_project

Exercises from day 2 of MLops course

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## How to run the project

All commands must be executed from the root directory of the project.

1. Install dependencies and the project itself:

pip install -r requirements.txt
pip install -e .

2. Prepare the data folders:

The raw data must be placed in:
data/raw/

The processed data will be saved in:
data/processed/

3. Preprocess the data:

Run the data preprocessing script with two arguments:
- first argument: path to raw data
- second argument: path to processed data

Command:
python src/my_project/data.py data/raw data/processed

Alternatively, you can use the predefined invoke task:
invoke preprocess-data

4. Train the model:

To train the model, run:
python src/my_project/train.py

Or using invoke:
invoke train

Trained models are saved in:
models/

5. Visualize learned embeddings:

To visualize intermediate representations using t-SNE, run:
python src/my_project/visualize.py models/model.pth

Optional argument:
--figure-name <name_of_output_png>

Example:
python src/my_project/visualize.py models/model.pt --figure-name embeddings.png

The visualization will be saved in:
reports/figures/

6. Invoke utility commands:

To list all available project tasks:
invoke --list


