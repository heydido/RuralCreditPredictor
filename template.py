import os
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "RuralCreditPredictor"

list_of_files = [
    ".github/workflows/.gitkeep",
    "artifacts/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "notebook/experiment.ipynb",
    "notebook/data/.gitkeep",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "templates/index.html"
]

for a_file in list_of_files:
    a_file = Path(a_file)
    file_dir, file_name = os.path.split(a_file)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for the file: {file_name}")

    if (not os.path.exists(a_file)) or (os.path.getsize(a_file) == 0):
        with open(a_file, "w") as f:
            pass
            logging.info(f"Creating empty file: {a_file}")
    else:
        logging.info(f"{file_name} is already exists")
