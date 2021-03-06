import subprocess

import pytask
from project.config import BLD
from project.config import SRC


DEPENDENCIES = list(SRC.joinpath("notebooks").glob("*.ipynb"))
OUTPUTDIR = str(BLD.joinpath("notebooks"))


@pytask.mark.depends_on(DEPENDENCIES)
def task_run_notebook(depends_on, produces):
    subprocess.call(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            "--execute",
            str(list(depends_on.values())[0]),
            "--output-dir",
            OUTPUTDIR,
        ]
    )
