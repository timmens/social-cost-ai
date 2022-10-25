import subprocess

import pytask
from scai.config import BLD
from scai.config import SRC


DEPENDENCIES = list(SRC.joinpath("notebooks").glob("*.ipynb"))
OUTPUTDIR = str(BLD.joinpath("notebooks"))


@pytask.mark.depends_on(DEPENDENCIES)
def task_run_notebook(depends_on, produces):  # noqa: U100
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
