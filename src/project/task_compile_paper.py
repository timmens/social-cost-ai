import shutil

import pytask
from project.config import BLD
from project.config import ROOT
from project.config import SRC


DEPENDENCIES = [
    SRC / "paper" / doc for doc in ("paper.tex", "preamble.tex", "bibliography.bib")
]


@pytask.mark.latex(
    [
        "--pdf",
        "--interaction=nonstopmode",
        "--synctex=1",
        "--cd",
        "--quiet",
        "--shell-escape",
    ]
)
@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.produces(BLD / "paper" / "paper.pdf")
def task_compile_documents():
    pass


@pytask.mark.depends_on(BLD / "paper" / "paper.pdf")
@pytask.mark.produces(ROOT / "paper.pdf")
def task_copy_to_root(depends_on, produces):
    shutil.copy(depends_on, produces)
