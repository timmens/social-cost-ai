import shutil

import pytask
from scai.config import BLD
from scai.config import ROOT
from scai.config import SRC


DEPENDENCIES = [
    SRC.joinpath("paper", doc)
    for doc in ("paper.tex", "preamble.tex", "bibliography.bib")
]
DEPENDENCIES += list(SRC.joinpath("paper", "sections").glob("*.tex"))


@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.latex(
    script=SRC.joinpath("paper", "paper.tex"),
    document=BLD.joinpath("paper", "paper.pdf"),
)
def task_compile_documents():
    pass


@pytask.mark.depends_on(BLD / "paper" / "paper.pdf")
@pytask.mark.produces(ROOT / "paper.pdf")
def task_copy_to_root(depends_on, produces):
    shutil.copy(depends_on, produces)
