import pytask
from project.config import BLD
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
