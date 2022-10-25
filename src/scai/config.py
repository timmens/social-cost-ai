from pathlib import Path


SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()
ROOT = BLD.parent.resolve()
