import click
import yaml
from scai.server.pipeline import train_pipeline
from scai.server.specification import SPECIFICATIONS


@click.command()
@click.option("--spec-name", default=None, type=str)
@click.option("--spec-file", default=None, type=click.Path(exists=True))
def main(spec_name, spec_file):
    """Main.

    Args:
        spec_name (str): Name of implemented specifications. Must be in {"test",
            "bert-base", "bert-large"}.
        spec_file (str): Path to model/data/training specification yaml file.

    """
    kwargs = SPECIFICATIONS.get(spec_name, None)

    if kwargs is None and spec_name is not None:
        msg = (
            f"spec_name is {spec_name} which is not in the set of implemented"
            f"specifications: {SPECIFICATIONS.keys()}"
        )
        raise ValueError(msg)
    elif kwargs is None and spec_file is not None:
        spec_file = click.format_filename(spec_file)
        with open(spec_file) as stream:
            kwargs = yaml.safe_load(stream)
    elif spec_file is not None:
        msg = "Only one option of {'spec-name', 'spec-file'} can be set."
        raise ValueError(msg)

    train_pipeline(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
