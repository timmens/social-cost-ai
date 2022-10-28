import click
import yaml
from scai.server.pipeline import train_pipeline


@click.command()
@click.argument("spec_path", type=click.Path(exists=True))
def main(spec_path):
    spec_path = click.format_filename(spec_path)
    with open(spec_path) as stream:
        specs = yaml.safe_load(stream)
    train_pipeline(**specs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
