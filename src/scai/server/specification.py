from pathlib import Path
from typing import List
from typing import Tuple
from typing import Union

from scai.server.classes import DataTrainingArguments
from scai.server.classes import ModelArguments
from transformers import TrainingArguments


def get_specification(
    list_of_datasets: List[Tuple[str]],
    model_type: str,
    tokenizer_name: str = None,
    output_dir: Union[str, Path] = "/tmp/",
    save_total_limit: int = 5,
    max_steps: int = 10,
):
    tokenizer_name = model_type if tokenizer_name is None else tokenizer_name

    model_args = ModelArguments(
        model_type=model_type,
        tokenizer_name=tokenizer_name,
    )

    list_of_data_args = [
        DataTrainingArguments(
            dataset_name=name,
            dataset_config_name=config_name,
        )
        for name, config_name in list_of_datasets
    ]

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        do_train=True,
        max_steps=max_steps,
    )

    return model_args, list_of_data_args, training_args
