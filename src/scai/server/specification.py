from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from transformers import MODEL_FOR_MASKED_LM_MAPPING
from transformers import TrainingArguments


MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

SPECIFICATIONS = {
    "test": {
        "model_type": "distilbert",
        "tokenizer_name": "distilbert-base-uncased",
        "list_of_datasets": [("ai2_arc", "ARC-Easy")],
        "output_dir": "tmp_test",
        "max_steps": 10,
        "save_total_limit": 5,
        "logger_name": "test",
    },
    "bert-base": {
        "model_type": "bert",
        "tokenizer_name": "bert-base-uncased",
        "list_of_datasets": [
            ("wikitext", "wikitext-2-raw-v1"),
            ("bookcorpus", "plain_text"),
        ],
        "output_dir": "tmp_bert_base",
        "max_steps": 100_000,
        "save_total_limit": 5,
        "logger_name": "bert-base-logger",
    },
}


def get_specification(
    list_of_datasets: List[Tuple[str]],
    model_type: str,
    tokenizer_name: str = None,
    output_dir: Union[str, Path] = "tmp",
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


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune,
    or train from scratch."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want"
                " to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained"
                " from scratch. Example: n_embd=10,resid_pdrop=0.2,scale_attn_weights="
                "false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Where do you want to store the pretrained models downloaded from"
                "huggingface.co"
            )
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use one of the fast tokenizer (backed by the tokenizers "
                "library) or not."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": (
                "The specific model version to use (can be a branch name, tag name or"
                "commit id)."
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` "
                "(necessary to use this script with private models)."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or "
                "--model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training
    and eval."""

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The configuration name of the dataset to use (via the datasets "
                "library)."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the perplexity on "
                "(a text file)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "The percentage of the train set used as validation set in case there's"
                " no validation split"
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences "
                "longer than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether distinct lines of text in the dataset are to be handled as "
                "distinct sequences."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. If False, will pad the"
                " samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of "
                "training examples to this value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of "
                "evaluation examples to this value if set."
            )
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError(
                        "`train_file` should be a csv, a json or a txt file."
                    )
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError(
                        "`validation_file` should be a csv, a json or a txt file."
                    )
