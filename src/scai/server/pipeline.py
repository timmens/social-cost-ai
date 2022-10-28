# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import transformers
from scai.server.data_management import get_data
from scai.server.data_management import preprocess_data
from scai.server.evaluate import do_evaluation
from scai.server.final import get_model_card_kwargs
from scai.server.logging import get_last_checkpoint
from scai.server.logging import get_logger
from scai.server.model import get_model
from scai.server.model import get_model_config
from scai.server.model import get_tokenizer
from scai.server.specification import get_specification
from scai.server.training import do_training
from transformers import DataCollatorForLanguageModeling
from transformers import is_torch_tpu_available
from transformers import Trainer
from transformers.utils import check_min_version
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version


# ======================================================================================
# Version checking and logger initialization
# ======================================================================================
check_min_version("4.24.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)


def train_pipeline(
    model_type,
    tokenizer_name,
    list_of_datasets,
    output_dir,
    max_steps,
    save_total_limit,
    logger_name,
):

    # ==================================================================================
    # Specification
    # ==================================================================================

    model_args, list_of_data_args, training_args = get_specification(
        list_of_datasets=list_of_datasets,
        model_type=model_type,
        tokenizer_name=tokenizer_name,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        max_steps=max_steps,
    )

    data_args = list_of_data_args[0]

    # Sending telemetry. Tracking the example usage helps us better allocate resources
    # to maintain them. The information sent is the one passed as arguments along with
    # your Python/PyTorch versions.
    send_example_telemetry("run_mlm", model_args, data_args, framework="pytorch")

    transformers.set_seed(training_args.seed)

    # ==================================================================================
    # Logging
    # ==================================================================================

    logger = get_logger(logger_name, training_args=training_args)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: "
        f"{bool(training_args.local_rank != -1)}, 16-bits training: "
        f"{training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # ==================================================================================
    # Checkpoint
    # ==================================================================================

    last_checkpoint = get_last_checkpoint(training_args, logger=logger)  # None if no
    # checkpoint exists

    # ==================================================================================
    # Data management: load data
    # ==================================================================================

    data = get_data(list_of_data_args, model_args)

    # ==================================================================================
    # Load pretrained model and tokenizer
    # ==================================================================================

    config = get_model_config(model_args, logger=logger)

    tokenizer = get_tokenizer(model_args)

    model = get_model(model_args, config=config, tokenizer=tokenizer, logger=logger)

    # ==================================================================================
    # Data management: preprocessing
    # ==================================================================================

    (
        train_dataset,
        eval_dataset,
        compute_metrics,
        preprocess_logits_for_metrics,
    ) = preprocess_data(
        data=data,
        data_args=data_args,
        training_args=training_args,
        tokenizer=tokenizer,
        logger=logger,
    )

    # ==================================================================================
    # Data collator
    # ==================================================================================

    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = (
        data_args.line_by_line
        and training_args.fp16
        and not data_args.pad_to_max_length
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    # ==================================================================================
    # Training
    # ==================================================================================

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    if training_args.do_train:
        trainer = do_training(
            trainer,
            training_args=training_args,
            train_dataset=train_dataset,
            data_args=data_args,
            last_checkpoint=last_checkpoint,
        )

    # ==================================================================================
    # Evaluation
    # ==================================================================================

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        trainer = do_evaluation(trainer, data_args, eval_dataset)

    # ==================================================================================
    # Create model card
    # ==================================================================================

    kwargs = get_model_card_kwargs(model_args, data_args)
    trainer.create_model_card(**kwargs)
