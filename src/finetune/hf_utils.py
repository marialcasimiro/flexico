from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from metrics.metrics import compute_metrics


def tokenize_inputs(
    examples,
    tokenizer: AutoTokenizer,
    max_input_length: int,
    max_target_length: int,
):
    inputs = list(examples["source"])
    targets = list(examples["target"])

    model_inputs = tokenizer(inputs, truncation=True, max_length=max_input_length)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, truncation=True, max_length=max_target_length)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    parameters: dict,
):
    print(f"[D] Tokenizing dataset {dataset}")
    return dataset.map(
        tokenize_inputs,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_input_length": parameters["max_input_length"],
            "max_target_length": parameters["max_target_length"],
        },
    )


def get_hf_training_args(parameters: dict):
    print("[D] Defining training arguments")
    return Seq2SeqTrainingArguments(
        output_dir=f"{parameters['model_checkpoint'].split('/')[-1]}-finetuneInstant_{parameters['finetune_instant']}-{parameters['source_lang']}-to-{parameters['target_lang']}",
        evaluation_strategy="no",
        include_inputs_for_metrics=True,
        learning_rate=parameters["learning_rate"],
        per_device_train_batch_size=parameters["batch_size"],
        per_device_eval_batch_size=parameters["batch_size"],
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=parameters["train_epochs"],
        predict_with_generate=True,  # Whether to use generate to calculate generative metrics (ROUGE, BLEU)
        fp16=parameters["fp16"],
        push_to_hub=False,
        use_cpu=(not parameters["gpu_on"]),
        save_strategy="no",  # No save is done during training
        logging_strategy="no",  # No logging is done during training
    )


def get_hf_trainer(
    model: AutoModelForSeq2SeqLM,
    training_args: Seq2SeqTrainingArguments,
    tokenized_datasets,
    metrics: dict,
    tokenizer,
    comet_batch_size: int = 64,
):
    # define model trainer
    print("[D] Instantiating Seq2Seq model trainer")
    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model
        ),  # deal with the padding for dynamic batching
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(
            p,
            metrics=metrics,
            tokenizer=tokenizer,
            comet_batch_size=comet_batch_size,
        ),
    )
