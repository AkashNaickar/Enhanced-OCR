"""Fine-tune ``t5-base`` to summarize news articles.

Importing this module has no side effects: the dataset download, tokenization,
model load and training all happen inside :func:`main`. The heavy ML
dependencies are imported lazily there, so the module can be imported and its
pure helpers unit tested without PyTorch or Transformers installed.
"""

from __future__ import annotations

import argparse

from enhanced_ocr.textutils import average_length, find_longest_length, make_summary_prompt

MODEL = "t5-base"
BATCH_SIZE = 2
EPOCHS = 1
OUT_DIR = "results_t5base"
MAX_LENGTH = 256


def build_preprocess_function(tokenizer, max_length=MAX_LENGTH):
    """Return a ``datasets.map`` function that tokenizes articles and summaries."""

    def preprocess_function(examples):
        inputs = [make_summary_prompt(article) for article in examples["Articles"]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        targets = list(examples["Summaries"])
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function


def build_compute_metrics(tokenizer, rouge):
    """Return a Trainer metric function that reports ROUGE and mean gen length."""

    def compute_metrics(eval_pred):
        import numpy as np

        predictions, labels = eval_pred.predictions[0], eval_pred.label_ids

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
            rouge_types=["rouge1", "rouge2", "rougeL"],
        )

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
        ]
        result["gen_len"] = float(np.mean(prediction_lens))

        return {key: round(value, 4) for key, value in result.items()}

    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    """Reduce logits to token ids to avoid the Trainer's memory blow-up."""
    import torch

    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


def describe_lengths(name, texts):
    """Print longest and average word counts plus the >N-word buckets."""
    longest, counter_4k, counter_2k, counter_1k, counter_500 = find_longest_length(texts)
    print(f"Longest {name} length: {longest} words")
    print(f"{name} larger than 4000 words: {counter_4k}")
    print(f"{name} larger than 2000 words: {counter_2k}")
    print(f"{name} larger than 1000 words: {counter_1k}")
    print(f"{name} larger than 500 words: {counter_500}")
    print(f"Average {name} length: {average_length(list(texts)):.2f} words")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune t5-base to summarize news articles.")
    parser.add_argument("--dataset", default="gopalkalpande/bbc-news-summary")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", default=OUT_DIR)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args(argv)

    import evaluate
    import torch
    from datasets import load_dataset
    from transformers import (
        T5ForConditionalGeneration,
        T5Tokenizer,
        Trainer,
        TrainingArguments,
    )

    dataset = load_dataset(args.dataset, split="train")
    full_dataset = dataset.train_test_split(test_size=0.2, shuffle=True)
    dataset_train = full_dataset["train"]
    dataset_valid = full_dataset["test"]

    print(dataset_train)
    print(dataset_valid)

    describe_lengths("article", dataset_train["Articles"])
    describe_lengths("summary", dataset_train["Summaries"])

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    preprocess_function = build_preprocess_function(tokenizer, args.max_length)

    tokenized_train = dataset_train.map(preprocess_function, batched=True, num_proc=1)
    tokenized_valid = dataset_valid.map(preprocess_function, batched=True, num_proc=1)

    model = T5ForConditionalGeneration.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    rouge = evaluate.load("rouge")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=args.output_dir,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="tensorboard",
        learning_rate=0.0001,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=build_compute_metrics(tokenizer, rouge),
    )

    trainer.train()

    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)

    return trainer


if __name__ == "__main__":
    main()
