#!/usr/bin/env python3
"""
Run OpenAI fine-tuning for SemEval 2026 Task 12.
"""

import argparse
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from openai import OpenAI


def upload_file(client: OpenAI, filepath: str, purpose: str = "fine-tune") -> str:
    """Upload a file and return the file ID."""
    print(f"Uploading {filepath}...")
    with open(filepath, "rb") as f:
        response = client.files.create(file=f, purpose=purpose)
    print(f"  File ID: {response.id}")
    return response.id


def create_finetune_job(
    client: OpenAI,
    train_file_id: str,
    val_file_id: str = None,
    model: str = "gpt-4o-mini-2024-07-18",
    suffix: str = "semeval-task12",
    n_epochs: int = 3,
) -> str:
    """Create a fine-tuning job and return the job ID."""
    print(f"\nCreating fine-tuning job...")
    print(f"  Model: {model}")
    print(f"  Epochs: {n_epochs}")

    kwargs = {
        "training_file": train_file_id,
        "model": model,
        "suffix": suffix,
        "hyperparameters": {"n_epochs": n_epochs},
    }

    if val_file_id:
        kwargs["validation_file"] = val_file_id

    job = client.fine_tuning.jobs.create(**kwargs)
    print(f"  Job ID: {job.id}")
    print(f"  Status: {job.status}")
    return job.id


def wait_for_job(client: OpenAI, job_id: str, poll_interval: int = 60):
    """Wait for a fine-tuning job to complete."""
    print(f"\nWaiting for job {job_id} to complete...")

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status

        print(f"  Status: {status}", end="")

        if hasattr(job, "trained_tokens") and job.trained_tokens:
            print(f" | Trained tokens: {job.trained_tokens}", end="")

        print()

        if status == "succeeded":
            print(f"\n✓ Fine-tuning complete!")
            print(f"  Fine-tuned model: {job.fine_tuned_model}")
            return job.fine_tuned_model

        elif status == "failed":
            print(f"\n✗ Fine-tuning failed!")
            if hasattr(job, "error") and job.error:
                print(f"  Error: {job.error}")
            return None

        elif status == "cancelled":
            print(f"\n✗ Fine-tuning cancelled!")
            return None

        time.sleep(poll_interval)


def list_jobs(client: OpenAI, limit: int = 10):
    """List recent fine-tuning jobs."""
    print("Recent fine-tuning jobs:")
    jobs = client.fine_tuning.jobs.list(limit=limit)
    for job in jobs.data:
        print(f"  {job.id}: {job.status} | {job.model} -> {job.fine_tuned_model or 'pending'}")


def main():
    parser = argparse.ArgumentParser(description="Run OpenAI fine-tuning")
    parser.add_argument("--action", default="create", choices=["create", "status", "list", "full"])
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--job-id", help="Job ID for status check")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for completion")
    parser.add_argument("--full", action="store_true", help="Use full training set (no validation)")

    args = parser.parse_args()

    client = OpenAI()

    if args.action == "list":
        list_jobs(client)
        return

    if args.action == "status":
        if not args.job_id:
            print("Error: --job-id required for status check")
            return
        job = client.fine_tuning.jobs.retrieve(args.job_id)
        print(f"Job: {job.id}")
        print(f"Status: {job.status}")
        print(f"Model: {job.model}")
        print(f"Fine-tuned model: {job.fine_tuned_model or 'pending'}")
        return

    # Create new fine-tuning job
    if args.full:
        train_file = "data/finetune/openai_full.jsonl"
        val_file = None
    else:
        train_file = "data/finetune/openai_train.jsonl"
        val_file = "data/finetune/openai_val.jsonl"

    # Upload files
    train_file_id = upload_file(client, train_file)
    val_file_id = upload_file(client, val_file) if val_file else None

    # Create job
    job_id = create_finetune_job(
        client,
        train_file_id,
        val_file_id,
        model=args.model,
        n_epochs=args.epochs,
    )

    if not args.no_wait:
        fine_tuned_model = wait_for_job(client, job_id)
        if fine_tuned_model:
            print(f"\n{'='*60}")
            print("To use your fine-tuned model:")
            print(f"{'='*60}")
            print(f"""
python run_baseline.py --model {fine_tuned_model} --split dev

Or in Python:
    response = client.chat.completions.create(
        model="{fine_tuned_model}",
        messages=[...]
    )
""")
    else:
        print(f"\nJob created: {job_id}")
        print(f"Check status with: python run_finetune.py --action status --job-id {job_id}")


if __name__ == "__main__":
    main()
