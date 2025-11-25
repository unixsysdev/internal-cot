# Training script for Coconut with Qwen3-0.6B
# Adapted from the original Coconut implementation by Meta Platforms, Inc.
# This script supports both distributed (multi-GPU) and single-GPU training.

import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from coconut_qwen import CoconutQwen
from dataset_qwen import (
    get_dataset,
    get_question_latent_dataset,
    get_cot_latent_dataset,
    MyCollator,
)

from tqdm import tqdm
from copy import copy
import itertools
import os
import sys
import yaml
import json
import gc
import argparse
import functools
from utils import Config, set_seed


def setup_distributed():
    """
    Initialize distributed training environment.
    Returns True if distributed training is available.
    """
    if torch.cuda.device_count() > 1 and "RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return True, local_rank, rank, world_size
    else:
        # Single GPU or CPU training
        rank = 0
        local_rank = 0
        world_size = 1
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return False, local_rank, rank, world_size


def get_auto_wrap_policy_for_qwen():
    """
    Get the auto-wrap policy for FSDP with Qwen3.
    """
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen2DecoderLayer},
        )
    except ImportError:
        # Fallback: no specific wrapping
        return None


def main():
    parser = argparse.ArgumentParser(description="Coconut training for Qwen3")
    parser.add_argument("config_file", help="Path to YAML configuration file")
    parser.add_argument("--single-gpu", action="store_true", help="Force single GPU training")
    args = parser.parse_args()

    # Setup distributed or single-GPU training
    if args.single_gpu:
        is_distributed = False
        local_rank = 0
        rank = 0
        world_size = 1
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    else:
        is_distributed, local_rank, rank, world_size = setup_distributed()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Load configuration
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Configuration:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    if is_distributed:
        torch.distributed.barrier()

    # Check for existing checkpoints (for resuming interrupted training)
    cur_ckpts = os.listdir(save_dir) if os.path.exists(save_dir) else []

    if len(cur_ckpts) > 0 and not configs.only_eval:
        if rank == 0:
            print("Found previous run, resuming from latest checkpoint...")

        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))

        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            configs.resume = int(latest_checkpoint.split("_")[1])
            load_dir = os.path.join(save_dir, latest_checkpoint)
            configs.load_model_path = load_dir
            print(f"Loading from epoch {configs.resume}")

    elif configs.resume != 0 and configs.load_model_path == "None":
        print(f"Warning: resume={configs.resume} but no checkpoint specified")

    # Load Qwen3-0.6B model and tokenizer
    if rank == 0:
        print(f"Loading model: {configs.model_id}")

    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)

    # Setup tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens for Coconut
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    if rank == 0:
        print(f"Special token IDs - latent: {latent_id}, start: {start_id}, end: {end_id}")

    loaded = False

    # Load pre-trained weights if specified
    if configs.load_model_path != "None":
        saved_weights = torch.load(
            configs.load_model_path, map_location=device
        )

        if configs.coconut and not any(
            k.startswith("base_causallm") for k in saved_weights.keys()
        ):
            # Loading a base model into Coconut wrapper
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif not configs.coconut and any(
            k.startswith("base_causallm") for k in saved_weights.keys()
        ):
            raise ValueError("Cannot load Coconut model weights into a base model")

        elif configs.coconut and any(
            k.startswith("base_causallm") for k in saved_weights.keys()
        ):
            # Loading from preempted Coconut run - handled later
            pass

        else:
            # Resume or evaluate base model
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    # Initialize new token embeddings
    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()

        # Initialize with a known token's embedding for stability
        # Try to use a common token like "(" or "["
        init_tokens = ["(", "[", "<<", "<"]
        target_id = None
        for t in init_tokens:
            tid = tokenizer.convert_tokens_to_ids(t)
            if tid != tokenizer.unk_token_id:
                target_id = tid
                break

        if target_id is None:
            # Fallback: use a random existing token
            target_id = 100

        if rank == 0:
            print(f"Initializing special tokens with embedding from token ID {target_id}")

        target_embedding = embeddings.weight.data[target_id].clone()
        for token_id in [latent_id, start_id, end_id]:
            embeddings.weight.data[token_id] = target_embedding.clone()
            # Also initialize LM head if not tied
            if hasattr(model, 'lm_head') and not model.config.tie_word_embeddings:
                model.lm_head.weight.data[token_id] = model.lm_head.weight.data[target_id].clone()

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    # Wrap model with Coconut if enabled
    if configs.coconut:
        model = CoconutQwen(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    if rank == 0:
        print(f"Running on device: {device}, distributed: {is_distributed}, world_size: {world_size}")

    model = model.to(device)

    # Apply mixed precision if specified
    if configs.bf16:
        model.to(torch.bfloat16)

    # Setup parallel training
    if is_distributed:
        auto_wrap_policy = get_auto_wrap_policy_for_qwen()

        if configs.only_eval:
            parallel_model = DDP(model, device_ids=[local_rank])
        else:
            if auto_wrap_policy:
                parallel_model = FSDP(
                    model, auto_wrap_policy=auto_wrap_policy, device_id=local_rank
                )
            else:
                parallel_model = DDP(model, device_ids=[local_rank])
    else:
        parallel_model = model

    del model

    if rank == 0:
        print(parallel_model)

    # Load validation data
    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [
        d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))
    ]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000
    )

    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, max_size=5000 if configs.debug else 100000000
        )

    # Set max tokens based on dataset
    if "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 128

    total_train_steps = 0

    # Initialize wandb logging
    if not configs.debug and not configs.only_eval and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(config_dict, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])
    else:
        wandb_run = None

    # Setup optimizer
    if configs.reset_optimizer:
        optimizer = None
    else:
        model_params = parallel_model.parameters() if not is_distributed else parallel_model.parameters()
        optimizer = optim.AdamW(
            model_params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )

    best_acc = 0
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    # Main training loop
    for epoch in range(configs.resume, configs.num_epochs):
        # Determine current training stage
        scheduled_stage = (
            0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
        )

        if rank == 0:
            print(f"\n=== Epoch {epoch + 1}/{configs.num_epochs}, Stage {scheduled_stage} ===")

        # Prepare validation dataset for generation
        dataset_gen_val = get_question_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
        )

        if is_distributed:
            valid_gen_dataloader = torch.utils.data.DataLoader(
                dataset_gen_val,
                num_workers=1,
                pin_memory=True,
                batch_size=1,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_gen_val, shuffle=False),
            )
        else:
            valid_gen_dataloader = torch.utils.data.DataLoader(
                dataset_gen_val,
                num_workers=1,
                pin_memory=True,
                batch_size=1,
                collate_fn=collator,
                shuffle=False,
            )

        # Training phase
        if not configs.only_eval:
            dataset_train = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                shuffle=True,
            )

            if is_distributed:
                train_dataloader = torch.utils.data.DataLoader(
                    dataset_train,
                    num_workers=1,
                    shuffle=False,
                    pin_memory=True,
                    batch_size=configs.batch_size_training,
                    collate_fn=collator,
                    sampler=DistributedSampler(dataset_train, shuffle=True),
                )
            else:
                train_dataloader = torch.utils.data.DataLoader(
                    dataset_train,
                    num_workers=1,
                    shuffle=True,
                    pin_memory=True,
                    batch_size=configs.batch_size_training,
                    collate_fn=collator,
                )

            # Validation loss dataloader
            dataset_loss_val = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            )

            if is_distributed:
                valid_loss_dataloader = torch.utils.data.DataLoader(
                    dataset_loss_val,
                    num_workers=1,
                    shuffle=False,
                    pin_memory=True,
                    batch_size=configs.batch_size_training,
                    collate_fn=collator,
                    sampler=DistributedSampler(dataset_loss_val, shuffle=False),
                )
            else:
                valid_loss_dataloader = torch.utils.data.DataLoader(
                    dataset_loss_val,
                    num_workers=1,
                    shuffle=False,
                    pin_memory=True,
                    batch_size=configs.batch_size_training,
                    collate_fn=collator,
                )

            # Reset optimizer per stage if configured
            if configs.reset_optimizer:
                if optimizer is not None:
                    del optimizer
                model_params = parallel_model.parameters()
                optimizer = optim.AdamW(
                    model_params,
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                )

            # Set model to training mode
            if is_distributed:
                parallel_model.module.train()
            else:
                parallel_model.train()

            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch + 1}",
                total=total_length,
                dynamic_ncols=True,
                disable=rank != 0,
            )

            # Training loop
            for step, batch in enumerate(train_dataloader):
                # Log first batch for debugging
                if step == 0 and wandb_run and rank == 0:
                    print("Logging training data sample...")
                    cur_bs = len(batch["input_ids"])
                    text_str = ""
                    for data_idx in range(min(cur_bs, 2)):  # Log first 2 samples
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            text_str += (
                                str(batch["input_ids"][data_idx][token_idx].item())
                                + " "
                                + str(batch["labels"][data_idx][token_idx].item())
                                + " "
                                + tokenizer.decode(batch["input_ids"][data_idx][token_idx])
                                + "\n"
                            )
                        text_str += "====" * 10 + "\n"
                    text_table.add_data(total_train_steps, text_str)
                    wandb_run.log({"data_table": copy(text_table)})

                total_train_steps += 1

                # Move batch to device
                batch = {
                    key: batch[key].to(device) for key in batch.keys() if key != "idx"
                }

                # Forward pass
                if is_distributed:
                    outputs = parallel_model(**batch)
                else:
                    outputs = parallel_model(**batch)

                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()

                # Optimizer step with gradient accumulation
                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                # Log to wandb
                if wandb_run and rank == 0:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float() * configs.gradient_accumulation_steps,
                    }
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"Training Epoch: {epoch + 1}/{configs.num_epochs}, "
                    f"batch {step}/{len(train_dataloader)} "
                    f"(loss: {float(loss.detach().float() * configs.gradient_accumulation_steps):.4f})"
                )

            pbar.close()

            if is_distributed:
                dist.barrier()

            # Save checkpoint
            if not configs.save_only_improve and not configs.debug and not configs.only_eval:
                if is_distributed:
                    states = parallel_model.state_dict()
                else:
                    states = parallel_model.state_dict()

                if rank == 0:
                    torch.save(states, os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
                    print("Saved checkpoint.")

                if is_distributed:
                    dist.barrier()
                del states
                gc.collect()
                torch.cuda.empty_cache()

            # Validation loss
            total_loss = 0
            with torch.no_grad():
                if is_distributed:
                    parallel_model.module.eval()
                else:
                    parallel_model.eval()

                for step, batch in enumerate(valid_loss_dataloader):
                    batch = {
                        key: batch[key].to(device) for key in batch.keys() if key != "idx"
                    }

                    if is_distributed:
                        outputs = parallel_model(**batch)
                    else:
                        outputs = parallel_model(**batch)

                    loss = outputs.loss
                    if is_distributed:
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        total_loss += loss.item() / world_size
                    else:
                        total_loss += loss.item()

                if wandb_run and rank == 0:
                    log_dict = {
                        "eval/loss": total_loss / len(valid_loss_dataloader),
                    }
                    wandb_run.log(log_dict)
                    print(f"Validation loss: {total_loss / len(valid_loss_dataloader):.4f}")

        # Generation accuracy evaluation
        total_length = len(valid_gen_dataloader)
        pbar = tqdm(
            colour="blue",
            desc="Test Accuracy",
            total=total_length,
            dynamic_ncols=True,
            disable=rank != 0,
        )

        cor = torch.tensor(0, device=device)
        cor_cot = torch.tensor(0, device=device)
        total = torch.tensor(0, device=device)

        with torch.no_grad():
            if is_distributed:
                parallel_model.module.eval()
            else:
                parallel_model.eval()

            for idx, batch in enumerate(valid_gen_dataloader):
                test_idx = batch["idx"][0]

                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if v is not None and k not in ["idx", "position_ids"]
                }

                assert len(batch["input_ids"]) == 1
                answer = answers_val[test_idx.cpu().item()]
                answer_cot = cot_val[test_idx.cpu().item()]
                question = question_val[test_idx.cpu().item()]

                total += 1

                # Generate
                if is_distributed:
                    outputs = parallel_model.module.generate(
                        **batch,
                        max_new_tokens=max_new_tokens,
                        synced_gpus=not configs.only_eval,
                    )
                else:
                    outputs = parallel_model.generate(
                        **batch,
                        max_new_tokens=max_new_tokens,
                        synced_gpus=False,
                    )

                text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer_output = text_output.split("#")[-1].replace(",", "").strip()
                cot_output = ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()

                # Print some examples
                if idx < 5 and rank == 0:
                    print(f"\nQuestion {test_idx}: Expected='{answer}' CoT='{answer_cot}'")
                    print(f"Generated: '{tokenizer.decode(outputs[0])}'")
                    print(f"Extracted answer: '{answer_output}'")

                cor += answer_output == answer
                cor_cot += cot_output == answer_cot

                pbar.update(1)
                pbar.set_description(
                    f"Test accuracy: {float(cor.detach().float() / total.detach().float()):.2f}"
                )

            pbar.close()

            if rank == 0:
                print(f"Device {rank}: Correct={cor.item()}, CoT={cor_cot.item()}, Total={total.item()}")

        # Aggregate results across GPUs
        if is_distributed:
            dist.all_reduce(cor_cot, op=dist.ReduceOp.SUM)
            dist.all_reduce(cor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

        cor_val = cor.item()
        cor_cot_val = cor_cot.item()
        total_val = total.item()

        if rank == 0:
            accuracy = cor_val / total_val if total_val > 0 else 0
            cot_em = cor_cot_val / total_val if total_val > 0 else 0
            print(f"\nValidation Accuracy: {cor_val} / {total_val} = {accuracy:.4f}")
            print(f"CoT Exact Match: {cor_cot_val} / {total_val} = {cot_em:.4f}")

        sys.stdout.flush()

        if wandb_run:
            wandb_run.log({
                "eval/acc": cor_val / total_val if total_val > 0 else 0,
                "eval/cot_em": cor_cot_val / total_val if total_val > 0 else 0
            })

        if configs.only_eval:
            break

        # Save best checkpoint
        if is_distributed:
            dist.barrier()

        accuracy = cor_val / total_val if total_val > 0 else 0
        if (
            accuracy > best_acc
            and configs.save_only_improve
            and not configs.debug
            and not configs.only_eval
        ):
            if is_distributed:
                states = parallel_model.state_dict()
            else:
                states = parallel_model.state_dict()

            if rank == 0:
                torch.save(states, os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
                print("Saved best checkpoint.")

            best_acc = accuracy

            if is_distributed:
                dist.barrier()
            del states
            gc.collect()
            torch.cuda.empty_cache()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
