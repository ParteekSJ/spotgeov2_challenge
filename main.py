import os
import torch
import json
from data import load_dataset
import argparse
import ipdb
from model import create_model
from datetime import datetime
import numpy as np
import random
from logger import setup_logger
from engine import train_one_epoch, validate
from utils import save_checkpoint


def get_args_parser():
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")

    parser = argparse.ArgumentParser("SpotGeo Challenge", add_help=False)
    parser.add_argument("--dataset_path", default="./data/spotGEO", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--save_dir", default=f"./checkpoints/{date_time_str}", type=str, help="Ckpt Save Location")

    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--model_type", default="deeplab", type=str)

    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--validate_freq", default=10, type=int)

    parser.add_argument("--retrain", default=False, type=bool)

    return parser


def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_logger(name="spotgeov2", log_dir=args.save_dir, timestamp=timestamp)
    logger.info(f"Directory '{args.save_dir}' created.")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    trainloader, valloader = load_dataset(args)

    ipdb.set_trace()
    # Create model
    if args.model_type == "deeplab":
        model, criterion = create_model(args)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    model.to(device)

    # Define loss function, optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_f1_score = 0.0

    # Save configuration
    with open(os.path.join(args.save_dir, f"config_{timestamp}.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    start_epoch = 0

    if args.retrain:
        logger.info(f"Retraining ckpt {args.retrain_ckpt} for {args.epochs} epochs")
        ckpt = torch.load(args.retrain_ckpt)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_f1_score = ckpt["val_acc"]
        start_epoch = ckpt["epoch"]
        args.epochs = ckpt["epoch"] + args.epochs

    # Training loop
    logger.info("Starting training...")

    train_losses = []

    # ipdb.set_trace()
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, criterion, trainloader, optimizer, device, epoch + 1, logger, args)

        # Print epoch results
        logger.info(f"EPOCH {epoch + 1}, MEAN LOSS: {train_loss:.4f}")
        # logger.info(f"EPOCH {epoch + 1}, MEAN LOSS: {train_loss:.4f}, MEAN ACCURACY: {train_acc:.4f}")

        # Save a model every `args.save_freq`
        if (epoch + 1) % args.save_freq == 0:
            if train_loss == min(train_losses):
                save_checkpoint(model, optimizer, epoch, args, f"{args.model_type}_{epoch}_model.pt")
                train_losses.append(train_loss)

        # Validate every `args.validate_freq`
        if (epoch + 1) % args.validate_freq == 0 or epoch + 1 == args.epochs:
            results_dict = validate(model, criterion, valloader, device)
            logger.info(
                f"VAL LOSS: {results_dict['val_loss']:.4f}, \
                    VAL ACC: {results_dict['val_loss']:.4f}%, \
                        VAL F1 SCORE: {results_dict['f1_macro']:.4f}, \
                            VAL AUROC: {results_dict['auroc']}"
            )

            # Save best model
            if results_dict["f1_macro"] > best_val_f1_score:
                best_val_f1_score = results_dict["f1_macro"]
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    args,
                    results_dict["f1_macro"],
                    name=f"spotgeo_{args.model_type}_bestckpt.pth",
                )
                logger.info(f"[*] MODEL SAVED AT EPOCH {epoch + 1} WITH AUROC => {results_dict['f1_macro']:.2f}%")

    # logger.info(f"BEST VALIDATION ACCURACY: {best_val_f1_score:.2f}%")
    logger.info("TRAINING COMPLETED!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SpotGeo Challenge", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
