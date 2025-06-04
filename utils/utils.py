import os
import torch


def save_checkpoint(model, optimizer, epoch, args, name, val_acc=None):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # "scheduler_state_dict": scheduler.state_dict(),
        "val_acc": 0 if val_acc is None else val_acc,
        "config": args.__dict__,
    }

    checkpoint_path = os.path.join(args.save_dir, name)
    torch.save(checkpoint, checkpoint_path)
