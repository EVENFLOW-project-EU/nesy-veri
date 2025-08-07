import os
import json
import copy
import torch
import inspect
import numpy as np
import torchmetrics
from pathlib import Path
from typing import Optional, List
from torch import nn, optim
from datetime import datetime
from torch.utils.data import DataLoader
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from dfki.data import DetectedRobotImages
from dfki.network_definitions import CNN3D, CNNLSTM, PretrainedLinear, RobotNet

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


def run_verification_dataloader(
    network,
    dataloader,
    verification_metrics: dict,
    method: str,
    epsilons: List[float],
    device,
):
    progress_bar = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("(verification)"),
        transient=True
    )
    
    verification_metrics[method] = {}

    network = network.eval()
    
    robust, correct, not_safe, total_samples = [], [], [], []
    non_robust_classifications = []
        
    
    with progress_bar as progress:    
        for eps_idx, epsilon in enumerate(epsilons):
                
            if eps_idx == 0:
                # Create a progress task for each epsilon
                verify_task = progress.add_task(
                    "[green] verifying: method: {} | epsilon: {} | [{}/{}]".format(method, epsilon, eps_idx, len(epsilons)), total=len(epsilons)
                )
                
            for idx, (inputs, labels, org_indices) in enumerate(dataloader):
                    
                if idx == 0:
                    num_robust = 0
                    num_correct = 0
                    
                    # Create a progress task for each input in the dataloader
                    verify_epsilon_dl_task = progress.add_task(
                        "[cyan] verifying batch: [{}/{}]".format(idx, len(dataloader)), total=len(dataloader)
                    )
                    
                    # Create the Bounded Model
                    lirpa_model = BoundedModule(net, torch.empty_like(inputs), device=device)
                    print("Running on", device)
                
                    
                inputs, labels = inputs.to(device), labels.to(device)
                    
                # Create the Bounded Tensor with Perturbation
                ptb = PerturbationLpNorm(norm = float("inf"), eps = epsilon)
                ptb_inputs = BoundedTensor(inputs, ptb)

                preds = lirpa_model(ptb_inputs)
                pred_targets = torch.argmax(preds, dim=1)

                lb, ub = lirpa_model.compute_bounds(x=(ptb_inputs,), method=method.split()[0])
                # outputs = network(inputs)



                with torch.no_grad():
                    # verification_metrics = {
                    #     metric_name: (
                    #         (idx * progress.tasks[epoch_task].fields[metric_name])
                    #         + metric(outputs, labels).item()
                    #     )
                    #     / (idx + 1)
                    #     for metric_name, metric in metrics.items()
                    # }
                    final_labels = torch.argmax(labels, dim=1) if labels.ndim > 1 else labels
                    for i in range(len(inputs)):
                        truth_idx = final_labels[i].item()
                        
                        
                        classification_robust = False
                        classification_correct = pred_targets[i].item() == truth_idx
                        
                        # Check if the lower bound of the predicted class is greater than the upper bounds of all other classes
                        if (
                            lb[i][truth_idx]
                            > torch.cat(
                                (
                                    ub[i][:truth_idx],
                                    ub[i][truth_idx + 1:],
                                )
                            )
                        ).all().item():
                            classification_robust = True
                        else:
                            classification_robust = False
                            non_robust_classifications.append(org_indices[i].item())
                            
                        # robust only if classification is safe and correct
                        num_robust += 1 if classification_correct and classification_robust else 0
                        num_correct += 1 if classification_correct else 0
                    
        
            robust.append(num_robust)
            correct.append(num_correct)
            not_safe.append(len(dataloader.dataset) - num_robust)
            total_samples.append(len(dataloader.dataset))

            progress.update(
                        task_id=verify_epsilon_dl_task,
                        advance=idx,
                        **verification_metrics,
                    )
        
        progress.update(
                        task_id=verify_task,
                        advance=eps_idx,
                    )

        verification_metrics[method]["robust"] = robust
        verification_metrics[method]["correct"] = correct
        verification_metrics[method]["not_safe"] = not_safe
        verification_metrics[method]["total_samples"] = total_samples
        verification_metrics[method]["non_robust_classifications"] = non_robust_classifications

    return network, verification_metrics


def cross_validation(
    video_indices: list[int], num_test_vids: int, num_folds: int, seed: Optional[int]
):

    if seed is not None:
        np.random.seed(seed)

    test_videos = np.random.choice(video_indices, num_test_vids, replace=False)

    remaining_videos = np.setdiff1d(video_indices, test_videos)
    np.random.shuffle(remaining_videos)

    val_splits = np.array_split(remaining_videos, num_folds)

    splits = []
    for val_videos in val_splits:
        train_videos = np.setdiff1d(remaining_videos, val_videos)
        splits.append({"train": train_videos.tolist(), "val": val_videos.tolist()})

    return test_videos, splits



def verify(net: nn.Module, data_config: dict, verify_config: dict, cv_splits: list[dict]):


    saved_models_dir = Path(verify_config["saved_model_dir"])
    if saved_models_dir is None or not os.path.exists(
        saved_models_dir
    ):
        raise RuntimeError(
            "The saved model directory does not exist: {}. Train the model prior to verification!".format(
                verify_config["saved_model_dir"])   
        )

    results_per_split = {}
    results_per_split["verify_config"] = verify_config
    results_per_split["data_config"] = data_config.copy()
    results_per_split["data_config"]["dataset_root"] = str(results_per_split["data_config"]["dataset_root"])

    # iterate through all train/validation splits
    for i, inner in enumerate(cv_splits):
        print(f"Split {i+1}/{len(cv_splits)}")
        
        # create train/val datasets
        val_dataset = DetectedRobotImages(
            data_config["downsample_img_by"],
            data_config["downsample_sequence"],
            data_config["imgs_per_sec"],
            data_config["image_sequences"],
            data_config["imgs_per_sequence"],
            data_config["time_spacing"],
            data_config["regress"],
            inner["val"],
            data_config["dataset_root"],
            data_config["dataset_index_needed"],
        )

        # define training config
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        # load the saved state dict
        net.load_state_dict(
            torch.load(
                Path(verify_config['saved_model_dir']) / f"{net.__class__.__name__}_split_{i+1}of{len(cv_splits)}.pt",
                map_location=torch.device(device),
            )
        )
        
        
        # create dataloaders for verification
        val_dl = DataLoader(val_dataset, verify_config["batch_size"], shuffle=True)

        net.to(device)
        verify_metrics = {}
        
        for method in verify_config["methods"]:
            net, verify_metrics = run_verification_dataloader(
                net,
                val_dl,
                verify_metrics,
                method=method,
                epsilons=verify_config["epsilons"],
                device=device,
            )

            results_per_split[i + 1] = {
                "verify_metrics": verify_metrics,
            }

    print(results_per_split)
    with open(f"{saved_models_dir}/verification_results_per_split.json", "w") as f:
        json.dump(results_per_split, f, indent=4)

    for fold, data in results_per_split.items():
        if fold not in ["verify_config", "data_config"]:
            print(f"Fold {fold}:")
            verify_metrics = data["verify_metrics"]

            print("  Verification:", end=" ")
            print(" | ".join(
                f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in verify_metrics.items()
            ))

            print()


if __name__ == "__main__":

    # declare datasets variables
    data_config = {
        "downsample_img_by": 8,
        "downsample_sequence": True,
        "imgs_per_sec": 3,
        "image_sequences": False,
        "imgs_per_sequence": 5,
        "time_spacing": 1.0,
        "regress": True,
        "dataset_root": (
            # Path(__file__).parents[4] / "srv/evenflow-data/DFKI/Dataset_4_100_traj"
            "/vol/bitbucket/svadakku/data/dfki/Dataset_4_100_traj"
        ),
        "dataset_index_needed": True,
        "num_classes": 4,
    }

    # get train/val/test splits
    test_videos, cv_splits = cross_validation(
        video_indices=list(range(100)),
        num_test_vids=10,
        num_folds=5,
        seed=42,
    )

    # create CNN
    # net = PretrainedLinear(
    #     pretrained=False, 
    #     num_classes=2 if data_config["regress"] else 10,
    #     softmax=not data_config["regress"],
    # )
    net = RobotNet(
        # num_classes=2 if data_config["regress"] else 10,
        num_classes=data_config["num_classes"],
        softmax=not data_config["regress"],
    )
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {total_params}")

    cls = net.__class__
    sig = inspect.signature(cls.__init__)
    
    arg_names = list(sig.parameters.keys())[1:]  # skip 'self'
    init_args = {}
    for name in arg_names:
        if hasattr(net, name):
            init_args[name] = getattr(net, name)
        else:
            init_args[name] = None  # or skip it, depending on your preference

    model_info = {
        'model_name': cls.__name__,
        'args': init_args
    }

    # define verfication config
    verify_config = {
        "batch_size": 10, #250
        "epsilons": [1e-1, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], #, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1
        "methods": ["ibp"],
        "model_info": model_info,
        "saved_model_dir": "/data2/svadakku/evenflow/dfki-robots/dfki/saved_models/2025-08-04_08:19:28/",
        "original_dataset_index_needed": True,
    }

    verify(net, data_config, verify_config, cv_splits)
