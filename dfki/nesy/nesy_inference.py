import torch
from pathlib import Path

from dfki.nesy.nesy_data import create_nesy_inference_dataset
from dfki.network_definitions import PretrainedLinear
from dfki.network_training import cross_validation, train


if __name__ == "__main__":

    # declare datasets variables
    data_config = {
        "downsample_img_by": 8,
        "downsample_sequence": True,
        "imgs_per_sec": 1,
        "image_sequences": False,
        "imgs_per_sequence": 5,
        "time_spacing": 1.0,
        "regress": True,
        "dataset_root": (
            # Path(__file__).parents[5] / "srv/evenflow-data/DFKI/Dataset_4_100_traj"
            "/vol/bitbucket/svadakku/data/dfki/Dataset_4_100_traj"
        ),
    }

    # get train/val/test splits
    test_videos, _ = cross_validation(
        video_indices=list(range(100)),
        num_test_vids=10,
        num_folds=5,
        seed=42,
    )

    inference_data = create_nesy_inference_dataset(
        data_path=data_config["dataset_root"],
        downsample_sequence=data_config["downsample_sequence"],
        imgs_per_sec=data_config["imgs_per_sec"],
        test_video_idxs=list(test_videos),
    )

    # create CNN and load trained CNN
    net = PretrainedLinear(num_classes=2, softmax=False)
    net.load_state_dict(
        torch.load(
            Path(__file__).parent
            / "saved_models/2025-05-09_14:13:39/PretrainedLinear_split_1of5.pt",
            weights_only=True,
        )
    )

    x_mean, x_std = -6.104, 3.060
    y_mean, y_std = 17.937, 4.035

    bin_mapping = {
        "a": [float("-inf"), -1.2816],
        "b": [-1.2816, -0.8416],
        "c": [-0.8416, -0.5244],
        "d": [-0.5244, -0.2533],
        "e": [-0.2533, 0.0000],
        "f": [0.0000, 0.2533],
        "g": [0.2533, 0.5244],
        "h": [0.5244, 0.8416],
        "i": [0.8416, 1.2816],
        "j": [1.2816, float("inf")],
    }

    def find_bin(value, bins_dict):
        for name, (low, high) in bins_dict.items():
            if low <= value < high:
                return name

    # for bin_, [low, high] in bin_mapping.items():
