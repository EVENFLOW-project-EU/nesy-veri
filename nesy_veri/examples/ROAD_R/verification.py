import torch
import numpy as np
from pathlib import Path
from auto_LiRPA import (
    BoundedModule,
    BoundedTensor,
    PerturbationLpNorm,
    register_custom_op,
)
from rich.progress import track
from torch.utils.data import random_split

from nesy_veri.examples.ROAD_R.network_training import get_road_network
from nesy_veri.examples.ROAD_R.road_utils import ROADRPropositional, get_road_constraint
from nesy_veri.custom_ops import CustomBoundSoftmax, CustomConcat
from nesy_veri.utils import NetworksPlusCircuit


if __name__ == "__main__":
    sample_every_n = 24
    downsample_img_by = 4
    num_epochs_objects = 20
    num_epochs_actions = 10
    model_dir = Path(__file__).parent / "checkpoints/model_checkpoints"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    object_net = get_road_network(
        model_dir,
        "objects",
        sample_every_n,
        downsample_img_by,
        device,
        num_epochs_objects,
    )

    action_net = get_road_network(
        model_dir,
        "actions",
        sample_every_n,
        downsample_img_by,
        device,
        num_epochs_actions,
    )

    gen = torch.Generator()
    gen.manual_seed(0)

    dataset = ROADRPropositional(
        dataset_path=Path(__file__).parents[3] / "dataset",
        subset="all",
        label_level="both",
        sample_every_n=sample_every_n,
        downsample_img_by=downsample_img_by,
        balance_feature_dataset=False,
    )

    _, test_dataset = random_split(dataset, [0.8, 0.2], generator=gen)

    nets_and_circuit = NetworksPlusCircuit(
        networks=[object_net, action_net],
        circuit=get_road_constraint(),
        categorical_idxs=[3, 4],
        parse_to_native=True,
    )

    safe_idxs = [
        idx
        for idx, (input_img, _) in enumerate(test_dataset)
        if nets_and_circuit(torch.stack([input_img] * 2)) > 0.5
    ]

    register_custom_op("onnx::Softmax", CustomBoundSoftmax)
    register_custom_op("onnx::Concat", CustomConcat)

    test_input = test_dataset[0][0]
    # torch.onnx.export(nets_and_circuit, test_input, "full.onnx")
    bounded_module = BoundedModule(
        nets_and_circuit,
        torch.empty_like(torch.stack([test_input] * 2)),
        verbose=True,
    )
    
    for method in ["CROWN", "IBP+CROWN", "IBP"]: #"forward"
        for epsilon in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]:

            num_samples_robust = 0

            for idx in track(safe_idxs):
                input_img, labels = test_dataset[idx]

                ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
                ptb_input = BoundedTensor(torch.stack([input_img] * 2), ptb)

                lb, ub = bounded_module.compute_bounds(x=ptb_input, method=method)

                num_samples_robust += lb.item() > 0.5

            print(
                f"Epsilon: {epsilon:<15}",
                f"#total: {len(test_dataset)}, \t ",
                f"#correct: {len(safe_idxs)}, {round(((len(safe_idxs) / len(test_dataset))*100), 2)}% \t ",
                f"#robust correct: {num_samples_robust}, {round(((num_samples_robust / len(test_dataset))*100), 2)}% ",
            )
