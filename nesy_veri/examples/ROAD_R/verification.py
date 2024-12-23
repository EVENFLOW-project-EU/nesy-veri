import torch
import numpy as np
from torch import nn
from pathlib import Path
from auto_LiRPA import (
    BoundedModule,
    BoundedTensor,
    PerturbationLpNorm,
    register_custom_op,
)

from nesy_veri.examples.ROAD_R.network_training import get_road_networks, get_road_resnets
from nesy_veri.examples.ROAD_R.road_utils import ROADRPropositional, get_road_constraint
from nesy_veri.custom_ops import CustomBoundSoftmax, CustomConcat
from nesy_veri.utils import NetworksPlusCircuit


if __name__ == "__main__":
    test_dataset = ROADRPropositional(
        dataset_path=Path(__file__).parents[-2] / "nmanginas/road/dataset",
        train=False,
        label_level="both",
        sample_every_n=50,
    )

    object_net, action_net = get_road_resnets()
    object_net.fc = nn.Sequential(object_net.fc, nn.Sigmoid())
    # action_net.fc = nn.Sequential(action_net.fc, nn.Sigmoid())

    test_input = test_dataset[0][0].unsqueeze(0)
    epsilon = 0.00001
    ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
    ptb_input = BoundedTensor(test_input, ptb)
    bounded = BoundedModule(
        object_net,
        torch.empty_like(test_input),
        verbose=True,
    )

    from auto_LiRPA.operators.pooling import bound_backward

    bounded.compute_bounds(x=ptb_input, method="Forward+Backward")

    # nets_and_circuit = NetworksPlusCircuit(
    #     networks=[object_net, action_net],
    #     circuit=get_road_constraint(),
    #     categorical_idxs=[3, 4],
    #     parse_to_native=True,
    # )


    # register_custom_op("onnx::Softmax", CustomBoundSoftmax)
    # register_custom_op("onnx::Concat", CustomConcat)

    # test_input = torch.stack([test_dataset[0][0]] * 2)
    # torch.onnx.export(nets_and_circuit, test_input, "pls.onnx")
    # bounded = BoundedModule(
    #     nets_and_circuit,
    #     torch.empty_like(test_input),
    #     verbose=True,
    # )

    # epsilon = 0.00001
    # ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
    # ptb_input = BoundedTensor(test_input, ptb)

    # bounded.compute_bounds(x=ptb_input, method="IBP")