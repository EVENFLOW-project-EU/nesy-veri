import torch
import operator
import numpy as np
from torch import nn
from pathlib import Path
from pysdd.sdd import SddNode
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from auto_LiRPA import (
    BoundedModule,
    BoundedTensor,
    PerturbationLpNorm,
    register_custom_op,
)

from nesy_veri.utils import sdd_parse_native, eval_sdd
from nesy_veri.examples.mnist_addition.network_training import (
    get_mnist_network,
)


def get_test_circuit():
    # fmt: off
    from pysdd.sdd import SddManager
    manager = SddManager(20, 0)
    (
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, 
    ) = [manager.literal(i) for i in range(1, 11)]


    # these contraints encapsulate that the different values for each digit are mutually exclusive
    # i.e. the digit *must* take a value (0-9) and if any value is true then no other value can be true
    constraints_x = ((~x0 | ~x1) & (~x0 | ~x2) & (~x0 | ~x3) & (~x0 | ~x4) & (~x0 | ~x5) & (~x0 | ~x6) & (~x0 | ~x7) & (~x0 | ~x8) & (~x0 | ~x9)
                    & (~x1 | ~x2) & (~x1 | ~x3) & (~x1 | ~x4) & (~x1 | ~x5) & (~x1 | ~x6) & (~x1 | ~x7) & (~x1 | ~x8) & (~x1 | ~x9)
                    & (~x2 | ~x3) & (~x2 | ~x4) & (~x2 | ~x5) & (~x2 | ~x6) & (~x2 | ~x7) & (~x2 | ~x8) & (~x2 | ~x9)
                    & (~x3 | ~x4) & (~x3 | ~x5) & (~x3 | ~x6) & (~x3 | ~x7) & (~x3 | ~x8) & (~x3 | ~x9)
                    & (~x4 | ~x5) & (~x4 | ~x6) & (~x4 | ~x7) & (~x4 | ~x8) & (~x4 | ~x9)
                    & (~x5 | ~x6) & (~x5 | ~x7) & (~x5 | ~x8) & (~x5 | ~x9)
                    & (~x6 | ~x7) & (~x6 | ~x8) & (~x6 | ~x9)
                    & (~x7 | ~x8) & (~x7 | ~x9)
                    & (~x8 | ~x9)
                    & (x0 | x1 | x2 | x3 | x4 | x5 | x6 | x7 | x8 | x9))
    # fmt: on

    sum6 = x6 & constraints_x
    sum6.ref()

    return sum6


class NetworkPlusCircuitMinimal(nn.Module):
    def __init__(
        self, network: torch.nn.Module, circuit: SddNode, parse_to_native: bool
    ):
        super().__init__()
        self.network = network
        self.softmax = nn.Softmax(dim=-1)
        self.circuit = circuit if not parse_to_native else sdd_parse_native(circuit)

    def forward(self, x):
        digit1_output = self.network(x)
        activation1 = self.softmax(digit1_output)
        sdd_output = eval_sdd(
            self.circuit, operator.add, operator.mul, 0, 1, activation1
        )

        return sdd_output


if __name__ == "__main__":
    # get trained CNN
    model_path = Path(__file__).parents[1] / "mnist_addition/checkpoints/trained_model.pth"
    mnist_cnn = get_mnist_network(model_path=model_path)

    # data stuff
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    test_dataset = MNIST(root="data/", train=True, download=True, transform=transform)
    test_dl = DataLoader(test_dataset, batch_size=32, shuffle=True)
    test_input = torch.unsqueeze(test_dataset[13][0], 0)

    # create the network+circuit nn.Module to pass to auto-LiRPA for bound computation
    test_nesy = NetworkPlusCircuitMinimal(
        network=mnist_cnn,
        circuit=get_test_circuit(),
        parse_to_native=True,
    )

    # let auto-LiRPA know I want to use BoundSoftmax
    from nesy_veri.custom_softmax import CustomBoundSoftmax
    register_custom_op("onnx::Softmax", CustomBoundSoftmax)

    # construct bounded module for the network+circuit graph
    lirpa_model = BoundedModule(test_nesy, torch.empty_like(test_input), verbose=False)

    # check what happens for several epsilons
    for epsilon in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:
        # create perturbed input
        ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
        my_input = BoundedTensor(test_input, ptb)

        # compute bounds for the perturbed test_input
        lb, ub = lirpa_model.compute_bounds(x=my_input, method="IBP")

        print(
            f"Epsilon: {epsilon}\t\t pred = {test_nesy(test_input).item():.3f},  lb = {lb.item():.3f},  ub = {ub.item():.3f}"
        )
