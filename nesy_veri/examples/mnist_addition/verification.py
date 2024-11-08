import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, mnist
from auto_LiRPA import (
    BoundedModule,
    BoundedTensor,
    PerturbationLpNorm,
    register_custom_op,
)

from nesy_veri.examples.mnist_addition.mnist_utils import get_sdds_for_sums
from nesy_veri.utils import NetworksPlusCircuit
from nesy_veri.examples.mnist_addition.network_training import (
    get_mnist_network,
)

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
    test_input = torch.stack((test_dataset[13][0], test_dataset[14][0]))

    # get an SDD circuit for each MNIST addition sum
    sdd_per_sum = get_sdds_for_sums()

    # create the network+circuit nn.Module to pass to auto-LiRPA for bound computation
    mnist_add_7 = NetworksPlusCircuit(
        networks=[mnist_cnn, mnist_cnn],
        softmax_net_outputs=[True, True],
        circuit=sdd_per_sum[7],
        parse_to_native=True,
    )
    print(mnist_add_7(test_input))

    # let auto-LiRPA know I want to use BoundSoftmax
    from nesy_veri.custom_ops import CustomBoundSoftmax, CustomConcat
    register_custom_op("onnx::Softmax", CustomBoundSoftmax)
    register_custom_op("onnx::Concat", CustomConcat)

    # construct bounded module for the network+circuit graph
    lirpa_model = BoundedModule(mnist_add_7, torch.empty_like(test_input), verbose=False)

    # check what happens for several epsilons
    for epsilon in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:
        # create perturbed input
        ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
        my_input = BoundedTensor(test_input, ptb)

        # compute bounds for the perturbed test_input
        lb, ub = lirpa_model.compute_bounds(x=my_input, method="IBP")

        print(
            f"Epsilon: {epsilon}\t\t pred = {mnist_add_7(test_input).item():.3f},  lb = {lb.item():.3f},  ub = {ub.item():.3f}"
        )
