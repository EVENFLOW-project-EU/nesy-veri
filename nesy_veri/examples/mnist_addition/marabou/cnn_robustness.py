import torch
import numpy as np
from time import time
from torch import nn
from pathlib import Path
from collections import Counter
from torchvision import transforms
from torchvision.datasets import MNIST
from maraboupy import Marabou
from maraboupy.Marabou import read_onnx
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


class MNIST_Net(nn.Module):
    def __init__(self, softmax=False, dense_input_size=16 * 4 * 4):
        super(MNIST_Net, self).__init__()

        self.size = dense_input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.size, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.Softmax(dim=-1) if softmax else nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        return x


def evaluateLocalRobustness(net, input, epsilon, verbose):
    inputVars = None
    inputVars = net.inputVars[0]
    if inputVars.shape != input_img.shape:
        raise RuntimeError(
            "Input shape of the model should be same as the input shape\n input shape of the model: {0}, shape of the input: {1}".format(
                inputVars.shape, input.shape
            )
        )

    # Add constratins to all input nodes
    flattenInputVars = inputVars.flatten()
    flattenInput = input_img.flatten()
    for i in range(len(flattenInput)):
        net.setLowerBound(flattenInputVars[i], flattenInput[i] - epsilon)
        net.setUpperBound(flattenInputVars[i], flattenInput[i] + epsilon)

    outputStartIndex = net.outputVars[0][0][0]
    outputLayerSize = len(net.outputVars[0][0])
    net.evaluateLocalRobustness

    # loop for all of output classes except for original class
    for classIndex in range(outputLayerSize):
        if classIndex == label:
            continue
        net.addInequality(
            [outputStartIndex + label, outputStartIndex + classIndex], [-1, 1], 0
        )

    # Call to Marabou solver (should be SAT)
    options = Marabou.createOptions(verbosity=verbose)
    start = time()
    exitCode, vals, stats = net.solve(options=options)
    end = time()
    solve_time = end - start

    return exitCode, solve_time


if __name__ == "__main__":
    dataset = MNIST(
        root="data/",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    )

    mnist_cnn = MNIST_Net(softmax=False)
    mnist_cnn.load_state_dict(
        torch.load(
            Path(__file__).parent.parent
            / f"checkpoints/model_checkpoints/trained_model_2_epochs_softmax.pth",
            weights_only=True,
        )
    )
    mnist_cnn.eval()
    save_path = Path(__file__).parent / "onnx_graphs/mnist/cnn_relu.onnx"

    # lirpa net
    bounded_cnn = BoundedModule(
        mnist_cnn,
        torch.empty_like(dataset[0][0].unsqueeze(0)),
        verbose=False,
    )

    print(mnist_cnn)

    results = {}
    for epsilon in [0.001]:
        results[epsilon] = {}
        for idx, (input_img, label) in enumerate(dataset):
            input_img = input_img.unsqueeze(0)

            if idx >= 100:
                continue

            # create perturbed input
            ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
            ptb_input = BoundedTensor(input_img, ptb)
            start = time()
            lb, ub = bounded_cnn.compute_bounds(x=ptb_input, method="IBP")
            end = time()

            # an example is robust if the upper bounds of all wrong classes
            # are lower than the lower bound of the correct class
            is_robust = all(
                [ub[0][c].item() < lb[0][label].item() for c in range(10) if c != label]
            )

            # marabou net
            net = read_onnx(save_path)
            exitCode, solve_time = evaluateLocalRobustness(
                net, input_img, epsilon, verbose=0
            )

            results[epsilon][idx] = {
                "auto_LiRPA": is_robust,
                "prop_time": end - start,
                "Marabou": exitCode,
                "solve_time": solve_time,
            }

        assert not (exitCode == "unsat" and is_robust)
    
    solve_times = [s["solve_time"] for s in results[0.001].values()]
    print(sum(solve_times) / len(solve_times))