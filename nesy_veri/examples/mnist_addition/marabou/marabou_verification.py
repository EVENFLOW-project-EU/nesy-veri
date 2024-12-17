import os
import torch
from pathlib import Path
from pysdd.sdd import SddNode
from rich.progress import track

from nesy_veri.utils import NetworksPlusCircuit
from nesy_veri.examples.mnist_addition.network_training import get_mnist_network
from nesy_veri.examples.mnist_addition.marabou.custom_marabou.custom_read_onnx import (
    custom_read_onnx,
)
from nesy_veri.examples.mnist_addition.mnist_utils import (
    MultiDigitAdditionDataset,
    get_correctly_classified_examples,
)


def create_and_save_onnx(
    softmax: bool,
    num_digits: int,
    save_dir: os.PathLike,
    sdd_per_sum: dict[int, SddNode],
    test_input: torch.Tensor,
):
    # get trained CNN
    this_dir = Path(__file__).parent
    model_path = (
        this_dir.parent
        / f"checkpoints/model_checkpoints/trained_model{'_softmax' if softmax else ''}.pth"
    )
    mnist_cnn = get_mnist_network(model_path=model_path, softmax=softmax)

    # for each sum, get a network+circuit module
    # these will be used both for inference and for bound propagation
    net_and_circuit_per_sum = {
        sum_: NetworksPlusCircuit(
            networks=[mnist_cnn] * num_digits,
            circuit=sdd_,
            categorical_idxs=[x + 1 for x in range(num_digits*10)],
            parse_to_native=True,
        )
        for sum_, sdd_ in sdd_per_sum.items()
    }

    for sum_, net_plus_circuit in net_and_circuit_per_sum.items():
        torch.onnx.export(
            net_plus_circuit,
            test_input,
            str(save_dir / f"mnist_{sum_}.onnx"),
        )

    return net_and_circuit_per_sum


if __name__ == "__main__":
    # set global parameters
    softmax = True
    num_digits = 2
    save_onnx_dir = Path(__file__).parent / f"onnx_graphs/mnist/{num_digits}_digits"

    # get the dataset for this number of digits
    test_dataset = MultiDigitAdditionDataset(train=False, num_digits=num_digits)

    # get the network+circuit graph for all sums and save them all in .onnx files
    net_and_circuit_per_sum = create_and_save_onnx(
        softmax=softmax,
        num_digits=num_digits,
        save_dir=save_onnx_dir,
        sdd_per_sum=test_dataset.sdd_per_sum,
        test_input=torch.empty(
            (num_digits, 1, 28, 28),
        ),
    )

    # get the dataset examples that were classified correctly
    # only there we perform verification for now
    results_path = (
        Path(__file__).parent.parent / "checkpoints/correctly_classified_examples"
    )
    correctly_classified_idxs = get_correctly_classified_examples(
        test_dataset, net_and_circuit_per_sum, results_path, softmax, num_digits
    )

    # create a Marabou network for each of the ONNX graphs saved
    # I'm using a custom version of the Marabou.read_onnx function
    # for more information on why and how read this directory's README
    marabou_nets = {
        sum_: custom_read_onnx(save_onnx_dir / f"mnist_{sum_}.onnx")
        for sum_ in range(num_digits * 9 + 1)
    }

    for idx in track(correctly_classified_idxs):
        input_imgs, sum_label = test_dataset[idx]

        # forward all networks with ONNX, not using Marabou
        for curr_sum, net in marabou_nets.items():
            print(
                f"p(sum = {curr_sum}) = {net.evaluateWithoutMarabou([input_imgs.numpy()])[0]:.3f} (without Marabou)"
            )

        # # PERFORMING A ROBUSTNESS CHECK CRASHES IN THE FIRST EXAMPLE
        # for curr_sum, net in marabou_nets.items():
        #     # Get the input and output variable numbers; [0] since first dimension is batch size
        #     inputVars = net.inputVars[0]
        #     outputVars = net.outputVars[0]

        #     # Setup a local robustness query
        #     epsilon = 0.001
        #     for d in range(inputVars.shape[0]):
        #         for c in range(inputVars.shape[1]):
        #             for h in range(inputVars.shape[2]):
        #                 for w in range(inputVars.shape[3]):
        #                     net.setLowerBound(
        #                         inputVars[d][c][h][w],
        #                         input_imgs[d][c][h][w].item() - epsilon,
        #                     )
        #                     net.setUpperBound(
        #                         inputVars[d][c][h][w],
        #                         input_imgs[d][c][h][w].item() + epsilon,
        #                     )

        #     # Set output bounds
        #     net.setLowerBound(outputVars.item(), 0.5)

        #     # Call to Marabou solver (should be SAT)
        #     exitCode, vals, stats = net.solve()
        #     assert exitCode == "sat"
        #     assert len(vals) > 0

        break