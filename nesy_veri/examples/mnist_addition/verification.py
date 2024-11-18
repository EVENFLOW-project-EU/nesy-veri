import os
import torch
import numpy as np
from pathlib import Path
from rich.progress import track
from torch.utils.data import Dataset
from auto_LiRPA import (
    BoundedModule,
    BoundedTensor,
    PerturbationLpNorm,
    register_custom_op,
)

from nesy_veri.utils import NetworksPlusCircuit
from nesy_veri.custom_ops import CustomBoundSoftmax, CustomConcat
from nesy_veri.examples.mnist_addition.mnist_utils import (
    AdditionDataset,
    get_sdds_for_sums,
)
from nesy_veri.examples.mnist_addition.network_training import (
    get_mnist_network,
)


def get_correctly_classified_examples(
    test_dataset: Dataset,
    net_and_circuit_per_sum: dict[int, NetworksPlusCircuit],
    results_path: os.PathLike,
    softmax: bool,
):
    print()
    filename = f"correctly_classified_imgs{'_softmax' if softmax else ''}.csv"
    correct_images_path = results_path / filename  # type: ignore

    # if the list has already been generated just load it
    if os.path.exists(correct_images_path):
        with open(correct_images_path, "r") as file:
            return list(map(int, file.read().split(",")))

    correctly_predicted_idxs = []
    for idx, (image_pair, label) in enumerate(test_dataset):
        pred_per_sum = {
            sum_: net_plus_circuit(image_pair).item()
            for sum_, net_plus_circuit in net_and_circuit_per_sum.items()
        }

        highest_pred = max(pred_per_sum, key=pred_per_sum.get)  # type: ignore
        if highest_pred == label:
            correctly_predicted_idxs.append(idx)

    # write to file for reading next time
    with open(correct_images_path, "w") as file:
        file.write(",".join(map(str, correctly_predicted_idxs)))

    return correctly_predicted_idxs


if __name__ == "__main__":
    # get trained CNN
    softmax = True
    model_path = (
        Path(__file__).parent
        / f"checkpoints/trained_model{'_softmax' if softmax else ''}.pth"
    )
    mnist_cnn = get_mnist_network(model_path=model_path, softmax=softmax)

    # get the dataset (this is copied from the DeepProbLog experiments)
    test_dataset = AdditionDataset(subset="test")

    # get an SDD circuit for each MNIST addition sum
    sdd_per_sum = get_sdds_for_sums()

    # for each sum, get a network+circuit module
    # these will be used both for inference and for bound propagation
    net_and_circuit_per_sum = {
        sum_: NetworksPlusCircuit(
            networks=[mnist_cnn, mnist_cnn],
            circuit=sdd_,
            softmax_net_outputs=[not softmax, not softmax],
            parse_to_native=True,
        )
        for sum_, sdd_ in sdd_per_sum.items()
    }

    # get the dataset examples that were classified correctly
    # only there we perform verification for now
    results_path = Path(__file__).parent / "results"
    correctly_classified_idxs = get_correctly_classified_examples(
        test_dataset, net_and_circuit_per_sum, results_path, softmax
    )

    # let auto-LiRPA know I want to use the custom operators for bounding
    # softmax and concatenation
    register_custom_op("onnx::Softmax", CustomBoundSoftmax)
    register_custom_op("onnx::Concat", CustomConcat)

    # construct bounded module for each of the 19 network+circuit graphs
    bounded_module_per_sum = {
        sum_: BoundedModule(
            net_plus_circuit,
            torch.empty_like(test_dataset[0][0]),
            verbose=False,
        )
        for sum_, net_plus_circuit in net_and_circuit_per_sum.items()
    }

    # check what happens for several epsilons
    for epsilon in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:

        num_samples_checked = 0
        num_samples_robust = 0

        for idx in track(correctly_classified_idxs):
            input_imgs, sum_label = test_dataset[idx]

            # create perturbed input
            # TODO: should this be outside of this for loop (i.e. one ptb per epsilon)
            ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
            ptb_input = BoundedTensor(input_imgs, ptb)

            bounds_per_sum = {
                sum_: [
                    bound.item()
                    for bound in bounded_module.compute_bounds(
                        x=ptb_input, method="IBP"
                    )
                ]
                for sum_, bounded_module in bounded_module_per_sum.items()
            }

            # an example is verifiably robust if the upper bounds of all wrong classes
            # are lower than the lower bound of the correct class
            wrong_upper_bounds = [
                lb for sum_, (lb, _) in bounds_per_sum.items() if sum_ != sum_label
            ]
            correct_lower_bound = bounds_per_sum[sum_label][0]
            robust = all(ub < correct_lower_bound for ub in wrong_upper_bounds)

            num_samples_checked += 1
            num_samples_robust += robust

        print(
            f"Epsilon: {epsilon:<15}",
            f"#total: {len(test_dataset)}, \t ",
            f"#correct: {len(correctly_classified_idxs)}, {round(((len(correctly_classified_idxs) / len(test_dataset))*100), 2)}% \t ",
            f"#robust correct: {num_samples_robust}, {round(((num_samples_robust / len(test_dataset))*100), 2)}% ",
        )
