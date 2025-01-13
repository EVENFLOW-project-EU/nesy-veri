import torch
import numpy as np
from pathlib import Path
from rich.progress import track
from auto_LiRPA import (
    BoundedModule,
    BoundedTensor,
    PerturbationLpNorm,
    register_custom_op,
)

from nesy_veri.utils import NetworksPlusCircuit, example_is_robust
from nesy_veri.custom_ops import CustomBoundSoftmax, CustomConcat
from nesy_veri.examples.mnist_addition.network_training import get_mnist_network
from nesy_veri.examples.mnist_addition.mnist_utils import (
    MultiDigitAdditionDataset,
    get_correctly_classified_examples,
)


def get_bounded_modules_and_samples_to_verify(
    num_digits: int, test_dataset: MultiDigitAdditionDataset
):
    model_path = (
        Path(__file__).parent
        / "checkpoints/model_checkpoints/trained_model_softmax.pth"
    )
    mnist_cnn = get_mnist_network(model_path=model_path)

    # for each sum, get a network+circuit module
    # these will be used both for inference and for bound propagation
    net_and_circuit_per_sum = {
        sum_: NetworksPlusCircuit(
            networks=[mnist_cnn] * num_digits,
            circuit=sdd_,
            categorical_idxs=[x + 1 for x in range(num_digits * 10)],
            parse_to_native=True,
        )
        for sum_, sdd_ in test_dataset.sdd_per_sum.items()
    }

    # get the dataset examples that were classified correctly
    # only there we perform verification for now
    results_path = Path(__file__).parent / "checkpoints/correctly_classified_examples"
    correctly_classified_idxs = get_correctly_classified_examples(
        test_dataset, net_and_circuit_per_sum, results_path, num_digits
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
            verbose=True,
            bound_opts={
                'conv_mode': 'matrix', 
                'optimize_bound_args': {
                        'iteration': 20,  # Adjust based on your needs
                        'beta': False  # Disable beta optimization if needed
                    }
            }
        )
        for sum_, net_plus_circuit in net_and_circuit_per_sum.items()
    }

    return bounded_module_per_sum, correctly_classified_idxs


if __name__ == "__main__":

    # declare number of MNIST digits for this experiment
    for num_digits in [1]: #[2, 3]

        # get the dataset for this number of digits
        test_dataset = MultiDigitAdditionDataset(train=False, num_digits=num_digits)

        # get a bounded version of the network+circuit structre for each sum
        # also get the indices that were classified correctly and so should be verified
        (
            bounded_module_per_sum,
            correctly_classified_idxs,
        ) = get_bounded_modules_and_samples_to_verify(num_digits, test_dataset)

        # check what happens for several epsilons
        for epsilon in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:

            num_samples_checked = 0
            num_samples_robust = 0
            

            for method in ["forward", "CROWN", "IBP+CROWN", "IBP"]:
                
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
                                x=ptb_input, method=method,
                            )
                        ]
                        for sum_, bounded_module in bounded_module_per_sum.items()
                    }

                    num_samples_checked += 1
                    num_samples_robust += example_is_robust(bounds_per_sum, sum_label)

                print(
                    f"Method: {method}, \t "
                    f"Epsilon: {epsilon:<15}",
                    f"#total: {len(test_dataset)}, \t ",
                    f"#correct: {len(correctly_classified_idxs)}, {round(((len(correctly_classified_idxs) / len(test_dataset))*100), 2)}% \t ",
                    f"#robust correct: {num_samples_robust}, {round(((num_samples_robust / len(test_dataset))*100), 2)}% ",
                )