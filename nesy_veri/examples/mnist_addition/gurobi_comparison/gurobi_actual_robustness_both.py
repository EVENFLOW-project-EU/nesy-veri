import torch
import numpy as np
from time import time
from pathlib import Path
from rich.progress import track
from auto_LiRPA import (
    BoundedModule,
    BoundedTensor,
    PerturbationLpNorm,
    register_custom_op,
)

from nesy_veri.custom_ops import CustomBoundSoftmax, CustomConcat
from nesy_veri.utils import NetworksPlusCircuit, example_is_robust
from nesy_veri.examples.mnist_addition.network_training import get_mnist_network
from nesy_veri.examples.mnist_addition.mnist_utils import MultiDigitAdditionDataset
from nesy_veri.examples.mnist_addition.gurobi_comparison.gurobi_robustness_comparison import (
    get_bounds_per_sum_gurobi,
)


if __name__ == "__main__":
    # get trained CNN
    softmax = True
    model_path = (
        Path(__file__).parent.parent
        / f"checkpoints/model_checkpoints/trained_model{'_softmax' if softmax else ''}.pth"
    )
    mnist_cnn = get_mnist_network(model_path=model_path, softmax=softmax)

    # TODO: DOES THIS NEED TO BE REPEATED?
    # let auto-LiRPA know I want to use the custom operators for bounding
    # softmax and concatenation
    register_custom_op("onnx::Softmax", CustomBoundSoftmax)
    register_custom_op("onnx::Concat", CustomConcat)

    # declare number of MNIST digits for this experiment
    for num_digits in range(2, 7):
        # get the dataset for this number of digits
        start = time()
        test_dataset = MultiDigitAdditionDataset(train=False, num_digits=num_digits)
        end = time()
        print(
            f"{num_digits}-digit SDD generation took {end-start:.4f} seconds, #sums = {len(test_dataset.sdd_per_sum.keys())}"
        )

        # for each sum, get a network+circuit module
        # these will be used both for inference and for bound propagation
        start = time()
        net_and_circuit_per_sum = {
            sum_: NetworksPlusCircuit(
                networks=[mnist_cnn] * num_digits,
                circuit=sdd_,
                softmax_net_outputs=[not softmax] * num_digits,
                parse_to_native=True,
            )
            for sum_, sdd_ in test_dataset.sdd_per_sum.items()
        }
        end = time()
        print(
            f"Constructing the Network+Circuit for all sums took {(end-start):.4f} seconds"
        )

        # construct bounded module for each of the network+circuit graphs
        start = time()
        bounded_module_per_sum = {
            sum_: BoundedModule(
                net_plus_circuit,
                torch.empty_like(test_dataset[0][0]),
                verbose=False,
            )
            for sum_, net_plus_circuit in net_and_circuit_per_sum.items()
        }
        end = time()
        print(f"Getting the BoundedModule for all sums took {(end-start):.4f} seconds")

        # get bounded NN
        start = time()
        bounded_cnn = BoundedModule(
            mnist_cnn,
            torch.empty_like(test_dataset[0][0][0].unsqueeze(0)),
            verbose=False,
        )
        end = time()
        print(f"Getting the BoundedModule for the CNN took {(end-start):.4f} seconds")

        # declare necessary metrics
        timings_e2e = []
        timings_grb = []
        num_samples_robust_e2e = 0
        num_samples_robust_grb = 0
        grb_problematic_samples = 0

        epsilon = 0.001

        for input_imgs, sum_label in track(test_dataset):

            # create perturbed input
            ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
            ptb_input = BoundedTensor(input_imgs, ptb)

            # get bounds for the end-to-end thing
            e2e_start = time()
            bounds_per_sum_e2e = {
                sum_: [
                    bound.item()
                    for bound in bounded_module.compute_bounds(
                        x=ptb_input, method="IBP"
                    )
                ]
                for sum_, bounded_module in bounded_module_per_sum.items()
            }
            e2e_end = time()

            try:
                grb_start = time()
                # get bounds per sum for the Gurobi thing
                bounds_per_sum_grb = get_bounds_per_sum_gurobi(
                    bounded_network=bounded_cnn,
                    input_imgs=input_imgs,
                    epsilon=epsilon,
                    sdd_per_sum=test_dataset.sdd_per_sum,
                    num_digits=num_digits,
                )
                grb_end = time()

                # append timings to lists
                timings_e2e.append(e2e_end - e2e_start)
                timings_grb.append(grb_end - grb_start)

                num_samples_robust_e2e += example_is_robust(
                    bounds_per_sum_e2e, sum_label
                )
                num_samples_robust_grb += example_is_robust(
                    bounds_per_sum_grb, sum_label
                )

            except Exception as e:
                grb_problematic_samples += 1
        
        print(
            f"#digits: {num_digits:<10}",
            f"time(E2E): {(sum(timings_e2e) / len(timings_e2e)):.4f}",
            f"time(GRB): {(sum(timings_grb) / len(timings_grb)):.4f}",
            f"robustness(E2E): {round(((num_samples_robust_e2e / len(test_dataset))*100), 2)}%",
            f"robustness(GRB): {round(((num_samples_robust_grb / len(test_dataset))*100), 2)}%",
            f"({grb_problematic_samples} problematic samples)",
        )

