import torch
import statistics
import numpy as np
import gurobipy as gp
from time import time
from pathlib import Path
from pysdd.sdd import SddNode
from rich.progress import track
from auto_LiRPA import (
    BoundedModule,
    BoundedTensor,
    PerturbationLpNorm,
    register_custom_op,
)

from nesy_veri.utils import NetworksPlusCircuit
from nesy_veri.custom_ops import CustomBoundSoftmax, CustomConcat
from nesy_veri.examples.mnist_addition.network_training import get_mnist_network
from nesy_veri.examples.mnist_addition.mnist_utils import MultiDigitAdditionDataset
from nesy_veri.examples.mnist_addition.gurobi_comparison.gurobi_full_comparison import (
    sdd_to_gurobi_model,
)


def get_bounds_for_sum_gurobi(
    bounded_network: BoundedModule,
    input_imgs: torch.Tensor,
    epsilon: float,
    correct_sum_sdd: SddNode,
    num_digits: int,
):

    images_bounds = {}
    ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
    for d in range(num_digits):
        # get perturbed input for this image
        ptb_input = BoundedTensor(input_imgs[d].unsqueeze(0), ptb)

        # compute lower/upper bounds for this image
        lb, ub = bounded_network.compute_bounds(x=(ptb_input,), method="IBP")

        # append this image bounds to a dict to be used by the SDD -> Gurobi translation
        img_bounds = {
            d * 10 + i + 1: [lb[0, i].item(), ub[0, i].item()] for i in range(10)
        }
        images_bounds.update(img_bounds)

    # for d, bounds in images_bounds.items():
    #     print(f"Image {(d-1) // 10}, Digit {(d-1) % 10} is in [{bounds[0]:.3f}, {bounds[1]:.3f}]")
    #     if d % 10 == 0:
    #         print()

    # calculate the minimum and maximum probability for the correct sum
    model, var = sdd_to_gurobi_model(
        correct_sum_sdd,
        bounds=images_bounds,
        categorical_groups=[
            list(range(i * 10 + 1, (i + 1) * 10 + 1)) for i in range(num_digits)
        ],
    )

    # find minimum
    model.setObjective(var, sense=gp.GRB.MINIMIZE)
    model.optimize()
    minimum = model.ObjVal

    # find maximum
    model.setObjective(var, sense=gp.GRB.MAXIMIZE)
    model.optimize()
    maximum = model.ObjVal

    return minimum, maximum


if __name__ == "__main__":
    # get trained CNN
    softmax = True
    model_path = (
        Path(__file__).parent.parent
        / f"checkpoints/model_checkpoints/trained_model_1_epoch{'_softmax' if softmax else ''}.pth"
    )
    mnist_cnn = get_mnist_network(model_path=model_path, softmax=softmax, num_epochs=1)

    # TODO: DOES THIS NEED TO BE REPEATED?
    # let auto-LiRPA know I want to use the custom operators for bounding
    # softmax and concatenation
    register_custom_op("onnx::Softmax", CustomBoundSoftmax)
    register_custom_op("onnx::Concat", CustomConcat)

    # declare number of MNIST digits for this experiment
    for num_digits in [2, 3, 4, 5]:
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

        epsilon = 0.001
        timings_e2e = []
        timings_grb = []
        num_samples_robust_e2e = 0
        num_samples_robust_grb = 0
        lower_bounds_e2e, upper_bounds_e2e = [], []
        lower_bounds_grb, upper_bounds_grb = [], []
        grb_problematic_samples = 0

        for input_imgs, sum_label in track(test_dataset):
            # create perturbed input
            ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
            ptb_input = BoundedTensor(input_imgs, ptb)

            e2e_start = time()
            correct_sum_lb, correct_sum_ub = [
                x.item()
                for x in bounded_module_per_sum[sum_label].compute_bounds(
                    x=ptb_input, method="IBP"
                )
            ]
            e2e_end = time()

            try:
                # get bounds per sum for the Gurobi thing
                grb_start = time()
                correct_sum_min, correct_sum_max = get_bounds_for_sum_gurobi(
                    bounded_network=bounded_cnn,
                    input_imgs=input_imgs,
                    epsilon=epsilon,
                    correct_sum_sdd=test_dataset.sdd_per_sum[sum_label],
                    num_digits=num_digits,
                )
                grb_end = time()

                # update timing info
                timings_e2e.append(e2e_end - e2e_start)
                timings_grb.append(grb_end - grb_start)

                # update robustness lists
                num_samples_robust_e2e += (correct_sum_lb > 0.5)
                num_samples_robust_grb += (correct_sum_min > 0.5)

                # update bound lists
                lower_bounds_e2e.append(correct_sum_lb)
                upper_bounds_e2e.append(min(correct_sum_ub, 1))
                lower_bounds_grb.append(correct_sum_min)
                upper_bounds_grb.append(min(correct_sum_max, 1))

            except Exception as e:
                grb_problematic_samples += 1

        print(
            f"E2E - ",
            f"mean lower: {statistics.mean(lower_bounds_e2e):.5f}, ",
            f"mean upper: {statistics.mean(upper_bounds_e2e):.5f}, ",
            f"mean diff: {statistics.mean([upper_bounds_e2e[i] - lower_bounds_e2e[i] for i in range(len(lower_bounds_e2e))]):.5f}",
        )
        print(
            f"GRB - ",
            f"mean lower: {statistics.mean(lower_bounds_grb):.5f}, ",
            f"mean upper: {statistics.mean(upper_bounds_grb):.5f}, ",
            f"mean diff: {statistics.mean([upper_bounds_grb[i] - lower_bounds_grb[i] for i in range(len(lower_bounds_grb))]):.5f}",
        )

        print(
            f"#digits: {num_digits:<10}",
            f"epsilon: {epsilon}  ",
            f"average E2E time: {(sum(timings_e2e) / len(timings_e2e)):.4f}   ",
            f"average GRB time: {(sum(timings_grb) / len(timings_grb)):.4f}   ",
            f"robustness(E2E): {round(((num_samples_robust_e2e / len(test_dataset))*100), 2)}%   ",
            f"robustness(GRB): {round(((num_samples_robust_grb / len(test_dataset))*100), 2)}%   ",
            f"({grb_problematic_samples} problematic samples)",
        )
