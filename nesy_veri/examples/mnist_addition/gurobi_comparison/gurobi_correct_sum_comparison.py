import os
import json
import torch
import random
import argparse
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
    nn_veri_time = 0
    for d in range(num_digits):
        # get perturbed input for this image
        ptb_input = BoundedTensor(input_imgs[d].unsqueeze(0), ptb)

        # compute lower/upper bounds for this image
        start = time()
        lb, ub = bounded_network.compute_bounds(x=(ptb_input,), method="IBP")
        end = time()
        nn_veri_time += end - start

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
    start = time()
    model, var = sdd_to_gurobi_model(
        correct_sum_sdd,
        bounds=images_bounds,
        categorical_groups=[
            list(range(i * 10 + 1, (i + 1) * 10 + 1)) for i in range(num_digits)
        ],
    )
    construction_time = time() - start

    start = time()
    # find minimum
    model.setObjective(var, sense=gp.GRB.MINIMIZE)
    model.optimize()
    minimum = model.ObjVal

    # find maximum
    model.setObjective(var, sense=gp.GRB.MAXIMIZE)
    model.optimize()
    maximum = model.ObjVal
    end = time()
    solve_time = end - start

    model.close()

    return minimum, maximum, nn_veri_time + solve_time, construction_time


def verify_experiment(digits: list[int], num_epochs: int, epsilon: float):
    print(f"Epsilon: {epsilon}")
    print(f"Digits: {digits}")
    print(f"Number of epochs: {num_epochs}")

    log_file = f"results_{epsilon}_{digits}_{num_epochs}"

    if os.path.exists(f"results/{log_file}.json"):
        raise RuntimeError("log file exists, bye bye!")

    torch.set_num_threads(1)
    # gp.setParam("Threads", 1)

    # get trained CNN
    model_path = (
        Path(__file__).parent.parent
        / f"checkpoints/model_checkpoints/trained_model_{num_epochs}_epochs_softmax.pth"
    )
    mnist_cnn = get_mnist_network(model_path=model_path, num_epochs=num_epochs)

    # TODO: DOES THIS NEED TO BE REPEATED?
    # let auto-LiRPA know I want to use the custom operators for bounding
    # softmax and concatenation
    register_custom_op("onnx::Softmax", CustomBoundSoftmax)
    register_custom_op("onnx::Concat", CustomConcat)

    data = {}

    def log():
        with open(f"results/{log_file}.json", "w") as f:
            json.dump(data, f, indent=2)

    # declare number of MNIST digits for this experiment
    for num_digits in digits:
        data[num_digits] = {}
        test_dataset = MultiDigitAdditionDataset(train=False, num_digits=num_digits)

        # for each sum, get a network+circuit module
        # these will be used both for inference and for bound propagation
        start = time()
        net_and_circuit_per_sum = {
            sum_: NetworksPlusCircuit(
                networks=[mnist_cnn] * num_digits,
                circuit=sdd_,
                categorical_idxs=[x + 1 for x in range(num_digits * 10)],
                parse_to_native=True,
            )
            for sum_, sdd_ in test_dataset.sdd_per_sum.items()
        }
        end = time()
        data[num_digits]["Network+Circuit construction time"] = end - start
        log()

        data[num_digits]["#nodes per sum"] = {
            sum_: sdd_.size() for sum_, sdd_ in test_dataset.sdd_per_sum.items()
        }
        log()

        # construct bounded module for each of the network+circuit graphs
        bounded_module_per_sum = {}
        data[num_digits]["Bounded Module construction time"] = {}
        for sum_, net_plus_circuit in net_and_circuit_per_sum.items():
            start = time()
            bounded_module_per_sum[sum_] = BoundedModule(
                net_plus_circuit,
                torch.empty_like(test_dataset[0][0]),
                verbose=False,
            )
            end = time()
            data[num_digits]["Bounded Module construction time"][sum_] = end - start
            log()

        # get bounded NN
        start = time()
        bounded_cnn = BoundedModule(
            mnist_cnn,
            torch.empty_like(test_dataset[0][0][0].unsqueeze(0)),
            verbose=False,
        )
        end = time()
        data[num_digits]["Bounded CNN construction time"] = end - start
        log()

        timings_e2e = []
        timings_grb = []
        num_samples_robust_e2e = 0
        num_samples_robust_grb = 0
        lower_bounds_e2e, upper_bounds_e2e = [], []
        lower_bounds_grb, upper_bounds_grb = [], []
        grb_problematic_samples = 0

        data[num_digits]["E2E"] = {}
        data[num_digits]["GRB"] = {}
        data[num_digits]["#problematic samples"] = 0

        for i, (input_imgs, sum_label) in track(
            enumerate(test_dataset), total=len(test_dataset)
        ):
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
                correct_sum_min, correct_sum_max, solve_time, construction_time = (
                    get_bounds_for_sum_gurobi(
                        bounded_network=bounded_cnn,
                        input_imgs=input_imgs,
                        epsilon=epsilon,
                        correct_sum_sdd=test_dataset.sdd_per_sum[sum_label],
                        num_digits=num_digits,
                    )
                )

                data[num_digits]["E2E"][i] = {
                    "sum_label": sum_label,
                    "runtime": e2e_end - e2e_start,
                    "lower_bound": max(correct_sum_lb, 0),
                    "upper_bound": min(correct_sum_ub, 1),
                    "robust": correct_sum_lb > 0.5,
                }
                data[num_digits]["GRB"][i] = {
                    "sum_label": sum_label,
                    "runtime": solve_time,
                    "construction_time": construction_time,
                    "lower_bound": correct_sum_min,
                    "upper_bound": correct_sum_max,
                    "robust": correct_sum_min > 0.5,
                }
                log()

                # update timing info
                timings_e2e.append(e2e_end - e2e_start)
                timings_grb.append(solve_time)

                # update robustness lists
                num_samples_robust_e2e += correct_sum_lb > 0.5
                num_samples_robust_grb += correct_sum_min > 0.5

                # update bound lists
                lower_bounds_e2e.append(correct_sum_lb)
                upper_bounds_e2e.append(min(correct_sum_ub, 1))
                lower_bounds_grb.append(correct_sum_min)
                upper_bounds_grb.append(min(correct_sum_max, 1))

            except Exception as e:
                grb_problematic_samples += 1
                data[num_digits]["#problematic samples"] += 1
                log()

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


if __name__ == "__main__":
    # the way to call this is:

    # python nesy_veri/examples/mnist_addition/gurobi_comparison/gurobi_correct_sum_comparison.py --digits 2 3 4 5 --num_epochs 10 --epsilon 0.01

    parser = argparse.ArgumentParser(description="Verify experiment settings.")

    parser.add_argument(
        "--digits",
        type=int,
        nargs="+",
        required=True,
        help="List of digits to verify (e.g., 1 2 3).",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        required=True,
        help="Number of epochs for the experiment.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="Tolerance value (epsilon).",
    )

    args = parser.parse_args()

    # Call the function with parsed arguments
    verify_experiment(
        digits=args.digits,
        num_epochs=args.num_epochs,
        epsilon=args.epsilon,
    )
