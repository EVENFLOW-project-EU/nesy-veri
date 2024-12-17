import torch
import operator
import numpy as np
import gurobipy as gp
from time import time
from pathlib import Path
from functools import reduce
from pysdd.sdd import SddNode
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


def sdd_to_gurobi_model(
    node: SddNode,
    bounds: dict[int, list[float]],
    categorical_groups: list[list[int]],
):
    # create environment and model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    model, categorical_values = (
        gp.Model(env=env),
        set(value for group in categorical_groups for value in group),
    )
    variable_vars = {
        key: model.addVar(
            lb=lb, ub=ub, vtype=gp.GRB.CONTINUOUS, name="literal:{}".format(key)
        )
        for key, (lb, ub) in bounds.items()
    }

    for group in categorical_groups:
        model.addConstr(
            reduce(operator.add, [variable_vars[var] for var in group]) == 1
        )

    def depth_first_search(node: SddNode):
        if node.is_true():
            # print("adding true leaf node {}".format(node.id))
            return 1
        elif node.is_false():
            # print("adding false leaf node {}".format(node.id))
            return 0
        elif node.is_literal():
            literal_id = abs(node.literal)
            if node.literal > 0:
                var = model.addVar(
                    vtype=gp.GRB.CONTINUOUS, name="node:{}".format(node.id)
                )
                model.addConstr(var == variable_vars[literal_id])
                # print("added positive literal node {}".format(node.literal))
                return var
            else:
                # Negative literals of categorical variables always get 1
                if literal_id in categorical_values:
                    # print(
                    #     "added negative literal node {} with value 1 (categorical)".format(
                    #         node.literal
                    #     )
                    # )
                    return 1

                var = model.addVar(
                    vtype=gp.GRB.CONTINUOUS, name="node:{}".format(node.id)
                )
                model.addConstr(var == 1 - variable_vars[literal_id])
                # print("added negative literal node {} (binary)".format(node.literal))
                return var

        elif node.is_decision():
            var = model.addVar(vtype=gp.GRB.CONTINUOUS, name="var_{}".format(node.id))
            model.addConstr(
                var
                == reduce(
                    operator.add,
                    [
                        operator.mul(depth_first_search(prime), depth_first_search(sub))
                        for prime, sub in node.elements()
                    ],
                )
            )
            # print("added decision node: {}".format(node.id))
            return var
        else:
            raise RuntimeError("what is happening?")

    optimization_target = depth_first_search(node)

    return model, optimization_target


def get_bounds_per_sum_gurobi(
    bounded_network: BoundedModule,
    input_imgs: torch.Tensor,
    epsilon: float,
    sdd_per_sum: dict[int, SddNode],
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

    bounds_per_sum = {}
    for curr_sum, curr_sdd in sdd_per_sum.items():
        # calculate the minimum and maximum probability for the correct sum
        model, var = sdd_to_gurobi_model(
            curr_sdd,
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

        bounds_per_sum[curr_sum] = [minimum, maximum]

    return bounds_per_sum

if __name__ == "__main__":
    torch.set_num_threads(1)

    # get trained CNN
    model_path = (
        Path(__file__).parent.parent
        / "checkpoints/model_checkpoints/trained_modelsoftmax.pth"
    )
    mnist_cnn = get_mnist_network(model_path=model_path)

    # TODO: DOES THIS NEED TO BE REPEATED?
    # let auto-LiRPA know I want to use the custom operators for bounding
    # softmax and concatenation
    register_custom_op("onnx::Softmax", CustomBoundSoftmax)
    register_custom_op("onnx::Concat", CustomConcat)

    # declare number of MNIST digits for this experiment
    for num_digits in [2, 3, 4, 5]:
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
                categorical_idxs=[x + 1 for x in range(num_digits*10)],
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

        epsilon = 0.005

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

