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
from nesy_veri.examples.mnist_addition.network_training import get_mnist_network
from nesy_veri.examples.mnist_addition.verification import (
    get_bounded_modules_and_samples_to_verify,
)
from nesy_veri.examples.mnist_addition.mnist_utils import (
    MultiDigitAdditionDataset,
)


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
        / f"checkpoints/model_checkpoints/trained_model{'_softmax' if softmax else ''}.pth"
    )
    mnist_cnn = get_mnist_network(model_path=model_path, softmax=softmax)

    # TODO: DOES THIS NEED TO BE REPEATED?
    # let auto-LiRPA know I want to use the custom operators for bounding
    # softmax and concatenation
    register_custom_op("onnx::Softmax", CustomBoundSoftmax)
    register_custom_op("onnx::Concat", CustomConcat)

    # declare number of MNIST digits for this experiment
    for num_digits in [4, 3, 2]:

        # get the dataset for this number of digits
        test_dataset = MultiDigitAdditionDataset(train=False, num_digits=num_digits)

        # get a bounded version of the network+circuit structre for each sum
        # also get the indices that were classified correctly and so should be verified
        (
            bounded_module_per_sum,
            correctly_classified_idxs,
        ) = get_bounded_modules_and_samples_to_verify(
            softmax,
            num_digits,
            test_dataset,
        )

        # get bounded NN
        bounded_cnn = BoundedModule(
            mnist_cnn,
            torch.empty_like(test_dataset[0][0][0].unsqueeze(0)),
            verbose=False,
        )

        epsilon = 0.001
        timings_e2e = []
        timings_grb = []
        
        for idx in track(correctly_classified_idxs):
            input_imgs, sum_label = test_dataset[idx]

            # create perturbed input
            ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
            ptb_input = BoundedTensor(input_imgs, ptb)

            e2e_start = time()
            correct_sum_lower_bound = (
                bounded_module_per_sum[sum_label]
                .compute_bounds(x=ptb_input, method="IBP")[0]
                .item()
            )
            e2e_end = time()

            try:
                # get bounds per sum for the Gurobi thing
                grb_start = time()
                correct_minimum, _ = get_bounds_for_sum_gurobi(
                    bounded_network=bounded_cnn,
                    input_imgs=input_imgs,
                    epsilon=epsilon,
                    correct_sum_sdd=test_dataset.sdd_per_sum[sum_label],
                    num_digits=num_digits,
                )
                grb_end = time()

                timings_e2e.append(e2e_end - e2e_start)
                timings_grb.append(grb_end - grb_start)

            except Exception as e:
                print(f"Sample {idx} \t error: {e}")

        print(
            f"#digits: {num_digits:<10}",
            f"epsilon: {epsilon}  ",
            f"average E2E time: {(sum(timings_e2e) / len(timings_e2e)):.4f}",
            f"average GRB time: {(sum(timings_grb) / len(timings_grb)):.4f}",
        )