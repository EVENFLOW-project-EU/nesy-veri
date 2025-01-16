import torch
import operator
from torch import nn
from functools import reduce
from pysdd.sdd import SddNode
from dataclasses import dataclass
from typing import Callable, TypeVar


class NetworksPlusCircuit(nn.Module):
    def __init__(
        self,
        networks: list[nn.Module],
        circuit: SddNode,
        categorical_idxs: list[int],
        parse_to_native: bool = True,
    ):
        super().__init__()
        self.categorical_idxs = categorical_idxs
        self.networks = nn.ModuleList(networks)
        self.circuit = circuit if not parse_to_native else sdd_parse_native(circuit)

    def forward(self, x):
        # evaluate the neural networks
        # the ith network is evaluated on the ith input
        network_outputs = [self.networks[i](x) for i in range(len(self.networks))]

        # concatenate the network outputs and flatten to pass to SDD
        sdd_input = torch.cat(network_outputs, dim=1)

        # evaluate the SDD on the NN outputs
        sdd_output = eval_sdd(
            node=self.circuit,
            add=operator.add,
            mul=operator.mul,
            add_neutral=0,
            mul_neutral=1,
            labelling=sdd_input,
            categorical_idxs=self.categorical_idxs,
        )

        return sdd_output


@dataclass(frozen=True)
class pair:
    prime: "decision_node | int | bool"
    sub: "decision_node | int | bool"


@dataclass(frozen=True)
class decision_node:
    children: tuple[pair, ...]


def sdd_parse_native(node: SddNode) -> decision_node | bool | int:
    if node.is_decision():
        children = []
        for prime, sub in node.elements():
            children.append(pair(sdd_parse_native(prime), sdd_parse_native(sub)))
        return decision_node(tuple(children))
    elif node.is_true():
        return True
    elif node.is_false():
        return False
    elif node.is_literal():
        return node.literal
    else:
        raise RuntimeError("unexpected pattern match in node")


T = TypeVar("T")


def eval_sdd(
    node: decision_node | int | bool,
    add: Callable[[T, T], T],
    mul: Callable[[T, T], T],
    add_neutral: T,
    mul_neutral: T,
    labelling: torch.Tensor,
    categorical_idxs: list[int],
) -> T:

    def do_eval(n: decision_node | int | bool) -> T:

        if n is True:
            return mul_neutral
        elif n is False:
            return add_neutral
        elif isinstance(n, int):
            if abs(n) in categorical_idxs:
                return labelling[:, abs(n) - 1] if n > 0 else 1
            return labelling[:, abs(n) - 1] if n > 0 else 1 - labelling[:, abs(n) - 1]
        else:
            children_values = []
            for p in n.children:
                prime_val, sub_val = do_eval(p.prime), do_eval(p.sub)
                children_values.append(mul(prime_val, sub_val))
            node_value = reduce(add, children_values)

            return node_value

    result = do_eval(node)

    return result


def example_is_robust(bounds_per_class: dict[int, list[float]], correct_class: int):
    # an example is verifiably robust if the upper bounds of all wrong classes
    # are lower than the lower bound of the correct class
    wrong_upper_bounds = [
        ub for class_, (_, ub) in bounds_per_class.items() if class_ != correct_class
    ]
    correct_lower_bound = bounds_per_class[correct_class][0]
    robust = all(ub < correct_lower_bound for ub in wrong_upper_bounds)

    return robust