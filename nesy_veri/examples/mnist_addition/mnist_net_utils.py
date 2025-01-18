import operator
from dataclasses import dataclass
from functools import reduce
from typing import Callable, TypeVar

import torch
from nesy_veri.utils import eval_sdd, sdd_parse_native
from pysdd.sdd import SddNode
from torch import nn


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
        network_outputs = [self.networks[i](x[i].unsqueeze(1)) for i in range(len(self.networks))]

        # concatenate the network outputs and flatten to pass to SDD
        # sdd_input = torch.cat(network_outputs, dim=1)

        # Define weight matrices for each output
        weight_0 = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Map network_outputs[0] to the first 10 positions
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Fill other positions with zeros
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,    
        )

        weight_1 = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Fill other positions with zeros
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Map network_outputs[0] to the last 10 positions
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )

        # Compute contributions from each output
        result_0 = torch.mm(network_outputs[0], weight_0.T)  # Shape (1, 20)
        result_1 = torch.mm(network_outputs[1], weight_1.T)  # Shape (1, 20)

        # Add the results to get the final tensor
        sdd_input = result_0 + result_1

        # # ensure this isn't wrong, compare with the previous version
        assert sdd_input.equal(torch.cat(network_outputs, dim=1))

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
