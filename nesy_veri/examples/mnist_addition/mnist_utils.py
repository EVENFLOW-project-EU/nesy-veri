import os
import torch
import operator
from functools import reduce
from rich.progress import track
from pysdd.sdd import SddManager
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from itertools import product, combinations

from nesy_veri.utils import NetworksPlusCircuit

def get_sdd_for_sums(num_digits: int):
    sdd_per_sum = {}
    manager = SddManager(num_digits * 10, 0)

    constraints = []
    for i in range(num_digits):
        # each digit *must* take a value (0-9)
        constraints.append(
            reduce(
                operator.or_,
                (manager.literal(j + 1) for j in range(i * 10, (i + 1) * 10)),
            )
        )

        # if one value is true, then no other value can be true (pairwise exclusive)
        constraints.append(
            reduce(
                operator.and_,
                [
                    ~(manager.literal(n + 1) & manager.literal(m + 1))
                    for n, m in combinations(range(i * 10, (i + 1) * 10), 2)
                ],
            )
        )
    constraints = reduce(operator.and_, constraints)

    # the sum is smallest when all digits are 0 and largest when all digits are 9
    for sum_ in range(9 * num_digits + 1):
        models = []
        all_worlds = product(range(10), repeat=num_digits)
        sum_worlds = filter(lambda comb: sum(comb) == sum_, all_worlds)
        for combination in sum_worlds:
            model = reduce(
                operator.and_,
                [
                    manager.literal(idx * 10 + value + 1)
                    for idx, value in enumerate(combination)
                ],
            )
            models.append(model)

        sdd_per_sum[sum_] = reduce(operator.or_, models) & constraints

    return sdd_per_sum


class MultiDigitAdditionDataset(Dataset):

    def __init__(self, train: bool, num_digits: int):
        self.num_digits = num_digits
        self.dataset = MNIST(
            root="data/",
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
        )

        self.sdd_per_sum = get_sdd_for_sums(self.num_digits)

    def __len__(self):
        return len(self.dataset) // self.num_digits

    def __getitem__(self, i: int):
        images = []
        sum_label = 0
        for x in range(self.num_digits):
            images.append(self.dataset[i * self.num_digits + x][0])
            sum_label += self.dataset[i * self.num_digits + x][1]

        return torch.stack(images), sum_label


def get_correctly_classified_examples(
    test_dataset: Dataset,
    net_and_circuit_per_sum: dict[int, NetworksPlusCircuit],
    results_path: os.PathLike,
    softmax: bool,
    num_digits: int,
):
    print()
    filename = (
        f"correctly_classified_imgs_{num_digits}{'_softmax' if softmax else ''}.csv"
    )
    correct_images_path = results_path / filename  # type: ignore

    # if the list has already been generated just load it
    if os.path.exists(correct_images_path):
        with open(correct_images_path, "r") as file:
            return list(map(int, file.read().split(",")))

    correctly_predicted_idxs = []
    for idx, (images, label) in track(enumerate(test_dataset)):
        pred_per_sum = {
            sum_: net_plus_circuit(images).item()
            for sum_, net_plus_circuit in net_and_circuit_per_sum.items()
        }

        highest_pred = max(pred_per_sum, key=pred_per_sum.get)  # type: ignore
        if highest_pred == label:
            correctly_predicted_idxs.append(idx)

    # write to file for reading next time
    with open(correct_images_path, "w") as file:
        file.write(",".join(map(str, correctly_predicted_idxs)))

    return correctly_predicted_idxs
