import os
import json
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from pysdd.sdd import SddManager
from torch.utils.data import Dataset
from torchvision.io import read_image


class BDDDataset(Dataset):
    def __init__(self, subset: str, labels: str):
        assert subset in ["train", "val", "test"]
        assert labels in ["concepts", "targets", "both"]

        data_dir = Path(__file__).parent / "unprocessed_data"
        actions = json.load(open(str(data_dir / f"{subset}_25k_images_actions.json")))
        reasons = json.load(open(str(data_dir / f"{subset}_25k_images_reasons.json")))

        actions["images"] = sorted(actions["images"], key=lambda k: k["file_name"])
        reasons = sorted(reasons, key=lambda k: k["file_name"])

        self.labels = labels
        self.img_paths = []
        self.target_labels = []
        self.concept_labels = []
        for idx, img in enumerate(actions["images"]):
            img_id = img["id"]
            target_labels = actions["annotations"][img_id]["category"]

            if len(target_labels) == 4 or target_labels[4] == 0:
                image_path = str(data_dir / "data" / img["file_name"])
                assert os.path.isfile(image_path)
                assert img["file_name"] == reasons[idx]["file_name"]

                self.img_paths.append(image_path)
                self.target_labels.append(target_labels)
                self.concept_labels.append(reasons[idx]["reason"])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        if self.labels == "concepts":
            return (
                read_image(self.img_paths[idx]) / 255,
                torch.Tensor(self.concept_labels[idx]),
            )
        elif self.labels == "targets":
            return (
                read_image(self.img_paths[idx]) / 255,
                torch.Tensor(self.target_labels[idx]),
            )
        else:
            return (
                read_image(self.img_paths[idx]) / 255,
                torch.Tensor(self.concept_labels[idx]),
                torch.Tensor(self.target_labels[idx]),
            )

    def get_class_support(self, print_: bool = True):
        all_concepts_stacked = torch.stack(
            [torch.Tensor(x) for x in self.concept_labels]
        )
        concept_occurrences = torch.sum(all_concepts_stacked, dim=0)
        concept_support = [occ.item() / len(self) for occ in concept_occurrences]

        if print_:
            for concept in range(21):
                print(
                    f"concept {concept} - support = {round(concept_support[concept]*100, 2)}%"
                )

        return concept_support


def get_BDD_constraints():
    manager = SddManager(26, 0)
    (
        # 4 actions + 1 useless
        move_forward,
        stop,
        turn_left,
        turn_right,
        confuse,
        # these imply you should move forward
        green_light,
        follow_traffic,
        road_clear,
        # these imply you should stop
        red_light,
        traffic_sign,
        obstacle_car,
        obstacle_person,
        obstacle_rider,
        obstacle_other,
        # these imply you can't turn left
        no_left_lane,
        left_obstacle,
        solid_line_left,
        # these imply you can turn left
        left_lane,
        left_green_light,
        left_follow,
        # these imply you can't turn right
        no_right_lane,
        right_obstacle,
        solid_line_right,
        # these imply you can turn right
        right_lane,
        right_green_light,
        right_follow,
    ) = [manager.literal(i) for i in range(1, 27)]

    # construct the constraints one action at a time
    move_forward_constraint = (
        ~(green_light | follow_traffic | road_clear)
    ) | move_forward

    obstacle = obstacle_car | obstacle_person | obstacle_rider | obstacle_other
    stop_constraint = (~(red_light | traffic_sign | obstacle)) | stop

    can_turn_left = left_lane | left_green_light | left_follow
    cannot_turn_left = no_left_lane | left_obstacle | solid_line_left
    turn_left_constraint = (~(can_turn_left & (~cannot_turn_left))) | turn_left

    can_turn_right = right_lane | right_green_light | right_follow
    cannot_turn_right = no_right_lane | right_obstacle | solid_line_right
    turn_right_constraint = (~(can_turn_right & (~cannot_turn_right))) | turn_right

    common_sense = (~(green_light & red_light)) & (~(road_clear & obstacle))

    constraints = (
        move_forward_constraint
        & stop_constraint
        & turn_left_constraint
        & turn_right_constraint
        & common_sense
    )

    constraints.ref()
    manager.minimize()

    return constraints


if __name__ == "__main__":
    concept_mapping = {
        0: "green_light",
        1: "follow_traffic",
        2: "road_clear",
        3: "red_light",
        4: "traffic_sign",
        5: "obstacle_car",
        6: "obstacle_person",
        7: "obstacle_rider",
        8: "obstacle_other",
        9: "no_left_lane",
        10: "left_obstacle",
        11: "solid_line_left",
        12: "left_lane",
        13: "left_green_light",
        14: "left_follow",
        15: "no_right_lane",
        16: "right_obstacle",
        17: "solid_line_right",
        18: "right_lane",
        19: "right_green_light",
        20: "right_follow",
    }

    dataset = BDDDataset(subset="train", labels="concepts")

    # for img, concepts in dataset:
    #     if concepts[3] == 1:
    #         plt.figure(figsize=(16,9))
    #         print([concept_mapping[i] for i, x in enumerate(concepts) if x == 1])
    #         plt.imshow(img.permute(1, 2, 0))
    #         plt.show()

    # fmt: off
    support = dataset.get_class_support()
    oia = [7805, 3489, 4838, 5381, 1539, 233, 163, 5255, 455, 150, 666, 316, 154, 885, 365, 4503, 4514, 3660, 6081, 4022, 2161]
    for c in range(21):
        print(f"Actual support: {support[c]:.3f}, BDD-OIA support: {round(oia[c]/22835, 3)}")
    # fmt: on
