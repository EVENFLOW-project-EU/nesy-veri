import json
import torch
from pathlib import Path
from pysdd.sdd import SddManager
from torch.utils.data import Dataset
from torchvision.io import read_image


class BDDDataset(Dataset):
    def __init__(self, subset: str, labels: str):
        assert subset in ["train", "val", "test"]
        assert labels in ["concepts", "targets", "both"]

        data_dir = Path(__file__).parent / "unprocessed_data"
        images_actions = json.load(
            open(str(data_dir / f"{subset}_25k_images_actions.json"))
        )
        images_reasons = json.load(
            open(str(data_dir / f"{subset}_25k_images_reasons.json"))
        )

        num_images = len(images_actions["images"])

        self.labels = labels
        self.img_paths = []
        self.target_labels = []
        self.concept_labels = []
        for img_idx in range(num_images):
            self.img_paths.append(
                str(data_dir / "data" / images_actions["images"][img_idx]["file_name"])
            )
            self.target_labels.append(
                images_actions["annotations"][img_idx]["category"].append(0)
                if len(images_actions["annotations"][img_idx]["category"]) == 4
                else images_actions["annotations"][img_idx]["category"]
            )
            self.concept_labels.append(images_reasons[img_idx]["reason"])

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
