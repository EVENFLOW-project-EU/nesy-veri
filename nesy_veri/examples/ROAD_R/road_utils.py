import os
import json
import torch
from functools import reduce
from collections import Counter
from pysdd.sdd import SddManager
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize


def frame_has_pedestrian_in_front(frame):
    return any(
        [0 in bb["agent_ids"] and 0 in bb["loc_ids"] for bb in frame["annos"].values()]
    )


def frame_has_red_light(frame):
    return any(
        [
            9 in bb["agent_ids"] and 0 in bb["action_ids"]
            for bb in frame["annos"].values()
        ]
    )


def frame_has_stopped_car_in_front(frame):
    return any(
        [
            any(agent_id in bb["agent_ids"] for agent_id in range(0, 10))
            and 8 in bb["action_ids"]
            and 0 in bb["loc_ids"]
            for bb in frame["annos"].values()
        ]
    )


def frame_has_vehicle_crossing(frame):
    return any(
        [
            any(agent_id in bb["agent_ids"] for agent_id in range(1, 10))
            and 19 in bb["action_ids"]
            and 18 in bb["action_ids"]
            for bb in frame["annos"].values()
        ]
    )


class ROADRPropositional(Dataset):
    def __init__(
        self,
        dataset_path: os.PathLike,
        subset: str,
        label_level: str,
        sample_every_n: int,
        downsample_img_by: int,
        balance_feature_dataset: bool,
    ):
        assert subset in ["train", "val", "test", "all"]

        # this controls what __getitem__ will return
        assert label_level in ["objects", "actions", "both"]
        self.label_level = label_level

        # specify image downsampling
        img_height = 960 / downsample_img_by
        img_width = 1280 / downsample_img_by
        assert img_height.is_integer() and img_width.is_integer()
        self.transform = Resize((int(img_height), int(img_width)))
        # self.transform = transforms.Compose([
        #     transforms.Resize((int(img_height), int(img_width))),
        #     transforms.Normalize(mean=[0.45, 0.45, 0.46], std=[0.307, 0.308, 0.307])
        # ])

        # extract data from json
        json_path = os.path.join(dataset_path, "road_trainval_v1.0.json")
        with open(json_path, "r") as input_file:
            annotation_data = json.load(input_file)
        self.annotation_data = annotation_data

        self.image_paths, self.features, self.labels, self.label_names = (
            [],
            [],
            [],
            annotation_data["all_av_action_labels"],
        )

        video_names = {
            "train": [
                "2014-06-25-16-45-34_stereo_centre_02",
                "2014-08-11-10-59-18_stereo_centre_02",
                "2014-11-25-09-18-32_stereo_centre_04",
                "2014-12-09-13-21-02_stereo_centre_01",
                "2015-02-03-08-45-10_stereo_centre_02",
                "2015-02-03-19-43-11_stereo_centre_04",
                "2015-02-06-13-57-16_stereo_centre_02",
                "2015-02-13-09-16-26_stereo_centre_02",
                "2015-02-13-09-16-26_stereo_centre_05",
                "2015-02-24-12-32-19_stereo_centre_04",
                "2015-03-03-11-31-36_stereo_centre_01",
            ],
            "val": [
                "2014-11-14-16-34-33_stereo_centre_06",
                "2014-11-18-13-20-12_stereo_centre_05",
                "2014-11-21-16-07-03_stereo_centre_01",
            ],
            "test": [
                "2014-06-26-09-53-12_stereo_centre_02",
                "2014-07-14-14-49-50_stereo_centre_01",
                "2014-07-14-15-42-55_stereo_centre_03",
                "2014-08-08-13-15-11_stereo_centre_01",
            ],
        }

        video_ids = (
            reduce(lambda x, y: x + y, video_names.values())
            if subset == "all"
            else video_names[subset]
        )

        video_data = {
            video_id: annotation_data["db"][video_id] for video_id in video_ids
        }

        for video_id, video in video_data.items():
            for i, frame in enumerate(video["frames"].values()):
                if i % sample_every_n != 0:
                    continue
                # if the AV is doing anything and there are bounding boxes
                if "annos" in frame and "av_action_ids" in frame:
                    av_moving = frame["av_action_ids"][0] == 1
                    av_stopped = frame["av_action_ids"][0] == 0

                    red_light = frame_has_red_light(frame)
                    stopped_car = frame_has_stopped_car_in_front(frame)
                    pedestrian = frame_has_pedestrian_in_front(frame)

                    positive_example = av_moving and not (
                        red_light or stopped_car or pedestrian
                    )
                    negative_example = av_stopped and (
                        red_light or stopped_car or pedestrian
                    )

                    if positive_example or negative_example:
                        self.image_paths.append(
                            f"{dataset_path}/rgb-images/{video_id}/{str(frame['rgb_image_id']).zfill(5)}.jpg"
                        )
                        # self.features.append([red_light, stopped_car, pedestrian])
                        self.features.append([red_light, stopped_car])
                        self.labels.append([av_stopped, av_moving])

        if balance_feature_dataset:
            self.balance_features()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tensor_img = self.transform(read_image(self.image_paths[index]) / 255)
        tensor_img = tensor_img.unsqueeze(0)

        if self.label_level == "objects":
            return (tensor_img, torch.Tensor(self.features[index]).float())
        elif self.label_level == "actions":
            return (tensor_img, torch.Tensor(self.labels[index]).float())
        else:
            return (
                tensor_img,
                torch.Tensor(self.features[index] + self.labels[index]).float(),
            )

    def balance_features(self):
        positive_counter = 0
        negative_counter = 0

        new_image_paths = []
        new_features = []
        new_labels = []

        for idx in range(len(self)):
            features = self.features[idx]

            # if it's a positive sample add it
            if sum(features) != 0:
                positive_counter += 1
                new_image_paths.append(self.image_paths[idx])
                new_features.append(self.features[idx])
                new_labels.append(self.labels[idx])

            # if it's a negative sample add it only if there are
            # fewer negatives than positives
            else:
                if negative_counter < positive_counter:
                    negative_counter += 1
                    new_image_paths.append(self.image_paths[idx])
                    new_features.append(self.features[idx])
                    new_labels.append(self.labels[idx])

        self.image_paths = new_image_paths
        self.features = new_features
        self.labels = new_labels

    def get_object_support(self):
        all_occurrences = torch.stack([torch.Tensor(x) for x in self.features])
        summed_per_class = all_occurrences.sum(dim=0)
        return summed_per_class / len(self)

    def get_action_support(self):
        all_occurrences = torch.stack([torch.Tensor(x) for x in self.labels])
        summed_per_class = all_occurrences.sum(dim=0)
        return summed_per_class / len(self)

    def print_action_support_per_video(self):
        print("\t\t\t\t\t\t", end="")
        print(reduce(lambda x, y: x + "  \t" + y, self.label_names))
        for video_id, video in self.annotation_data["db"].items():
            print(video_id, end="\t\t")
            for k in [0, 1, 2, 3, 4, 5, 6, 7]:
                print(Counter(video["frame_labels"])[k], end="\t\t")
            print()


def get_road_constraint():
    manager = SddManager(var_count=4)
    (
        red_traffic_light,
        stopped_car_in_front,
        stop,
        move_forward,
    ) = [manager.literal(i) for i in range(1, 5)]

    common_sense = ~(stop & move_forward) & (stop | move_forward)
    move_constraint = (~(red_traffic_light | stopped_car_in_front)) | ~move_forward

    constraints = move_constraint & common_sense
    # constraints = move_constraint
    constraints.ref()

    manager.minimize()

    return constraints
