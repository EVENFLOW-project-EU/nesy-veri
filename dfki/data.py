import os
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader


def get_images_df_from_txt(txt_file, image_dir):
    image_files = []
    image_timestamps = []

    with open(txt_file) as f:
        for image_filename in f.read().splitlines():
            if image_filename.endswith(".png"):
                seconds, nanoseconds = map(
                    int, image_filename.replace(".png", "").split("_")
                )
                timestamp = seconds + nanoseconds * 1e-9

                image_files.append(image_dir / image_filename)
                image_timestamps.append(timestamp)

    images_df = pd.DataFrame(
        {
            "filename": image_files,
            "current_time": image_timestamps,
        }
    )
    images_df = images_df.sort_values("current_time").reset_index(drop=True)

    return images_df


def create_dataset(data_path):
    # this will include all of the data structure by trajectory and robot IDs
    trajectory_robot_data = {}

    for traj_id in range(10):
        # create a new dict for this trajectory
        trajectory_robot_data[traj_id] = {}

        for robot_id in [1, 2]:

            # create a new nested dict, this has the goals of robot with robot_id
            trajectory_robot_data[traj_id][robot_id] = {}

            # read robot info for this robot and trajectory
            data_df = pd.read_csv(
                data_path / f"output_robot{robot_id}data/out{traj_id}.csv",
                usecols=["current_time", "goal_status"],
            )

            # get image directory for the OTHER robot, since it is the one seeing THIS robot
            other_robot = 1 if robot_id == 2 else 2
            image_dir = (
                data_path
                / f"output_rosbag{other_robot}/bag{traj_id}/images/traj_{traj_id}"
            )

            for camera in ["left", "right"]:
                # get list of images that contain the other robot for this camera
                robot_detected = image_dir / f"detected_images_{camera}.txt"
                camera_img_dir = image_dir / camera

                # from the list of images containing the other robot, this
                # creates a df with two columns: image filenames, and timestamp of each image in seconds
                images_df = get_images_df_from_txt(robot_detected, camera_img_dir)

                # for each image, find the row of the original frame with the closest timestamp
                image_label_pairs = pd.merge_asof(
                    images_df,
                    data_df,
                    on="current_time",
                    direction="nearest",
                )

                # create a new nested dict
                trajectory_robot_data[traj_id][robot_id][camera] = (
                    list(image_label_pairs["filename"]),
                    list(image_label_pairs["goal_status"]),
                )

    return trajectory_robot_data


class DetectedRobotImages(Dataset):
    def __init__(self, path_to_dataset_root: os.PathLike):
        self.trajectory_robot_data = create_dataset(path_to_dataset_root)
        self.image_paths = []
        self.labels = []

        for trajectory_id in range(10):
            for robot_id in [1, 2]:
                for camera in ["left", "right"]:
                    self.image_paths.extend(
                        self.trajectory_robot_data[trajectory_id][robot_id][camera][0]
                    )
                    self.labels.extend(
                        self.trajectory_robot_data[trajectory_id][robot_id][camera][1]
                    )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, x):
        image = read_image(self.image_paths[x])
        label = self.labels[x]

        return image, label

    def get_action_distribution_per_trajectory(self):
        data = {
            "goal_label": [
                "(unknown)",
                "initial position",
                "moving to Station1",
                "moving to Station2",
                "moving to Station3",
                "moving to Station4",
                "moving to Station5",
                "moving to Station6",
                "stopped (unknown)",
                "stopped at Station1",
                "stopped at Station2",
                "stopped at Station3",
                "stopped at Station4",
                "stopped at Station5",
                "stopped at Station6",
            ],
        }

        for trajectory_id in range(10):
            for robot_id in [1, 2]:
                l = self.trajectory_robot_data[trajectory_id][robot_id]["left"][1]
                r = self.trajectory_robot_data[trajectory_id][robot_id]["right"][1]
                occurrences = Counter(l + r)
                data[f"Trajectory {trajectory_id}, Robot {robot_id}"] = [
                    occurrences[label] for label in data["goal_label"]
                ]

        return data


if __name__ == "__main__":
    path_to_dataset_root = (
        Path(__file__).parents[4] / "srv/evenflow-data/DFKI/Dataset_2"
    )
    dataset = DetectedRobotImages(path_to_dataset_root)

    # action distribution across all trajectories
    occurrences_all = Counter(dataset.labels)

    # get the action distribution per trajectory, turn it to a df, and plot it
    dist = dataset.get_action_distribution_per_trajectory()
    df = pd.DataFrame(dist)
    df.plot.bar(
        x="goal_label",
        subplots=True,
        layout=(10, 2),
        sharey=True,
        legend=False,
        color="blue",
    )
    plt.tight_layout()
    plt.show()
