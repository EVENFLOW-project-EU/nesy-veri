import os
import json
import torch
import shutil
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from collections.abc import Iterable
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize


def parse_ros_timestamp(timestamp_str):
    """
    Parses a ROS timestamp string of the form 'builtin_interfaces.msg.Time(sec=..., nanosec=...)'.
    Extracts sec and nanosec values and returns the total time in seconds as a float.
    """
    sec_start = timestamp_str.find("sec=")
    sec_end = timestamp_str.find(",", sec_start)

    nanosec_start = timestamp_str.find("nanosec=")
    nanosec_end = timestamp_str.find(")", nanosec_start)

    # Extract sec and nanosec as integers (default to 0 if empty)
    if sec_start != -1 and sec_end != -1:
        sec_str = timestamp_str[sec_start + 4 : sec_end]
        sec = int(sec_str) if sec_str else 0  # Default to 0 if empty
    else:
        sec = 0  # Default to 0 if not found

    if nanosec_start != -1 and nanosec_end != -1:
        nanosec_str = timestamp_str[nanosec_start + 8 : nanosec_end]
        nanosec = int(nanosec_str) if nanosec_str else 0  # Default to 0 if empty
    else:
        nanosec = 0  # Default to 0 if not found

    return sec + nanosec * 1e-9


def get_images_df_from_txt(txt_file, image_dir) -> pd.DataFrame:
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
            "ros_left_stamp": image_timestamps,
        }
    )
    images_df = images_df.sort_values("ros_left_stamp").reset_index(drop=True)

    return images_df


def downsample(l, imgs_per_sec):
    checkpoint = -1
    downsampled = []
    time_til_next_img = 1 / imgs_per_sec
    for idx in range(len(l)):
        if l[idx] - checkpoint > time_til_next_img:
            checkpoint = l[idx]
            downsampled.append(checkpoint)
    return downsampled


def get_labelled_sequences(images_df, data_df, num_imgs, time_spacing):
    image_sequences = []
    goal_labels = []
    min_required_time = (num_imgs - 1) * time_spacing

    for _, img_row in images_df.iterrows():
        img_stamp = img_row["ros_left_stamp"]

        # Ensure enough historical data exists
        if img_stamp - min_required_time < images_df["ros_left_stamp"].min():
            continue  # Skip if not enough history

        # Collect image paths spaced apart by time_spacing seconds
        image_paths = []
        timestamps = [img_stamp - j * time_spacing for j in range(num_imgs)]

        for ts in timestamps:
            closest_idx = (images_df["ros_left_stamp"] - ts).abs().idxmin()
            image_paths.append(images_df.loc[closest_idx, "filename"])

        closest_goal_idx = (data_df["ros_left_stamp"] - img_stamp).abs().idxmin()
        goal_status = data_df.iloc[closest_goal_idx]["goal_status"]

        image_sequences.append(image_paths)
        goal_labels.append(goal_status)

    return image_sequences, goal_labels


def create_dataset(
    data_path: Path,
    downsample_sequence: bool,
    imgs_per_sec: int,
    image_sequences: bool,
    imgs_per_sequence: int,
    time_spacing: float,
):
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
                usecols=["ros_left_stamp", "goal_status"],  # type: ignore
            )

            data_df["ros_left_stamp"] = data_df["ros_left_stamp"].apply(
                parse_ros_timestamp
            )

            # combine all unimportant labels into one
            data_df["goal_status"] = data_df["goal_status"].apply(
                lambda x: (
                    "other"
                    if x in ["(unknown)", "initial position", "stopped (unknown)"]
                    else x
                )
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

                if downsample_sequence:
                    images_df = images_df[
                        images_df["ros_left_stamp"].isin(
                            downsample(
                                l=images_df["ros_left_stamp"].to_list(),
                                imgs_per_sec=imgs_per_sec,
                            )
                        )
                    ]

                if image_sequences:
                    # get a sequence of images leading up to each image in imaged_df
                    img_sequences, goal_labels = get_labelled_sequences(
                        images_df,
                        data_df,
                        num_imgs=imgs_per_sequence,
                        time_spacing=time_spacing,
                    )

                    # create a new nested dict
                    trajectory_robot_data[traj_id][robot_id][camera] = (
                        img_sequences,
                        goal_labels,
                    )
                else:
                    # for each image, find the row of the original frame with the closest timestamp
                    image_label_pairs = pd.merge(
                        images_df,
                        data_df,
                        on="ros_left_stamp",
                        how="inner",
                    )

                    # create a new nested dict
                    trajectory_robot_data[traj_id][robot_id][camera] = (
                        list(image_label_pairs["filename"]),
                        list(image_label_pairs["goal_status"]),
                    )

    return trajectory_robot_data


def plot_trajectories(
    data_path: Path,
    downsample_sequence: bool,
    imgs_per_sec: int,
    num_trajectories: int,
    filename: str,
):
    sns.set_style(style="whitegrid")
    fig, axes = plt.subplots(
        nrows=2, ncols=num_trajectories, figsize=(100, 15), sharey=False
    )
    custom_order = [
        "Spawing on floor",
        "initial position",
        "moving to Station1",
        "stopped at Station1",
        "moving to Station2",
        "stopped at Station2",
        "moving to Station3",
        "stopped at Station3",
        "moving to Station4",
        "stopped at Station4",
        "moving to Station5",
        "stopped at Station5",
        "moving to Station6",
        "stopped at Station6",
        "stopped (unknown)",
        "(unknown)",
    ]

    fig.suptitle("Goal Status Over Time")

    for traj_id in range(num_trajectories):
        for robot_id in [1, 2]:
            # read robot info for this robot and trajectory
            data_df = pd.read_csv(
                data_path / f"output_robot{robot_id}data/out{traj_id}.csv",
                usecols=["ros_left_stamp", "goal_status"],  # type: ignore
            )

            data_df["ros_left_stamp"] = data_df["ros_left_stamp"].apply(
                parse_ros_timestamp
            )

            # combine all unimportant labels into one
            data_df["goal_status"] = data_df["goal_status"].apply(
                lambda x: (
                    "other"
                    if x in ["(unknown)", "initial position", "stopped (unknown)"]
                    else x
                )
            )

            # get image directory for the OTHER robot, since it is the one seeing THIS robot
            other_robot = 1 if robot_id == 2 else 2
            image_dir = (
                data_path
                / f"output_rosbag{other_robot}/bag{traj_id}/images/traj_{traj_id}"
            )

            # Create plotting df and highlight conditions
            temp = data_df
            temp["highlight"] = "Normal"  # Default
            temp["goal_status"] = pd.Categorical(
                temp["goal_status"], categories=custom_order, ordered=True
            )

            for camera in ["left", "right"]:
                # get list of images that contain the other robot for this camera
                robot_detected = image_dir / f"detected_images_{camera}.txt"
                camera_img_dir = image_dir / camera

                # from the list of images containing the other robot, this
                # creates a df with two columns: image filenames, and timestamp of each image in seconds
                images_df = get_images_df_from_txt(robot_detected, camera_img_dir)

                if downsample_sequence:
                    images_df = images_df[
                        images_df["ros_left_stamp"].isin(
                            downsample(
                                l=images_df["ros_left_stamp"].to_list(),
                                imgs_per_sec=imgs_per_sec,
                            )
                        )
                    ]

                # for each image, find the row of the original frame with the closest timestamp
                image_label_pairs = pd.merge(
                    images_df,
                    data_df,
                    on="ros_left_stamp",
                    how="inner",
                )

                # Highlight points in plotting df based on whether
                # they exist in the left/right cam of the other robot
                temp.loc[
                    temp.set_index(["ros_left_stamp", "goal_status"]).index.isin(
                        image_label_pairs.set_index(
                            ["ros_left_stamp", "goal_status"]
                        ).index
                    ),
                    "highlight",
                ] = (
                    "Left (Orange)" if camera == "left" else "Right (Green)"
                )

            # Plot all points normally (blue)
            sns.scatterplot(
                ax=axes[robot_id - 1][traj_id],
                data=temp[temp["highlight"] == "Normal"],
                x="ros_left_stamp",
                y="goal_status",
                color="blue",
                alpha=1.0,
                s=10,
                label="Normal",
                edgecolors=None,
            )
            # Plot points from left camera
            sns.scatterplot(
                ax=axes[robot_id - 1][traj_id],
                data=temp[temp["highlight"] == "Left (Orange)"],
                x="ros_left_stamp",
                y="goal_status",
                color="orange",
                alpha=1.0,
                s=20,
                label=f"In R{other_robot}'s LEFT cam (Orange)",
            )
            # Plot points from right camera
            sns.scatterplot(
                ax=axes[robot_id - 1][traj_id],
                data=temp[temp["highlight"] == "Right (Green)"],
                x="ros_left_stamp",
                y="goal_status",
                color="green",
                alpha=1,
                s=20,
                label=f"In R{other_robot}'s RIGHT cam (Green)",
            )

            axes[robot_id - 1][traj_id].set_xlabel("Time")
            axes[robot_id - 1][traj_id].set_title(
                f"Trajectory {traj_id}, Robot {robot_id}"
            )

    plt.tight_layout(pad=2.0)
    plt.savefig(f"dfki/trajectory_figures/{filename}.pdf", dpi=300, transparent=True)


class DetectedRobotImages(Dataset):
    """
    Dataset class for parsing the robot simulation data and creating a dataset for task/location recognition.

    Parameters
    ----------
    downsample_img_by : int
        Factor by which to downsample images. A value of 2 means reducing each dimension by half.
    downsample_sequence : bool
        Whether to downsample the sequence itself by skipping frames.
    imgs_per_sec : int
        Number of images per second to retain when downsampling sequences. Used if `downsample_sequence` is True.
    image_sequences : bool
        Whether to provide additional history in the form of preceding images to the image of interest.
        False: Each data point contains one image with its corresponding goal label.
        True: Each data point contains a sequence of images and the goal label of the last image.
    imgs_per_sequence : int
        Number of images in each sequence. Used if `image_sequences` is True.
    time_spacing : float
        Time gap (in seconds) between consecutive images in a sequence. Used if `image_sequences` is True.
    original_dataset_root : Path
        Root directory of the original dataset.
    """

    def __init__(
        self,
        downsample_img_by: int,
        downsample_sequence: bool,
        imgs_per_sec: int,
        image_sequences: bool,
        imgs_per_sequence: int,
        time_spacing: float,
        video_idxs: Iterable[int],
        original_dataset_root: Optional[Path] = None,
    ):

        self.video_idxs = video_idxs
        self.folder_path = (
            Path(__file__).parent.parent
            / "data"
            / f"dataset_{downsample_img_by}_{downsample_sequence}_{imgs_per_sec}_{image_sequences}_{imgs_per_sequence}_{time_spacing}"
        )
        self.image_sequences = image_sequences

        # if the saved dataset exists, just load it
        if os.path.exists(self.folder_path):
            self.load_dataset()
        else:
            # if the saved dataset doesn't exist, and you don't have
            # access to the original dataset either, you are in trouble
            if original_dataset_root is None or not os.path.exists(
                original_dataset_root
            ):
                raise RuntimeError(
                    "You have neither a saved dataset nor access to the original dataset, seek help"
                )

            # if the saved dataset doesn't exist, but you have access
            # to the original dataset, create this version and save it
            else:
                self.trajectory_robot_data = create_dataset(
                    original_dataset_root,
                    downsample_sequence,
                    imgs_per_sec,
                    image_sequences,
                    imgs_per_sequence,
                    time_spacing,
                )

                self.save_dataset(self.folder_path)
                self.load_dataset()

        self.label_idx_mapping = {
            "other": 0,
            "moving to Station1": 1,
            "moving to Station2": 2,
            "moving to Station3": 3,
            "moving to Station4": 4,
            "moving to Station5": 5,
            "moving to Station6": 6,
            "stopped at Station1": 7,
            "stopped at Station2": 8,
            "stopped at Station3": 9,
            "stopped at Station4": 10,
            "stopped at Station5": 11,
            "stopped at Station6": 12,
        }

        # self.smaller_mapping = {
        #     "other": 0,
        #     "moving to Station1": 1,
        #     "moving to Station2": 2,
        #     "moving to Station3": 3,
        #     "moving to Station4": 4,
        #     "moving to Station5": 5,
        #     "moving to Station6": 6,
        #     "stopped at Station1": 1,
        #     "stopped at Station2": 2,
        #     "stopped at Station3": 3,
        #     "stopped at Station4": 4,
        #     "stopped at Station5": 5,
        #     "stopped at Station6": 6,
        # }

        # specify image downsampling
        img_height = 720 / downsample_img_by
        img_width = 1280 / downsample_img_by
        assert img_height.is_integer() and img_width.is_integer()
        self.transform = Resize((int(img_height), int(img_width)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # create a one-hot label based on the string label and the label-index mapping
        one_hot_label = torch.zeros(len(set(self.label_idx_mapping.values())))
        one_hot_label[self.label_idx_mapping[self.labels[index]]] = 1

        # if each data sample contains only one image, read that image
        if not self.image_sequences:
            tensor_img = self.transform(read_image(self.image_paths[index]) / 255)

        # else, read all the images in the sequence into a single tensor
        else:
            tensor_img = torch.stack(
                [
                    self.transform(read_image(img) / 255)
                    for img in self.image_paths[index]
                ]
            ).permute(1, 0, 2, 3)

        return tensor_img, one_hot_label

    def load_dataset(self):
        self.image_paths = []
        self.labels = []

        # for trajectory_id in sorted(os.listdir(self.folder_path)):
        for trajectory_id in self.video_idxs:
            for robot_id in [1, 2]:
                for camera in ["left", "right"]:
                    this_folder = (
                        self.folder_path / str(trajectory_id) / str(robot_id) / camera
                    )

                    # load the labels for these images
                    with open(this_folder / "img_label_mapping.json", "r") as f:
                        img_label_mapping = json.load(f)

                    # single image per sample
                    if not self.image_sequences:
                        # for each image in the directory, append its path and label
                        # for filename in os.listdir(this_folder):
                        for filename in img_label_mapping.keys():
                            if filename.endswith("png"):
                                self.image_paths.append(this_folder / filename)
                                self.labels.append(img_label_mapping[filename])

                    else:
                        # load the labels for these images
                        with open(this_folder / "img_sequence_mapping.json", "r") as f:
                            img_sequence_mapping = json.load(f)

                        # for each image in the directory, append its path and label
                        # for filename in os.listdir(this_folder):
                        for filename in img_label_mapping.keys():
                            if filename.endswith("png"):
                                self.image_paths.append(
                                    [
                                        this_folder / img_file
                                        for img_file in img_sequence_mapping[filename]
                                    ]
                                )
                                self.labels.append(img_label_mapping[filename])

    def save_dataset(self, folder_path):
        for trajectory_id in self.trajectory_robot_data.keys():
            for robot_id in [1, 2]:
                for camera in ["left", "right"]:
                    # create folder for this video, this robot, and this camera
                    this_folder = (
                        folder_path / str(trajectory_id) / str(robot_id) / camera
                    )
                    os.makedirs(this_folder)

                    img_paths, labels = self.trajectory_robot_data[trajectory_id][
                        robot_id
                    ][camera]

                    # only there is only one image per data point
                    if not self.image_sequences:
                        # keep just the image filename instead of the full path
                        img_filenames = [str(path).split("/")[-1] for path in img_paths]

                    # else use only the image of interest for the label and
                    # flatten the list to save all the images of the sequences
                    else:
                        img_sequence_mapping = {}
                        for paths in img_paths:
                            stripped = [
                                str(path).split("/")[-1] for path in paths
                            ]  # keep just the image filename instead of the full path
                            img_sequence_mapping[stripped[0]] = stripped

                        with open(this_folder / "img_sequence_mapping.json", "w") as f:
                            json.dump(img_sequence_mapping, f)

                        img_filenames = img_sequence_mapping.keys()
                        img_paths = [img for l in img_paths for img in l]

                    # save dictionary with image-label mapping
                    img_label_mapping = dict(zip(img_filenames, labels))

                    with open(this_folder / "img_label_mapping.json", "w") as f:
                        json.dump(img_label_mapping, f)

                    # save each image in this directory
                    for img_path in set(img_paths):
                        shutil.copy(img_path, this_folder)


if __name__ == "__main__":
    path_to_dataset_root = (
        Path(__file__).parents[4] / "srv/evenflow-data/DFKI/Dataset_3"
    )
    dataset = DetectedRobotImages(
        downsample_img_by=8,
        downsample_sequence=True,
        imgs_per_sec=1,
        image_sequences=False,
        imgs_per_sequence=5,
        time_spacing=1.0,
        original_dataset_root=path_to_dataset_root,
        video_idxs=[0, 1, 2, 3, 4, 5],
    )

    print(dataset[0][0].shape)
