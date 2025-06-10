import pandas as pd
from pathlib import Path

from dfki.data import downsample, get_images_df_from_txt, parse_ros_timestamp


def create_nesy_inference_dataset(
    data_path: Path, downsample_sequence: bool, imgs_per_sec: int, test_video_idxs: list[int],
):
    # this will include all of the data structure by trajectory and robot IDs
    trajectory_robot_data = {}

    for traj_id in test_video_idxs:
        # create a new dict for this trajectory
        trajectory_robot_data[traj_id] = {}

        for robot_id in [1, 2]:

            # create a new nested dict, this has the goals of robot with robot_id
            trajectory_robot_data[traj_id][robot_id] = {}

            # read robot info for this robot and trajectory
            data_df = pd.read_csv(
                data_path / f"output_robot{robot_id}data/out{traj_id}.csv",
                usecols=["ros_left_stamp", "goal_status", "px", "py"],  # type: ignore
            )

            data_df["ros_left_stamp"] = data_df["ros_left_stamp"].apply(
                parse_ros_timestamp
            )

            # combine all unimportant labels into one
            data_df["goal_status"] = data_df["goal_status"].apply(
                lambda x: (
                    "other"
                    if x
                    in [
                        "Spawning on floor",
                        "(unknown)",
                        "initial position",
                        "stopped (unknown)",
                    ]
                    else x
                )
            )

            # get image directory for this robot for inference, not the OTHER robot as in training
            image_dir = (
                data_path
                / f"output_rosbag{robot_id}/bag{traj_id}/images/traj_{traj_id}"
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

                labels = list(
                    zip(
                        list(image_label_pairs["px"]),
                        list(image_label_pairs["py"]),
                    )
                )

                # create a new nested dict
                trajectory_robot_data[traj_id][robot_id][camera] = (
                    list(image_label_pairs["filename"]),
                    labels,
                )

    return trajectory_robot_data
