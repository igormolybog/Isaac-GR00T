#!/usr/bin/env python
"""
Convert GMR results.pkl to LeRobot v2 dataset format for Unitree G1.

This script takes a .pkl file containing robot trajectories (dof_pos, root_pos, etc.)
and converts it into a LeRobot dataset compatible with the UNITREE_G1 embodiment.
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Unitree G1 29-DOF mapping (indices in the 29-D vector)
G1_JOINT_MAPPING = {
    "left_leg": list(range(0, 6)),
    "right_leg": list(range(6, 12)),
    "waist": list(range(12, 15)),
    "left_arm": list(range(15, 22)),
    "right_arm": list(range(22, 29)),
}

def to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value

def main():
    parser = argparse.ArgumentParser(description="Convert GMR .pkl to LeRobot dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to input .pkl file")
    parser.add_argument("--output", type=str, required=True, help="Path to output dataset directory")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load data
    print(f"Loading {input_path}...")
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    if "dof_pos" not in data:
        raise ValueError("Input .pkl missing 'dof_pos' key")

    dof_pos = np.array(data["dof_pos"]).astype(np.float32)
    num_frames, num_dofs = dof_pos.shape
    print(f"Loaded {num_frames} frames with {num_dofs} DOFs")

    if num_dofs != 29:
        print(f"WARNING: Expected 29 DOFs, got {num_dofs}. Proceeding anyway...")

    # Clean output directory
    if output_path.exists():
        print(f"Cleaning output directory {output_path}...")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    
    meta_path = output_path / "meta"
    data_path = output_path / "data" / "chunk-000"
    meta_path.mkdir()
    data_path.mkdir(parents=True)

    # Prepare DataFrame
    df_data = {
        "index": np.arange(num_frames),
        "episode_index": np.zeros(num_frames, dtype=np.int64),
        "timestamp": np.arange(num_frames) / args.fps,
        # Language/Annotation key (mandatory for UNITREE_G1 in some loaders)
        # 0 corresponds to the task index in tasks.jsonl
        "annotation.human.task_description": np.zeros(num_frames, dtype=np.int64),
    }

    # Split 29-D dof_pos into structured modalities
    for group_name, indices in G1_JOINT_MAPPING.items():
        # For observation.state
        state_key = f"observation.state.{group_name}"
        df_data[state_key] = [dof_pos[i, indices] for i in range(num_frames)]
        
        # For action
        action_key = f"action.{group_name}"
        df_data[action_key] = [dof_pos[i, indices] for i in range(num_frames)]

    # Add missing action/state keys with defaults for UNITREE_G1
    default_keys = {
        "action.left_hand": (2,), 
        "action.right_hand": (2,),
        "action.base_height_command": (1,),
        "action.navigate_command": (3,),
        "observation.state.left_hand": (2,),
        "observation.state.right_hand": (2,),
    }
    for key, shape in default_keys.items():
        if key not in df_data:
            df_data[key] = [np.zeros(shape, dtype=np.float32) for _ in range(num_frames)]

    df = pd.DataFrame(df_data)

    # Write Parquet
    parquet_filename = "episode_000000.parquet"
    parquet_path = data_path / parquet_filename
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)
    print(f"Saved trajectory data to {parquet_path}")

    # Generate Metadata
    # 1. info.json
    info = {
        "codebase_version": "v2.0",
        "robot_type": "unitree_g1",
        "total_episodes": 1,
        "total_frames": num_frames,
        "total_tasks": 1,
        "fps": args.fps,
        "splits": {"train": [0, 0]},
        "chunks_size": 1000,
        "data_path": "data/chunk-000/episode_{episode_index:06d}.parquet"
    }
    with open(meta_path / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # 2. tasks.jsonl
    with open(meta_path / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "GMR Replay"}) + "\n")

    # 3. episodes.jsonl
    episode_meta = {
        "episode_index": 0,
        "tasks": ["GMR Replay"],
        "length": num_frames,
    }
    with open(meta_path / "episodes.jsonl", "w") as f:
        f.write(json.dumps(episode_meta) + "\n")

    # 4. modality.json
    # Maps joint groups to the separate columns we created
    modality = {
        "video": {
            "ego_view": {"original_key": "video.ego_view"} # Placeholder
        },
        "state": {
            "left_leg": {"start": 0, "end": 6, "original_key": "observation.state.left_leg"},
            "right_leg": {"start": 0, "end": 6, "original_key": "observation.state.right_leg"},
            "waist": {"start": 0, "end": 3, "original_key": "observation.state.waist"},
            "left_arm": {"start": 0, "end": 7, "original_key": "observation.state.left_arm"},
            "right_arm": {"start": 0, "end": 7, "original_key": "observation.state.right_arm"},
            "left_hand": {"start": 0, "end": 2, "original_key": "observation.state.left_hand"},
            "right_hand": {"start": 0, "end": 2, "original_key": "observation.state.right_hand"},
        },
        "action": {
            "left_arm": {"start": 0, "end": 7, "original_key": "action.left_arm"},
            "right_arm": {"start": 0, "end": 7, "original_key": "action.right_arm"},
            "left_hand": {"start": 0, "end": 2, "original_key": "action.left_hand"},
            "right_hand": {"start": 0, "end": 2, "original_key": "action.right_hand"},
            "waist": {"start": 0, "end": 3, "original_key": "action.waist"},
            "base_height_command": {"start": 0, "end": 1, "original_key": "action.base_height_command"},
            "navigate_command": {"start": 0, "end": 3, "original_key": "action.navigate_command"},
        },
        "annotation": {
            "human.task_description": {"original_key": "annotation.human.task_description"}
        }
    }
    with open(meta_path / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)

    # 5. stats.json
    stats = {}
    for col in df.columns:
        if col in ["index", "episode_index", "timestamp"]:
            continue
        col_data = np.stack(df[col].values)
        stats[col] = {
            "mean": to_serializable(np.mean(col_data, axis=0)),
            "std": to_serializable(np.std(col_data, axis=0)),
            "min": to_serializable(np.min(col_data, axis=0)),
            "max": to_serializable(np.max(col_data, axis=0)),
            "q01": to_serializable(np.percentile(col_data, 1, axis=0)),
            "q99": to_serializable(np.percentile(col_data, 99, axis=0)),
        }
    with open(meta_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Dataset conversion complete: {output_path}")

if __name__ == "__main__":
    main()
