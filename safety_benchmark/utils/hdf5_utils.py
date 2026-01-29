"""
General utility to explore any HDF5 file structure recursively.

Usage:
    python debug_scripts/explore_hdf5.py <path_to_hdf5> [--path <group_path>]

Examples:
    python debug_scripts/explore_hdf5.py file.hdf5
    python debug_scripts/explore_hdf5.py file.hdf5 --path data/demo_1
    python debug_scripts/explore_hdf5.py file.hdf5 --path data/demo_0/obs
"""

import h5py
import sys
import argparse
import numpy as np
from typing import List

def format_attrs(obj, prefix=""):
    """Format and print attributes of an HDF5 object."""
    if len(obj.attrs) == 0:
        return
    
    for attr_name, attr_val in obj.attrs.items():
        # Format attribute value for display
        if isinstance(attr_val, np.ndarray):
            if attr_val.size > 5:
                val_str = f"ndarray{attr_val.shape} {attr_val.dtype}"
            else:
                val_str = repr(attr_val.tolist())
        elif isinstance(attr_val, bytes):
            decoded = attr_val.decode('utf-8', errors='replace')
            val_str = repr(decoded[:50] + "..." if len(decoded) > 50 else decoded)
        else:
            val_str = repr(attr_val)
            if len(val_str) > 80:
                val_str = val_str[:77] + "..."
        print(f"{prefix}@ {attr_name}: {val_str}")


def explore_hdf5(file_or_group, indent=0, name=None):
    """
    Recursively explore and print HDF5 structure.
    
    Args:
        file_or_group: h5py.File or h5py.Group object
        indent: Current indentation level
        name: Name to display for this level (optional)
    """
    prefix = "  " * indent
    
    # Print name if provided (for root or filtered path)
    if name is not None:
        if isinstance(file_or_group, h5py.Group):
            print(f"{prefix}{name}/")
        else:
            print(f"{prefix}{name}")
        indent += 1
        prefix = "  " * indent
    
    # Print attributes for this group/file
    format_attrs(file_or_group, prefix)
    
    # Iterate through items
    for key in file_or_group.keys():
        item = file_or_group[key]
        
        if isinstance(item, h5py.Group):
            print(f"{prefix}{key}/")
            explore_hdf5(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            dtype_str = str(item.dtype)
            if item.dtype.kind == 'O':
                dtype_str = "object"
            print(f"{prefix}{key}: {item.shape} {dtype_str}")
            # Also print dataset attributes if any
            format_attrs(item, prefix + "  ")


def copy_group_recursive(source_group, target_group):
    """
    Recursively copy a group and all its contents (subgroups, datasets, attributes).
    
    Args:
        source_group: Source h5py.Group
        target_group: Target h5py.Group
    """
    # Copy all attributes
    for attr_name, attr_val in source_group.attrs.items():
        target_group.attrs[attr_name] = attr_val
    
    # Copy all items
    for key in source_group.keys():
        source_item = source_group[key]
        
        if isinstance(source_item, h5py.Group):
            # Create subgroup and recurse
            target_subgroup = target_group.create_group(key)
            copy_group_recursive(source_item, target_subgroup)
        elif isinstance(source_item, h5py.Dataset):
            # Copy dataset, preserving shape and dtype
            target_group.create_dataset(key, data=source_item[:], dtype=source_item.dtype)
            # Copy dataset attributes
            for attr_name, attr_val in source_item.attrs.items():
                target_group[key].attrs[attr_name] = attr_val


def combine_hdf5_files(input_paths: List[str], output_path: str):
    """
    Combine multiple HDF5 files by concatenating demos at the demo level.
    
    Demos from subsequent files are renumbered sequentially (e.g., if first file has
    demo_0 to demo_14, second file's demos start at demo_15). All attributes and
    structure within demos are preserved. The n_episodes and n_steps attributes
    in the data group are updated to be the sum across all files.
    
    Args:
        input_paths: List of paths to input HDF5 files
        output_path: Path for the combined output HDF5 file
    """
    if not input_paths:
        raise ValueError("input_paths must be non-empty")
    
    total_episodes = 0
    total_steps = 0
    demo_counter = 0
    
    # Open first file to get structure and metadata
    with h5py.File(input_paths[0], "r") as first_file:
        if "data" not in first_file:
            raise ValueError(f"File {input_paths[0]} does not contain 'data' group")
        
        first_data = first_file["data"]
        
        # Create output file and data group
        with h5py.File(output_path, "w") as out_file:
            out_data = out_file.create_group("data")
            
            # Copy attributes from first file's data group (except n_episodes and n_steps)
            for attr_name, attr_val in first_data.attrs.items():
                if attr_name not in ["n_episodes", "n_steps"]:
                    out_data.attrs[attr_name] = attr_val
            
            # Process all input files
            for file_idx, input_path in enumerate(input_paths):
                print(f"Processing file {input_path}")
                with h5py.File(input_path, "r") as in_file:
                    if "data" not in in_file:
                        raise ValueError(f"File {input_path} does not contain 'data' group")
                    
                    in_data = in_file["data"]
                    
                    # Get n_episodes and n_steps from this file
                    if "n_episodes" in in_data.attrs:
                        total_episodes += int(in_data.attrs["n_episodes"])
                    if "n_steps" in in_data.attrs:
                        total_steps += int(in_data.attrs["n_steps"])
                    
                    # Find all demo groups and sort them numerically
                    demo_keys = [key for key in in_data.keys() if key.startswith("demo_")]
                    # Sort by demo number (extract number after "demo_")
                    # Use sorted() to ensure proper numerical ordering (0, 1, 2, ..., 10, 11, ...)
                    demo_keys = sorted(demo_keys, key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else float('inf'))
                    
                    # Copy each demo with new numbering
                    for old_demo_key in demo_keys:
                        new_demo_key = f"demo_{demo_counter}"
                        source_demo = in_data[old_demo_key]
                        print(f"Copying {old_demo_key} to {new_demo_key}")
                        
                        # Create new demo group
                        target_demo = out_data.create_group(new_demo_key)
                        
                        # Recursively copy the demo group
                        copy_group_recursive(source_demo, target_demo)
                        
                        demo_counter += 1
            
            # Update n_episodes and n_steps
            out_data.attrs["n_episodes"] = total_episodes
            out_data.attrs["n_steps"] = total_steps
    
    print(f"Combined {len(input_paths)} HDF5 files into {output_path}")
    print(f"Total episodes: {total_episodes}, Total steps: {total_steps}")
    print(f"Total demos: {demo_counter}")


def print_hdf5_structure(hdf5_path, group_path=None):
    """
    Print the structure of an HDF5 file.
    
    Args:
        hdf5_path: Path to the HDF5 file
        group_path: Optional path to a specific group (e.g., "data/demo_1")
    """
    print(f"\n{'='*60}")
    if group_path:
        print(f"HDF5 Structure: {hdf5_path} [{group_path}]")
    else:
        print(f"HDF5 Structure: {hdf5_path}")
    print(f"{'='*60}\n")
    
    with h5py.File(hdf5_path, "r") as f:
        if group_path:
            if group_path not in f:
                print(f"Error: Path '{group_path}' not found in file")
                print(f"\nAvailable top-level keys: {list(f.keys())}")
                return
            target = f[group_path]
            explore_hdf5(target, indent=0, name=group_path)
        else:
            explore_hdf5(f)
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Explore HDF5 file structure recursively",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python debug_scripts/explore_hdf5.py file.hdf5
  python debug_scripts/explore_hdf5.py file.hdf5 --path data/demo_1
  python debug_scripts/explore_hdf5.py file.hdf5 --path data/demo_0/obs
        """
    )
    parser.add_argument("hdf5_path", help="Path to HDF5 file")
    parser.add_argument(
        "--path", "-p",
        default=None,
        help="Path to specific group/dataset (e.g., data/demo_1)"
    )
    
    args = parser.parse_args()
    print_hdf5_structure(args.hdf5_path, args.path)


def remove_one_step_demos():
    import h5py
    import shutil

    input_path = "resources/teleop_data/pour_water/trial_2.hdf5"
    output_path = "resources/teleop_data/pour_water/trial_1_cleaned.hdf5"

    with h5py.File(input_path, "a") as f:  # "a" = read/write mode
        to_delete = []

        # First collect (don’t delete while iterating)
        for demo_name in f["data"].keys():
            num_samples = f["data"][demo_name].attrs.get("num_samples", None)

            if num_samples is None or num_samples <= 1:
                print(f"❌ Marking {demo_name} for deletion ({num_samples})")
                to_delete.append(demo_name)
            else:
                print(f"✅ Keeping {demo_name} ({num_samples})")

        breakpoint()
        # Now delete
        for demo_name in to_delete:
            del f["data"][demo_name]

if __name__ == "__main__":
    # explore the hdf5 file
    # hdf5_path = "resources/teleop_data/trial_2.hdf5"
    # f = h5py.File(hdf5_path, "r")
    # breakpoint()
    # main()
    # remove_one_step_demos()

    combine_hdf5_files(
        input_paths=[
            "resources/teleop_data/pour_water/trial_1.hdf5",
            "resources/teleop_data/pour_water/trial_2.hdf5",
        ],
        output_path="resources/teleop_data/pour_water/all_data.hdf5"
    )