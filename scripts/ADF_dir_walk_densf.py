import os
from pathlib import Path
import subprocess

def walk_and_rename_rkf_files(root_dir):
    """
    Walk through directory structure and generate new filenames for .rkf files.
    
    Args:
        root_dir (str): The root directory to start walking from
    
    Returns:
        list: List of tuples containing (original_path, new_filename)
    """
    root_path = Path(root_dir)
    results = []
    
    # Walk through all subdirectories
    for current_dir, subdirs, files in os.walk(root_dir):
        current_path = Path(current_dir)
        
        # Check if current directory contains any .rkf files
        rkf_files = [f for f in files if f.endswith('.rkf')]
        
        if rkf_files:
            # Get relative path from root to current directory
            relative_path = current_path.relative_to(root_path)
            
            # Create the new filename by joining path parts with underscores
            if relative_path == Path('.'):  # If we're in the root directory
                path_parts = []
            else:
                path_parts = relative_path.parts
            
            for rkf_file in rkf_files:
                # Create new filename: path_parts joined with underscores + .t41
                if path_parts:
                    new_filename = '_'.join(path_parts) + '.t41'
                else:
                    # If in root directory, use just the original filename with .t41
                    new_filename = Path(rkf_file).stem + '.t41'
                
                original_path = current_path / rkf_file
                # Create absolute path for new file in the same directory as original
                new_absolute_path = current_path / new_filename
                results.append((str(original_path), str(new_absolute_path)))
    
    return results

def execute_densf_commands(file_mappings):
    """
    Execute densf commands for each file mapping.
    
    Args:
        file_mappings (list): List of tuples containing (original_path, new_path)
    """
    for original_path, new_path in file_mappings:
        # Skip if new_path already exists
        if os.path.exists(new_path):
            print(f"Skipping {new_path}, already exists.")
            continue
        # Build the densf command
        densf_input = f"""ADFFILE {original_path}
OUTPUTFILE {new_path}
GRID save fine
END
density scf
KinDens scf"""
        
        # Create the shell command
        command = ["$AMSBIN/densf"]
        
        print(f"Processing: {original_path}")
        print(f"Output: {new_path}")
        
        try:
            # Execute the command with the input, showing output in real-time
            process = subprocess.run(
                command,
                input=densf_input,
                text=True,
                shell=True,
                check=True
            )
            print(f"Success: {new_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {original_path}: {e}")
        except Exception as e:
            print(f"Unexpected error processing {original_path}: {e}")
        
        print("-" * 50)

# Example usage
if __name__ == "__main__":
    # Define your starting root directory here
    root_directory = "/Users/haiiro/NoSync/2025_AMSPythonData/CP450_Heme_NEBs/for_tape_41s"
    
    # Get the results
    file_mappings = walk_and_rename_rkf_files(root_directory)
    
    # Print the results
    for original, new_path in file_mappings:
        print(f"Original: {original}")
        print(f"New path: {new_path}")
        print("-" * 50)
    
    # Execute densf commands
    print("\nExecuting densf commands...")
    execute_densf_commands(file_mappings)