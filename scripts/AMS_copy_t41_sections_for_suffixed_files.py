import os
import sys
import re
import argparse
import shutil
from collections import defaultdict

def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="""
        Recursively process TAPE41/t41 files in a directory structure.
        
        This script searches for related groups of TAPE41 or .t41 files where one file 
        is a base file and others are suffixed versions (e.g., "TAPE41" and "TAPE41_02" 
        or "Ir.t41" and "Ir_02.t41"). When such groups are found, it uses the 'cpkf' 
        tool to copy a specified section from each suffixed file into the base file, 
        using the file's suffix in the copy operation.
        
        For each matching group of files, the script will run commands like:
        cpkf suffixed_file base_file "section_name --suffix _XX"
        
        Example:
        For files TAPE41, TAPE41_02, TAPE41_03 and section 'XC', it will run:
        cpkf TAPE41_02 TAPE41 "XC --suffix _02"
        cpkf TAPE41_03 "XC --suffix _03"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('directory', 
                       help='Root directory to search for TAPE41/t41 files')
    parser.add_argument('section', 
                       help='Section name to copy between files')
    parser.add_argument('-v', '--verbose', 
                       action='store_true',
                       help='Print detailed progress information')
    
    return parser

def find_related_files(directory):
    """Group related t41/TAPE41 files in the directory."""
    file_groups = defaultdict(list)
    
    for filename in os.listdir(directory):
        if filename.endswith('.t41') or 'TAPE41' in filename:
            match = re.match(r'(.*?)(_\d+)?(\.t41)?$', filename)
            if match:
                base_name = match.group(1)
                suffix = match.group(2) or ''
                file_groups[base_name].append((filename, suffix))
    
    return {base: files for base, files in file_groups.items() if len(files) >= 2}

def process_directory(directory, section_name, verbose=False):
    """Process files in a single directory, running cpkf for each suffixed file."""
    file_groups = find_related_files(directory)
    
    for base_name, files in file_groups.items():
        if verbose:
            print(f"Found file group with base name: {base_name}")
            
        # Find the base file (the one without a suffix)
        base_file = next(f[0] for f in files if not f[1])
        base_path = os.path.join(directory, base_file)
        
        # Create the combined filename
        combined_file = os.path.splitext(base_file)[0] + '_combined'
        if base_file.endswith('.t41'):
            combined_file += '.t41'
        combined_path = os.path.join(directory, combined_file)
        
        # Create a copy of the base file
        if verbose:
            print(f"Creating combined file: {combined_file}")
        shutil.copy2(base_path, combined_path)
        
        # Process each suffixed file
        for filename, suffix in files:
            if suffix:
                # Quote the filenames and command arguments
                source_file = f'"{os.path.join(directory, filename)}"'
                dest_file = f'"{combined_path}"'
                section_arg = f'"{section_name} --rename {section_name}{suffix}"'
                
                cmd = f'cpkf {source_file} {dest_file} {section_arg}'
                
                print(f"Executing: {cmd}")
                exit_status = os.system(cmd)
                
                if exit_status != 0:
                    print(f"Error processing {filename}: command returned {exit_status}", 
                          file=sys.stderr)

def process_files(root_directory, section_name, verbose=False):
    """Walk through directory tree and process files in each directory."""
    for dirpath, dirnames, filenames in os.walk(root_directory):
        if not any(f.endswith('.t41') or 'TAPE41' in f for f in filenames):
            continue
            
        if verbose:
            print(f"\nProcessing directory: {dirpath}")
        process_directory(dirpath, section_name, verbose)

def main(args=None):
    parser = create_parser()
    args = parser.parse_args(args)
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        sys.exit(1)
    
    process_files(args.directory, args.section, args.verbose)

if __name__ == '__main__':
    main(['/Users/haiiro/NoSync/AMSPython/data/tape41_difference_dat', 'FOO', '-v'])
    # main()