#!/usr/bin/env python3
"""
This file used to add more RAG evaluation cases. Instructions:
To complete it, do the next steps: 
1. select 10 random files from test_content folder, by this script @/select_files.py
2. for each of those files, generate  question, one sentence long, related to the content.
  Select 10 words present in the file, they should be in search result.
  Select 10 words that are not expected in search result, but present in the 'test_content' folder, use search tool to check
  Chose words that are the least common in english. Do not use numbers, dates, or other specific entities.
3. For each question, add line to colon separated file 'test_content/evaluation_tests.csv' ,
 with 3 columns: "Query","Expected qords","Unwanted words". 
 Put each question to the first column, and comma separated expected/unwanted words in the second and third columns rspectively.
 Do not wrap columns in quotas
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Tuple

def get_numeric_directories(base_path: Path) -> List[Path]:
    """Get directories that start with numbers 10-90."""
    numeric_dirs = []
    for item in base_path.iterdir():
        if item.is_dir():
            name = item.name
            # Check if name starts with a number between 10-90
            if name.startswith(('10', '30', '50', '70', '90')):
                numeric_dirs.append(item)
    return numeric_dirs

def collect_files_from_subfolders(numeric_dirs: List[Path]) -> Dict[str, List[Path]]:
    """Collect all .md files from subfolders, grouped by subfolder path."""
    subfolder_files = {}
    
    for numeric_dir in numeric_dirs:
        for root, dirs, files in os.walk(numeric_dir):
            root_path = Path(root)
            # Skip the root numeric directory itself, only include subfolders
            if root_path == numeric_dir:
                continue
                
            md_files = [root_path / f for f in files if f.endswith('.md')]
            if md_files:
                subfolder_key = str(root_path.relative_to(Path('test_content')))
                subfolder_files[subfolder_key] = md_files
    
    return subfolder_files

def select_random_files(subfolder_files: Dict[str, List[Path]], count: int = 10) -> List[Tuple[str, Path]]:
    """Select random files from different subfolders."""
    selected_files = []
    available_subfolders = list(subfolder_files.keys())
    
    if len(available_subfolders) < count:
        print(f"Warning: Only {len(available_subfolders)} subfolders available, selecting from all")
        count = len(available_subfolders)
    
    selected_subfolders = random.sample(available_subfolders, count)
    
    for subfolder in selected_subfolders:
        files_in_subfolder = subfolder_files[subfolder]
        selected_file = random.choice(files_in_subfolder)
        selected_files.append((subfolder, selected_file))
    
    return selected_files

def main():
    base_path = Path('test_content')
    
    print("Finding numeric directories (10-90)...")
    numeric_dirs = get_numeric_directories(base_path)
    print(f"Found directories: {[d.name for d in numeric_dirs]}")
    
    print("Collecting files from subfolders...")
    subfolder_files = collect_files_from_subfolders(numeric_dirs)
    print(f"Found {len(subfolder_files)} subfolders with .md files")
    
    print("Selecting 10 random files from different subfolders...")
    selected_files = select_random_files(subfolder_files, 10)
    
    print("\nSelected files:")
    for i, (subfolder, file_path) in enumerate(selected_files, 1):
        print(f"{i:2d}. {subfolder}: {file_path.name}")
        print(f"    Full path: {file_path}")
    
    return selected_files

if __name__ == '__main__':
    main()