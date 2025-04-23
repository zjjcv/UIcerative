import os
from pathlib import Path

def print_directory_structure():
    """Print the structure of the data directories."""
    base_path = Path("concatenated_images")
    
    for category in ["UC", "CD", "ITB"]:
        category_path = base_path / category
        print(f"\nCategory: {category}")
        
        if not category_path.exists():
            print(f"  Directory not found: {category_path}")
            continue
            
        # Print all numbered folders and their contents
        for case_folder in sorted(category_path.glob("[0-9]*")):
            print(f"\n  Folder: {case_folder.name}")
            csv_file = case_folder / f"{case_folder.name}.csv"
            jpg_file = case_folder / f"{case_folder.name}.jpg"
            
            if csv_file.exists():
                print(f"    ✓ Found CSV: {csv_file.name}")
            else:
                print(f"    ✗ Missing CSV: {csv_file.name}")
                
            if jpg_file.exists():
                print(f"    ✓ Found JPG: {jpg_file.name}")
            else:
                print(f"    ✗ Missing JPG: {jpg_file.name}")

if __name__ == "__main__":
    print("Checking data directory structure...")
    print_directory_structure()