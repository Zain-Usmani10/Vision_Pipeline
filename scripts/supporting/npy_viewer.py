import numpy as np

# 1. Replace with your actual file path
file_path = "filename.npy"

try:
    data = np.load(file_path, allow_pickle=False)
    
    print(f"=== Analysis for: {file_path} ===")
    
    # 3. Print Dimensions and Metadata
    print(f"Dimensions (Shape) : {data.shape}")
    print(f"Total Elements      : {data.size}")
    print(f"Data Type           : {data.dtype}")
    print("-" * 40)
    
    # 4. Print Statistics
    # Check if data contains numbers to prevent errors
    if np.issubdtype(data.dtype, np.number):
        print(f"Minimum Value       : {data.min()}")
        print(f"Maximum Value       : {data.max()}")
        print(f"Average (Mean)      : {data.mean()}")
    else:
        print("Statistics Notice   : Data is not numeric (cannot calculate min/max/average).")
        
    print("-" * 40)
    
    # 5. Print Data Values
    print("Values snippet:")
    print(data)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the path.")
except ValueError as e:
    print(f"Error loading file: {e}")
