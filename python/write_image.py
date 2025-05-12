import numpy as np

def read_image_matrix(file_path):
    # Read the image from the image.txt file and convert it to a numpy array
    image = np.loadtxt(file_path)
    
    # Check if the image is 28x28
    if image.shape != (28, 28):
        raise ValueError("Image must be 28x28 pixels.")
    
    return image

def convert_matrix_to_asm_format(matrix):
    # Flatten the matrix and convert it into a string of .float values
    flattened_matrix = matrix.flatten()
    asm_lines = []
    
    # Group the values into 28-per-line to match the 28x28 matrix format
    for i in range(0, len(flattened_matrix), 28):
        line = ', '.join([f'{x:.8f}' for x in flattened_matrix[i:i+28]])
        asm_lines.append(f"    .float {line}")
    
    return "\n".join(asm_lines)

def insert_matrix_into_asm(asm_file_path, matrix_asm, tag="image:"):
    # Open the .s file and insert the matrix at the specified tag
    with open(asm_file_path, 'r') as f:
        asm_lines = f.readlines()
    
    # Find the position of the tag and insert the matrix
    for i, line in enumerate(asm_lines):
        if line.strip() == tag:
            insert_index = i + 1  # Insert after the tag line
            break
    else:
        raise ValueError(f"Tag '{tag}' not found in the assembly file.")
    
    # Insert the matrix at the found position
    asm_lines.insert(insert_index, matrix_asm + "\n")
    
    # Write the modified assembly back to the file
    with open(asm_file_path, 'w') as f:
        f.writelines(asm_lines)

def main():
    # Paths for image.txt and assembly file
    image_file_path = "image9.txt"  # Your image.txt file
    asm_file_path = "Vectorized.s"     # Your assembly file where you want to insert the matrix
    
    # Step 1: Read the image matrix from the text file
    image_matrix = read_image_matrix(image_file_path)
    
    # Step 2: Convert the matrix to assembly format
    matrix_asm = convert_matrix_to_asm_format(image_matrix)
    
    # Step 3: Insert the matrix into the assembly file at the tag "image:"
    insert_matrix_into_asm(asm_file_path, matrix_asm, tag="image6:")
    print("Matrix inserted into the assembly file.")

if __name__ == "__main__":
    main()