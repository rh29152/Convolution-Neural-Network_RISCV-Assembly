# CNN-v2 Project

This repository contains files related to a Convolutional Neural Network (CNN) implementation, with a focus on RISC-V vectorized code, data processing, and compiler files for various operations.

## Folder Structure

### 1. **Data**
   - Contains MNIST image test data in matrix form.
   - The data is used as input for the CNN model and is formatted for efficient processing in subsequent operations.

### 2. **assembly**
   - Contains vectorized RISC-V assembly code for the CNN operations.
   - This folder includes optimized assembly implementations for CNN layers, enabling better performance on RISC-V processors.

### 3. **python**
   - Contains Python scripts used for various tasks related to the CNN model.
     - **Script 1**: Writes the MNIST image test data to `CNN.s` file for assembly processing.
     - **Script 2**: Prints logs for tracking model performance and data flow.

### 4. **veer**
   - Contains files related to the Veer compiler, which is used to compile the CNN model's assembly code for execution on a RISC-V processor.
   - The Veer compiler optimizes the CNN operations to improve runtime efficiency on RISC-V hardware.

### 5. **Makefile**
   - A build automation tool used to compile the project. It helps in compiling the RISC-V assembly code, running the Python scripts, and managing dependencies across various components of the project.

### **RUN Command**  
      make allV
