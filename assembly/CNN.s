#define STDOUT 0xd0580000
.section .text
.global _start
_start:
## START YOUR CODE HERE
################################################################################################################################################
#                             Main Function                                                                                                    #
################################################################################################################################################


#####################################  Digit image selector for Testing Purpose ############################################

   li a0,3
   call Image_selector

#////////////////////  Convolutional layer function calls ///////////////////////////////////////////

    #call 1
    la a1, filter1          # Load address of filter into a1b
    lw a2, sizeM            # Load matrix sizeM into a2
    lw a3, sizeF            # Load filter sizeF into a3
    la a4, R1               # Load address of result matrix into a4
    la a5,bias1
    call conv2d_vectorized   # Call the Convo function
    
    # #call 2
  
    la a1, filter2          # Load address of filter into a1
    lw a2, sizeM            # Load matrix sizeM into a2
    lw a3, sizeF            # Load filter sizeF into a3
    la a4, R2             # Load address of result matrix into a4
    la a5,bias2
    call conv2d_vectorized   # Call the Convo function

    # #call 3
    la a1, filter3         # Load address of filter into a1
    lw a2, sizeM            # Load matrix sizeM into a2
    lw a3, sizeF            # Load filter sizeF into a3
    la a4, R3              # Load address of result matrix into a4
    la a5,bias3
    call conv2d_vectorized   # Call the Convo function

    # #call 4
    la a1, filter4         # Load address of filter into a1
    lw a2, sizeM            # Load matrix sizeM into a2
    lw a3, sizeF            # Load filter sizeF into a3
    la a4, R4             # Load address of result matrix into a4
    la a5,bias4
    call conv2d_vectorized   # Call the Convo function

    # #call 5
    la a1, filter5         # Load address of filter into a1
    lw a2, sizeM            # Load matrix sizeM into a2
    lw a3, sizeF            # Load filter sizeF into a3
    la a4, R5             # Load address of result matrix into a4
    la a5,bias5
    call conv2d_vectorized   # Call the Convo function

    # #call 6
    la a1, filter6         # Load address of filter into a1
    lw a2, sizeM            # Load matrix sizeM into a2
    lw a3, sizeF            # Load filter sizeF into a3
    la a4, R6            # Load address of result matrix into a4
    la a5,bias6
    call conv2d_vectorized   # Call the Convo function

    # #call 7
    la a1, filter7         # Load address of filter into a1
    lw a2, sizeM            # Load matrix sizeM into a2
    lw a3, sizeF            # Load filter sizeF into a3
    la a4, R7             # Load address of result matrix into a4
    la a5,bias7
    call conv2d_vectorized   # Call the Convo function

    # #call 8
    la a1, filter8          # Load address of filter into a1
    lw a2, sizeM            # Load matrix sizeM into a2
    lw a3, sizeF            # Load filter sizeF into a3
    la a4, R8            # Load address of result matrix into a4
    la a5,bias8
    call conv2d_vectorized   # Call the Convo function
    

#////////////////////  Max pooling layer function calls///////////////////////////////////////////

   # call_1
    la a0, R1          #   Load address of input matrix
    li a1, 24           # Input matrix size (24x24)
    la a2,R_1           # Load address of output matrix
    call vector_max_pooling

    # call_2
    la a0, R2         #   Load address of input matrix
    li a1, 24           # Input matrix size (24x24)
    la a2,R_2          # Load address of output matrix
    call vector_max_pooling

    # call_3
    la a0, R3           #   Load address of input matrix
    li a1, 24           # Input matrix size (24x24)
    la a2,R_3          # Load address of output matrix
    call vector_max_pooling

    # call_4
    la a0, R4         #   Load address of input matrix
    li a1, 24           # Input matrix size (24x24)
    la a2,R_4          # Load address of output matrix
    call vector_max_pooling

    # call_5
    la a0, R5         #   Load address of input matrix
    li a1, 24           # Input matrix size (24x24)
    la a2,R_5          # Load address of output matrix
    call vector_max_pooling

    # call_6
    la a0, R6        #   Load address of input matrix
    li a1, 24           # Input matrix size (24x24)
    la a2,R_6          # Load address of output matrix
    call vector_max_pooling

    # call_7
    la a0, R7         #   Load address of input matrix
    li a1, 24           # Input matrix size (24x24)
    la a2,R_7          # Load address of output matrix
    call vector_max_pooling

    # call_8
    la a0, R8         #   Load address of input matrix
    li a1, 24           # Input matrix size (24x24)
    la a2,R_8           # Load address of output matrix
    call vector_max_pooling

#//////////////////// FC layer function calls///////////////////////////////////////////
   # Load matrix size and prepare for operations
    li t0, 12        # t0 = matrix size (12)
    mul t1, t0, t0       # t1 = matrix elements (12*12 = 144)
    
    # Load matrix addresses
    la a0,R_1       # R_1
    la a1,R_2       # R_2
    la a2,R_3       # R_3
    la a3,R_4       # R_4
    la a4,R_5       # R_5
    la a5,R_6       # R_6
    la a6,R_7       # R_7
    la a7,R_8       # R_8
    la s0, flat          # s0 = destination array for flattened matrices
    
    # # Call flatten function
    call flatten
   
   #call Zfc
    la a0,weights
    la a1, flat
    li a2,1 
    la a3,fcOut
    li a4,10
    la a5,bias
    call Zfc
   #relu
    la a0,fcOut
    li t0,10
    call relu 

#//////////////////// Softmax layer function calls///////////////////////////////////////////

    la a0,fcOut     # Load input array address
    la a1, softmaxOut   # Load output array address
    li a2, 10       # Number of elements in the array
    call softmax    # Call softmax function


    la a0,softmaxOut 
    li a1,10
    call printToLogVectorized
    
    j _finish

    

################################################################################################################################################
#                             Functions                                                                                                        #
################################################################################################################################################



# ------------------------------------------------------------------------#
# Function: Image selector                               #
# ------------------------------------------------------------------------#
# array_selector: Returns the address of the specified array (0-8)
#
# Parameters:
# a0 = Array index (0-9)
#
# Returns:
# a0 = Address of the selected array

Image_selector:

    # Check if index is in valid range (0-8)
    li t0, 10
    bgeu a0, t0, invalid_index
    
    # Use branch table approach
    beqz a0, select_array0      # If index == 0, select array0
    li t0, 1
    beq a0, t0, select_array1   # If index == 1, select array1
    li t0, 2
    beq a0, t0, select_array2   # If index == 2, select array2
    li t0, 3
    beq a0, t0, select_array3   # If index == 3, select array3
    li t0, 4
    beq a0, t0, select_array4   # If index == 4, select array4
    li t0, 5
    beq a0, t0, select_array5   # If index == 5, select array5
    li t0, 6
    beq a0, t0, select_array6   # If index == 6, select array6
    li t0, 7
    beq a0, t0, select_array7   # If index == 7, select array7
    li t0, 8
    beq a0, t0, select_array8   # If index == 8, select array8
     li t0, 9
    beq a0, t0, select_array9   # If index == 9, select array8
    
    # Should never reach here due to range check
    j invalid_index
    
select_array0:
    la a0, image0    # Load address of image0
    j return_address

select_array1:
    la a0, image1     # Load address of image1
    j return_address

select_array2:
    la a0,image2      # Load address of  image
    j return_address

select_array3:
    la a0, image3    # Load address of  image
    j return_address

select_array4:
    la a0, image4    # Load address of image4
    j return_address

select_array5:
    la a0, image5    # Load address of image5
    j return_address

select_array6:
    la a0, image6    # Load address of image6
    j return_address

select_array7:
    la a0, image7    # Load address of image7
    j return_address

select_array8:
    la a0, image8    # Load address of image8
    j return_address

select_array9:
    la a0, image9    # Load address of image9
    j return_address

invalid_index:
    # Handle invalid index - either return null/0 or a default array
    li a0, 0         # Return null pointer
return_address:
    ret

# ------------------------------------------------------------------------#
# Function: Convolution2D + Relu activation                               #
# ------------------------------------------------------------------------#

# conv2d_vectorized: Performs vectorized 2D convolution on an input matrix with a filter, adds bias, and applies ReLU activation.
#
# Parameters:
# a0 = input matrix pointer (28x28 float array)
# a1 = filter/kernel pointer (5x5 float array)
# a2 = input_size (width/height of square input, 28)
# a3 = filter_size (width/height of square filter, 5)
# a4 = output pointer (for 24x24 result)
# a5 = bias value pointer (single float)
conv2d_vectorized:
    # Save arguments with new register configuration
    mv      s0, a0          # s0 = input pointer
    mv      s1, a1          # s1 = filter pointer
    mv      s3, a2          # s3 = input_size (28)
    mv      s4, a3          # s4 = filter_size (5)
    mv      s2, a4          # s2 = output pointer
    mv      s6, a5          # s6 = bias value pointer
    
    # Load bias value into a float register
    flw     fa1, 0(s6)      # fa1 = bias value
    
    # Calculate output size = input_size - filter_size + 1
    sub     s5, s3, s4
    addi    s5, s5, 1       # s5 = output_size (24)
    
    # Initialize row index (i = 0)
    li      t0, 0           # t0 = i
    
convo_outer_loop:
    # Check if row index (i) is within bounds
    bge     t0, s5, done
    
    # Initialize column index (j = 0)
    li      t1, 0           # t1 = j
    
convo_inner_loop:
    # Check if column index (j) is within bounds
    bge     t1, s5, convo_next_row
    
    # Determine vector length for this iteration (min(output_size - j, max_vlen))
    sub     t2, s5, t1
    vsetvli t3, t2, e32, m4  # t3 = vlen (up to 4 registers wide for floats)
    
    # Initialize accumulator vector to zeros
    vfmv.v.f v0, ft0        # v0 = all zeros (ft0 assumed to be 0.0)
    
    # Loop through filter rows
    li      t4, 0           # t4 = fr (filter row)
    
filter_row_loop:
    # Check if filter row is within bounds
    bge     t4, s4, end_filter
    
    # Calculate input row offset: (i + fr) * input_size
    add     t6, t0, t4
    mul     t6, t6, s3
    
    # Loop through filter columns
    li      t5, 0           # t5 = fc (filter column)
    
filter_col_loop:
    # Check if filter column is within bounds
    bge     t5, s4, next_filter_row
    
    # Calculate filter offset: fr * filter_size + fc
    mul     a6, t4, s4
    add     a6, a6, t5
    slli    a6, a6, 2       # Multiply by 4 bytes per float
    add     a6, a6, s1      # a6 = &filter[fr * filter_size + fc]
    
    # Load filter value
    flw     fa0, 0(a6)      # fa0 = filter[fr * filter_size + fc]
    
    # Calculate input pointer for current strip
    add     a7, t6, t1      # a7 = (i + fr) * input_size + j
    add     a7, a7, t5      # a7 = (i + fr) * input_size + j + fc
    slli    a7, a7, 2       # Multiply by 4 bytes per float
    add     a7, a7, s0      # a7 = &input[(i + fr) * input_size + j + fc]
    
    # Load vector from input
    vle32.v v4, (a7)        # v4 = multiple input elements
    
    # Broadcast filter value to vector
    vfmv.v.f v8, fa0        # v8 = {filter[...], filter[...], ...}
    
    # Multiply-accumulate: v0 += v8 * v4
    vfmacc.vv v0, v8, v4    # v0 += filter_val * input_vals
    
    # Next filter column
    addi    t5, t5, 1
    j       filter_col_loop
    
next_filter_row:
    # Next filter row
    addi    t4, t4, 1
    j       filter_row_loop
    
end_filter:
    # Broadcast bias to vector register
    vfmv.v.f v4, fa1        # v4 = {bias, bias, ...}
    
    # Add bias to accumulated result: v0 = v0 + v4
    vfadd.vv v0, v0, v4     # Add bias to accumulated convolution result
    
    # Apply ReLU activation function: max(0, x)
    # Create zero vector
    vfmv.v.f v8, ft0        # v8 = {0.0, 0.0, ...} (ft0 assumed to be 0.0)
    
    # Compare with zero and take maximum (ReLU operation)
    vfmax.vv v0, v0, v8     # v0 = max(v0, 0) - keeps positive values, zeros out negatives
    
    # Calculate output pointer
    mul     a6, t0, s5      # a6 = i * output_size
    add     a6, a6, t1      # a6 = i * output_size + j
    slli    a6, a6, 2       # Multiply by 4 bytes per float
    add     a6, a6, s2      # a6 = &output[i * output_size + j]
    
    # Store result vector
    vse32.v v0, (a6)        # Store result to output
    
    # Increment column index by vector length
    add     t1, t1, t3
    j       convo_inner_loop
    
convo_next_row:
    # Increment row index
    addi    t0, t0, 1
    j       convo_outer_loop
    
done:
    
ret
                #/////////////////////////////////#


# ------------------------------------------------------------------------#
# Function: Max_Pooling                                                   #
# ------------------------------------------------------------------------#

# Vector Max Pooling implementation
# a0: pointer to input matrix (equivalent to int* input)
# a1: input matrix size (N) (equivalent to input_size)
# a2: pointer to output matrix (equivalent to int* output)
vector_max_pooling:
    # Store arguments
    mv s0, a0          # s0 = input pointer
    mv s1, a1          # s1 = input size (N)
    mv s2, a2          # s2 = output pointer
    
    # Calculate output size = input_size / 2 (equivalent to output_size = input_size / 2;)
    srli a3, s1, 1     # a3 = output_size (N/2)
    
    # Initialize row counter (equivalent to for (int i = 0; i < input_size; i += 2))
    li t0, 0           # t0 = row (i)
    
outer_loop:
    # Check if we've processed all rows
    slli t1, t0, 1     # t1 = i*2 (row in input matrix)
    bge t1, s1, vector_max_pooling_done   # if (i*2 >= input_size) exit
    
    # Initialize column counter (equivalent to for (int j = 0; j < input_size; j += 2))
    li t2, 0           # t2 = column (j)
    
inner_loop:
    # Check if we've processed all columns for this row
    slli t3, t2, 1     # t3 = j*2
    bge t3, s1, next_row   # if (j*2 >= input_size) go to next row
    
    # Calculate input matrix offset for the current 2x2 block
    # This is equivalent to:
    # int idx1 = i * input_size + j;
    # int idx2 = i * input_size + (j + 1);
    # int idx3 = (i + 1) * input_size + j;
    # int idx4 = (i + 1) * input_size + (j + 1);
    
    slli t3, t1, 2        # t3 = (i*2) * 4 (scale row by 4 bytes per element)
    mul t3, t3, s1        # t3 = (i*2) * 4 * N (byte offset to start of row)
    slli t4, t2, 1        # t4 = j*2
    slli t4, t4, 2        # t4 = (j*2) * 4 (scale column by 4 bytes per element)
    add t4, t3, t4        # t4 = row offset + column offset
    add t4, s0, t4        # t4 = base address + offset = &input[i*2][j*2] = &input[idx1]
    
    # Calculate stride between rows
    slli t5, s1, 2        # t5 = N*4 (bytes per row)
    
    # Set vector length to 2 (2 elements per row in a 2x2 block)
    li t6, 2
    vsetvli t6, t6, e32, m1  # 32-bit elements, one vector register
    
    # Load first row of 2x2 block
    # This loads input[idx1] and input[idx2]
    vle32.v v0, (t4)        # v0 = [input[i*2][j*2], input[i*2][j*2+1]]
    
    # Load second row of 2x2 block
    # This loads input[idx3] and input[idx4]
    add t4, t4, t5          # Move to next row
    vle32.v v1, (t4)        # v1 = [input[i*2+1][j*2], input[i*2+1][j*2+1]]
    
    # Find maximum value in this 2x2 block using vector max
    # This is equivalent to:
    # double max_val = input[idx1];
    # if (input[idx2] > max_val) max_val = input[idx2];
    # if (input[idx3] > max_val) max_val = input[idx3];
    # if (input[idx4] > max_val) max_val = input[idx4];
    vfmax.vv v2, v0, v1     # v2 = max(v0, v1) element-wise
    
    # Do one more max to get the final maximum value
    vfredmax.vs v3, v2, v2  # Reduce v2 to a single max value in v3[0]
    
    # Store result to output matrix
    # Calculate output matrix offset
    # This is equivalent to: output[(i/2) * output_size + (j/2)] = max_val;
    slli t3, t0, 2        # t3 = i*4 (scale row by 4 bytes per element)
    mul t3, t3, a3        # t3 = i*4*(N/2) (byte offset to start of row)
    slli t4, t2, 2        # t4 = j*4 (scale column by 4 bytes per element)
    add t4, t3, t4        # t4 = row offset + column offset
    add t4, s2, t4        # t4 = base address + offset = &output[i][j]
    
    # Store the max value to output matrix
    vfmv.f.s fa0, v3      # Move scalar value from vector register
    fsw fa0, 0(t4)        # Store max value to output[i][j]
    
    # Increment column counter (j += 2 in the C code)
    addi t2, t2, 1        # j++ (note: we're incrementing by 1 here because 
                          # we are tracking j/2 in our assembly)
    j inner_loop
    
next_row:
    addi t0, t0, 1        # i++ (note: we're incrementing by 1 here because 
                          # we are tracking i/2 in our assembly)
    j outer_loop
    
vector_max_pooling_done:
    ret
               #////////////////////////////////////////////# 



# ------------------------------------------------------------------------#
# Function: Flatten                                                       #
# ------------------------------------------------------------------------#

# flatten: Reorganizes 8 input matrices into a single interleaved flat array using RISC-V vector instructions.
#
# Parameters:
# a0-a7 = Eight input matrix pointers (source matrices)
# s0 = Output array pointer (destination for flattened data)
# t1 = Total number of elements to process from each matrix
flatten:
# Setup
li s1, 0               # i = 0 (processed elements counter)
mv s2, s0              # s2 = destination pointer for flat array (using s0 as originally intended)
# Use t1 directly for element count, keep s0 as destination pointer

# Process the matrices in chunks that fit in vector registers
process_loop:
# Check if we've processed all elements
bge s1, t1, flatten_done_opt

# Calculate remaining elements
sub t4, t1, s1         # t4 = remaining elements

# We'll process 8 elements at a time (one from each matrix)
li t5, 8               # Process min(8, remaining) elements
bge t4, t5, set_vl
mv t5, t4              # If less than 8 remaining, process just those

set_vl:
# Set vector length to process t5 elements
vsetvli t6, t5, e32, m1  # 32-bit elements, standard LMUL

# Calculate offset into source matrices
slli t4, s1, 2         # t4 = i * 4 bytes

# Load one element from each matrix into a vector register
add t6, a0, t4
vle32.v v0, (t6)       # v0 = R_1[i]
add t6, a1, t4
vle32.v v1, (t6)       # v1 = R_2[i]
add t6, a2, t4
vle32.v v2, (t6)       # v2 = R_3[i]
add t6, a3, t4
vle32.v v3, (t6)       # v3 = R_4[i]
add t6, a4, t4
vle32.v v4, (t6)       # v4 = R_5[i]
add t6, a5, t4
vle32.v v5, (t6)       # v5 = R_6[i]
add t6, a6, t4
vle32.v v6, (t6)       # v6 = R_7[i]
add t6, a7, t4
vle32.v v7, (t6)       # v7 = R_8[i]

# Store all vectors in interleaved fashion
# Each will store a single element from the 8 matrices
li s3, 0               # j = 0

store_loop:
beq s3, t5, store_done  # If processed all elements in chunk

# Extract single element from each vector and store
# Use more direct approach with indexed store operations
li t2, 0              # Initialize index counter

# Handle index 0 directly
beq s3, zero, index0
li t2, 1              # If not index 0, start at index 1
j index_check

index0:
# For index 0, use vmv.x.s directly
vmv.x.s t6, v0
sw t6, 0(s2)
addi s2, s2, 4
vmv.x.s t6, v1
sw t6, 0(s2)
addi s2, s2, 4
vmv.x.s t6, v2
sw t6, 0(s2)
addi s2, s2, 4
vmv.x.s t6, v3
sw t6, 0(s2)
addi s2, s2, 4
vmv.x.s t6, v4
sw t6, 0(s2)
addi s2, s2, 4
vmv.x.s t6, v5
sw t6, 0(s2)
addi s2, s2, 4
vmv.x.s t6, v6
sw t6, 0(s2)
addi s2, s2, 4
vmv.x.s t6, v7
sw t6, 0(s2)
addi s2, s2, 4
j store_completed

index_check:
# Handle index 1 if needed
bne s3, t2, check_index2
vslidedown.vi v8, v0, 1
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v1, 1
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v2, 1
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v3, 1
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v4, 1
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v5, 1
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v6, 1
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v7, 1
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
j store_completed

check_index2:
# Handle index 2 if needed
addi t2, t2, 1
bne s3, t2, check_index3
vslidedown.vi v8, v0, 2
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v1, 2
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v2, 2
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v3, 2
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v4, 2
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v5, 2
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v6, 2
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v7, 2
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
j store_completed

check_index3:
# Handle index 3 if needed
addi t2, t2, 1
bne s3, t2, check_index4
vslidedown.vi v8, v0, 3
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v1, 3
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v2, 3
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v3, 3
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v4, 3
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v5, 3
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v6, 3
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v7, 3
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
j store_completed

check_index4:
# Handle index 4 if needed
addi t2, t2, 1
bne s3, t2, check_index5
vslidedown.vi v8, v0, 4
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v1, 4
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v2, 4
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v3, 4
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v4, 4
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v5, 4
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v6, 4
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v7, 4
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
j store_completed

check_index5:
# Handle index 5 if needed
addi t2, t2, 1
bne s3, t2, check_index6
vslidedown.vi v8, v0, 5
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v1, 5
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v2, 5
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v3, 5
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v4, 5
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v5, 5
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v6, 5
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v7, 5
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
j store_completed

check_index6:
# Handle index 6 if needed
addi t2, t2, 1
bne s3, t2, check_index7
vslidedown.vi v8, v0, 6
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v1, 6
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v2, 6
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v3, 6
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v4, 6
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v5, 6
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v6, 6
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v7, 6
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
j store_completed

check_index7:
# Handle index 7 if needed
vslidedown.vi v8, v0, 7
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v1, 7
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v2, 7
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v3, 7
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v4, 7
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v5, 7
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v6, 7
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4
vslidedown.vi v8, v7, 7
vmv.x.s t6, v8
sw t6, 0(s2)
addi s2, s2, 4

store_completed:

addi s3, s3, 1         # j++
j store_loop

store_done:
add s1, s1, t5         # i += processed_elements
j process_loop

flatten_done_opt:
ret

            #////////////////////////////////////////////////#

# ------------------------------------------------------------------------#
# Function: Zfc                                                           #
# ------------------------------------------------------------------------#
# Zfc: Performs fully-connected layer operation using RISC-V vector instructions.
#
# Parameters:
# a0 = Weights matrix pointer
# a1 = Flattened input array pointer
# a3 = Output array pointer
# a5 = Bias values pointer
#
Zfc:
# Initialize index counter
 li t1, 0 # i = t1 = current index
 li t0, 10 # Changed from 10 to 7 to match matrix dimensions
for_i:
# Check if we've processed all elements
 bge t1, t0, zfc_done
 li s0, 0 # j = s0 = 0
 li s1, 1152 # Changed from 1152 to 7 to match matrix dimensions
 fcvt.s.w ft0, zero # Initialize accumulator to zero (floating point)

 vsetvli s6,s1,e32
 vmv.v.x v3,zero
for_j:
 bge s0, s1, next_i # (j >= 7)
# Calculate how many elements remain
 sub t2, s1, s0 # t2 = 7 - # of processed elements
# Set vector length for this iteration
 vsetvli t2, t2, e32 # Set vector length
# Calculate address for flat
 slli t3, s0, 2 # t3(offset) = s0(j) * 4
 add t3, a1, t3 # t3 = flat_base address + offset
# Calculate address for weights - corrected row indexing
 slli t4, t1, 2 # t4 = i * 4 
 mul t4, t4, s1 # t4 = i * 4 * 7 (correct row offset: i * row_size)
 add t4, a0, t4 # t4 = matrix_base address + row offset
 slli t5, s0, 2 # t5 = j * 4 (for column offset)
 add t4, t4, t5 # t4 = matrix_base address + row offset + column offset
# Load elements of flat & weights matrix
 vle32.v v0, (t3) # v0 = flat elements
 vle32.v v1, (t4) # v1 = matrix elements
# perform element-wise multiplication
 vfmul.vv v2, v0, v1 # v2 = v0 * v1
# Reduction- sum of all elements in v2
 vfredsum.vs v3, v2, v3 # v3[0] += sum(v2)
# increment index
 add s0, s0, t2
 j for_j
next_i:
# Move vector result to floating point register
 vfmv.f.s ft1, v3
 fadd.s ft0, ft0, ft1 # Accumulate result
# calculate the addresses for bias and result
 slli s3, t1, 2 # offset(i*4)
 add s4, a5, s3 # s4 = bias + offset
 add s5, a3, s3 # s5 = baseresultAdress + offset
# load bias value as floating point
 flw ft2, 0(s4) # load bias value
# add the bias
 fadd.s ft0, ft0, ft2
# store result in result matrix
 fsw ft0, 0(s5)
# increment index
 addi t1, t1, 1
 # Reset v3 for next iteration
 vsetvli zero, zero, e32
 vmv.s.x v3, zero
 j for_i
zfc_done:
 ret

     #/////////////////////////////////////////#

# ------------------------------------------------------------------------#
# Function: Relu                                                          #
# ------------------------------------------------------------------------#

# Function: relu
# Sets all negative values in a matrix to zero using vector instructions
# Inputs:
#   - a0: Base address of matrix
#   - t0: Size of matrix (n for n×n matrix)
# Clobbers: t1, t2, t3, t4
relu:
    addi sp, sp, -4
    sw ra, 0(sp)

    
    # Initialize index counter
    li t1, 0              # t1 = current index
    
    # Create vector of zeros for comparison
    vsetvli t2, t0, e32   # Set vector length based on remaining elements
    vmv.v.i v2, 0         # v2 = vector of zeros for max comparison

relu_loop:
    # Check if we've processed all elements
    bge t1, t0, relu_done
    
    # Calculate how many elements remain
    sub t3, t0, t1
    
    # Set vector length for this iteration
    vsetvli t2, t3, e32   # Set vector length to remaining elements or max VLEN
    
    # Calculate address for this chunk
    slli t4, t1, 2        # t4 = t1 * 4 (float size)
    add t4, a0, t4        # t4 = base address + offset
    
    # Load vector of matrix elements
    vle32.v v1, (t4)      # v1 = matrix elements
    
    # Apply ReLU: max(0, x)
    vfmax.vv v1, v1, v2   # v1 = max(v1, 0)
    
    # Store results back to memory
    vse32.v v1, (t4)      # Store updated values
    
    # Update index counter and continue
    add t1, t1, t2        # Increment index by vector length processed
    j relu_loop

relu_done:
    lw ra, 0(sp)
    addi sp, sp, 4
    ret


# ------------------------------------------------------------------------#
# Function: softmax                                                       #
# ------------------------------------------------------------------------#

# softmax: Computes softmax activation function on an array of values using RISC-V vector instructions.
#
# Parameters:
# a0 = Input array pointer (float values)
# a1 = Output array pointer (for results)
# a2 = Number of elements in the array
softmax:
 
 mv s0, a0       # s0 = input array pointer
 mv s1, a1       # s1 = output array pointer
 mv s2, a2       # s2 = total number of elements
 
 #calculate e^x approximation for each element
 mv s3, s0       # Current input pointer
 mv s4, s1       # Current output pointer
 mv t0, s2       # Remaining elements to process
 
 #constants
 la t4, half
 flw fa0, 0(t4)  # fa0 = 0.5
 la t5, sixth
 flw fa1, 0(t5)  # fa1 = 1/6
 la t6, one
 flw fa2, 0(t6)  # fa2 = 1.0
 
 #sum accumulator
 fmv.w.x fa3, zero
 
exp_loop:
 vsetvli t1, t0, e32, ta, ma  # Set vector length for this iteration
 vle32.v v1, (s3)             # v1 = current chunk of input vector
 
 vfmul.vv v2, v1, v1          # v2 = x²
 
 vfmul.vv v3, v2, v1          # v3 = x³
 
 vfmv.v.f v4, fa0
 vfmul.vv v4, v4, v2          # v4 = x² * 0.5
 
 vfmv.v.f v5, fa1
 vfmul.vv v5, v5, v3          # v5 = x³ * 1/6
 
 vfadd.vv v6, v1, v4          # v6 = x + x²/2
 vfadd.vv v6, v6, v5          # v6 = x + x²/2 + x³/6
 vfmv.v.f v7, fa2
 vfadd.vv v7, v7, v6          # v7 = 1 + x + x²/2 + x³/6 ≈ e^x Taylor Series
 
 # Store e^x approximations temporarily to output array
 vse32.v v7, (s4)
 
 # Accumulate the sum of all e^x values
 vmv.v.i v8, 0
 vfredusum.vs v8, v7, v8      # Sum all elements in v7
 vfmv.f.s ft0, v8             # Extract scalar result
 fadd.s fa3, fa3, ft0         # Add to running total
 
 slli t2, t1, 2               # t2 = bytes processed (4 bytes per float)
 add s3, s3, t2               # Update input pointer
 add s4, s4, t2               # Update output pointer
 sub t0, t0, t1               # Decrease remaining elements
 bnez t0, exp_loop            # Continue if elements remain
 
 #Normalize by dividing each e^x by the sum
 mv s3, s1                    # Reset to beginning of output array
 mv s4, s1                    # We'll overwrite the same array
 mv t0, s2                    # Reset element counter
 
normalize_loop:
 vsetvli t1, t0, e32, ta, ma  # Set vector length for this iteration
 vle32.v v1, (s3)             # Load e^x values
 
 vfmv.v.f v2, fa3             # Broadcast sum to v2
 vfdiv.vv v3, v1, v2          # v3 = e^x / sum
 
 vse32.v v3, (s4)
 
 slli t2, t1, 2               # t2 = bytes processed
 add s3, s3, t2               # Update input pointer
 add s4, s4, t2               # Update output pointer
 sub t0, t0, t1               # Decrease remaining elements
 bnez t0, normalize_loop      # Continue if elements remain
 
 ret # return


 #Function: print
# Logs values from array in a0 into registers v1 for debugging and output.
# Inputs:
#   - a0: Base address of array
#   - a1: Size of array i.e. number of elements to log
# Clobbers: t0,t1, t2,t3 ft0, ft1.
printToLogVectorized:        
    addi sp, sp, -4
    sw a0, 0(sp)

    li t0, 0x123                 # Pattern for help in python script
    li t0, 0x456                 # Pattern for help in python script
    mv a1, a1                   # moving size to get it from log 
	li t0, 0		                # load i = 0
    printloop:
        vsetvli t3, a1, e32           # Set VLEN based on a1
        slli t4, t3, 2                # Compute VLEN * 4 for address increment

        vle32.v v1, (a0)              # Load real[i] into v1
        add a0, a0, t4                # Increment pointer for real[] by VLEN * 4
        add t0, t0, t3                # Increment index

        bge t0, a1, endPrintLoop      # Exit loop if i >= size
        j printloop                   # Jump to start of loop
    endPrintLoop:
    li t0, 0x123                    # Pattern for help in python script
    li t0, 0x456                    # Pattern for help in python script
	
    lw a0, 0(sp)
    addi sp, sp, 4

	jr ra


_finish:
    li x3, 0xd0580000
    addi x5, x0, 0xff
    sb x5, 0(x3)
    beq x0, x0, _finish

    .rept 100
        nop
    .endr

################################################################################################################################################
#                          |------------ DATA ----------------|                                                                                #
################################################################################################################################################

.section .data
.equ MatrixSize, 28
.equ FilterSize,5
.equ FeatureMapSize,24
sizeM:    .word MatrixSize
sizeF:    .word FilterSize
sizeMap:  .word FeatureMapSize


  #////////////////////////////////////////////////////////// Different Test 28 by 28 MNIST Images /////////////////////////////////////////////
image0:
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.20000000, 0.62000000, 0.99000000, 0.62000000, 0.20000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.19000000, 0.93000000, 0.99000000, 0.99000000, 0.99000000, 0.93000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.21000000, 0.89000000, 0.99000000, 0.99000000, 0.94000000, 0.91000000, 0.99000000, 0.22000000, 0.02000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.04000000, 0.24000000, 0.88000000, 0.99000000, 0.99000000, 0.99000000, 0.79000000, 0.33000000, 0.99000000, 0.99000000, 0.48000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.64000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.38000000, 0.74000000, 0.99000000, 0.65000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.20000000, 0.93000000, 0.99000000, 0.99000000, 0.75000000, 0.45000000, 0.99000000, 0.89000000, 0.18000000, 0.31000000, 1.00000000, 0.66000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.19000000, 0.93000000, 0.99000000, 0.99000000, 0.70000000, 0.05000000, 0.29000000, 0.47000000, 0.08000000, 0.00000000, 0.00000000, 0.99000000, 0.95000000, 0.20000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.15000000, 0.65000000, 0.99000000, 0.91000000, 0.82000000, 0.33000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.99000000, 0.99000000, 0.65000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.03000000, 0.70000000, 0.99000000, 0.94000000, 0.28000000, 0.07000000, 0.11000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.99000000, 0.99000000, 0.76000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.22000000, 0.99000000, 0.99000000, 0.25000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.99000000, 0.99000000, 0.76000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.78000000, 0.99000000, 0.75000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.99000000, 0.77000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.30000000, 0.96000000, 0.99000000, 0.44000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.99000000, 0.99000000, 0.58000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.33000000, 0.99000000, 0.90000000, 0.10000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.03000000, 0.53000000, 0.99000000, 0.73000000, 0.05000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.33000000, 0.99000000, 0.87000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.03000000, 0.51000000, 0.99000000, 0.88000000, 0.28000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.33000000, 0.99000000, 0.57000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.19000000, 0.65000000, 0.99000000, 0.68000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.34000000, 0.99000000, 0.88000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.45000000, 0.93000000, 0.99000000, 0.64000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.33000000, 0.99000000, 0.98000000, 0.57000000, 0.19000000, 0.11000000, 0.33000000, 0.70000000, 0.88000000, 0.99000000, 0.87000000, 0.65000000, 0.22000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.33000000, 0.99000000, 0.99000000, 0.99000000, 0.90000000, 0.84000000, 0.99000000, 0.99000000, 0.99000000, 0.77000000, 0.51000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.11000000, 0.78000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.91000000, 0.57000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.10000000, 0.50000000, 0.99000000, 0.99000000, 0.99000000, 0.55000000, 0.15000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
image1:
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.49000000, 0.99000000, 1.00000000, 0.25000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.38000000, 0.96000000, 0.98000000, 0.99000000, 0.24000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.50000000, 0.98000000, 0.98000000, 0.99000000, 0.24000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.27000000, 0.93000000, 0.98000000, 0.83000000, 0.12000000, 0.03000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.24000000, 0.89000000, 0.98000000, 0.98000000, 0.37000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.61000000, 0.99000000, 0.99000000, 0.74000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.08000000, 0.99000000, 0.98000000, 0.92000000, 0.26000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.13000000, 0.80000000, 0.99000000, 0.98000000, 0.49000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.41000000, 0.98000000, 0.99000000, 0.72000000, 0.06000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.31000000, 0.94000000, 0.98000000, 0.76000000, 0.09000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.13000000, 0.99000000, 0.99000000, 0.99000000, 0.62000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.59000000, 0.98000000, 0.98000000, 0.98000000, 0.15000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.19000000, 0.87000000, 0.98000000, 0.98000000, 0.67000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.92000000, 0.98000000, 0.98000000, 0.77000000, 0.05000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.99000000, 0.98000000, 0.98000000, 0.35000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.62000000, 1.00000000, 0.99000000, 0.99000000, 0.12000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.19000000, 0.89000000, 0.99000000, 0.97000000, 0.55000000, 0.03000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.25000000, 0.98000000, 0.99000000, 0.86000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.25000000, 0.98000000, 0.99000000, 0.86000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.09000000, 0.76000000, 0.99000000, 0.86000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
image2:
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.05000000, 0.10000000, 0.39000000, 0.48000000, 0.03000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.13000000, 0.59000000, 0.82000000, 0.99000000, 0.99000000, 0.99000000, 0.57000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.16000000, 0.60000000, 0.96000000, 0.99000000, 0.99000000, 0.88000000, 0.83000000, 0.99000000, 0.91000000, 0.16000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.06000000, 0.60000000, 0.94000000, 0.99000000, 0.99000000, 0.99000000, 0.85000000, 0.12000000, 0.15000000, 0.99000000, 0.99000000, 0.24000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.38000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.85000000, 0.11000000, 0.00000000, 0.15000000, 0.99000000, 0.99000000, 0.24000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.71000000, 0.99000000, 0.99000000, 0.86000000, 0.65000000, 0.12000000, 0.00000000, 0.00000000, 0.30000000, 0.99000000, 0.99000000, 0.24000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.10000000, 0.50000000, 0.23000000, 0.09000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.39000000, 0.99000000, 0.99000000, 0.24000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.62000000, 0.99000000, 0.99000000, 0.24000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.43000000, 0.47000000, 0.48000000, 0.47000000, 0.79000000, 0.99000000, 0.76000000, 0.01000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.04000000, 0.21000000, 0.70000000, 0.99000000, 0.99000000, 1.00000000, 0.99000000, 0.99000000, 0.89000000, 0.14000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.02000000, 0.21000000, 0.89000000, 0.99000000, 0.95000000, 0.89000000, 0.67000000, 0.95000000, 0.99000000, 0.99000000, 0.91000000, 0.46000000, 0.02000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.02000000, 0.31000000, 0.99000000, 0.99000000, 0.49000000, 0.23000000, 0.00000000, 0.07000000, 0.82000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.34000000, 0.03000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.02000000, 0.53000000, 0.99000000, 0.99000000, 0.71000000, 0.06000000, 0.00000000, 0.08000000, 0.80000000, 0.99000000, 0.97000000, 0.51000000, 0.68000000, 0.99000000, 0.99000000, 0.72000000, 0.26000000, 0.19000000, 0.19000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.01000000, 0.53000000, 0.99000000, 0.95000000, 0.42000000, 0.07000000, 0.00000000, 0.21000000, 0.78000000, 0.99000000, 0.85000000, 0.25000000, 0.00000000, 0.05000000, 0.28000000, 0.64000000, 0.95000000, 0.99000000, 0.99000000, 0.87000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.41000000, 0.99000000, 0.95000000, 0.35000000, 0.07000000, 0.29000000, 0.67000000, 0.96000000, 0.99000000, 0.49000000, 0.11000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.35000000, 0.71000000, 0.71000000, 0.15000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.91000000, 0.99000000, 0.96000000, 0.80000000, 0.85000000, 0.99000000, 0.99000000, 0.99000000, 0.49000000, 0.01000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.81000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.70000000, 0.45000000, 0.14000000, 0.02000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.05000000, 0.36000000, 0.56000000, 0.47000000, 0.09000000, 0.02000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
image3:
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.15000000, 0.17000000, 0.41000000, 1.00000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.68000000, 0.02000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.17000000, 0.55000000, 0.88000000, 0.89000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.62000000, 0.05000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.70000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.23000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.43000000, 0.99000000, 0.99000000, 0.90000000, 0.52000000, 0.52000000, 0.52000000, 0.52000000, 0.74000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.23000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.02000000, 0.11000000, 0.11000000, 0.09000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.05000000, 0.89000000, 0.99000000, 0.99000000, 0.67000000, 0.03000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.33000000, 0.95000000, 0.99000000, 0.99000000, 0.56000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.35000000, 0.74000000, 0.99000000, 0.99000000, 0.99000000, 0.05000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.36000000, 0.83000000, 0.97000000, 0.99000000, 0.99000000, 0.99000000, 0.80000000, 0.04000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.13000000, 0.49000000, 0.76000000, 0.76000000, 0.76000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.93000000, 0.40000000, 0.11000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.18000000, 0.87000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.69000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.18000000, 0.87000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 1.00000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.29000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.12000000, 0.48000000, 0.20000000, 0.17000000, 0.17000000, 0.17000000, 0.17000000, 0.56000000, 0.99000000, 0.99000000, 0.29000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.06000000, 0.99000000, 0.99000000, 0.29000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.34000000, 0.99000000, 0.99000000, 0.29000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.02000000, 0.29000000, 0.04000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.38000000, 0.95000000, 0.99000000, 0.99000000, 0.29000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.24000000, 0.72000000, 0.99000000, 0.11000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.07000000, 0.36000000, 0.94000000, 0.99000000, 0.99000000, 0.95000000, 0.25000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.82000000, 0.99000000, 0.99000000, 0.58000000, 0.53000000, 0.53000000, 0.53000000, 0.53000000, 0.80000000, 0.99000000, 0.99000000, 0.99000000, 0.74000000, 0.33000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.82000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.90000000, 0.60000000, 0.03000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.19000000, 0.62000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.85000000, 0.81000000, 0.57000000, 0.18000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.03000000, 0.40000000, 0.92000000, 0.99000000, 0.67000000, 0.40000000, 0.09000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
image4:
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.26000000, 0.91000000, 0.15000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.24000000, 0.32000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.47000000, 0.71000000, 0.15000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.49000000, 0.64000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.01000000, 0.60000000, 0.82000000, 0.16000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.86000000, 0.64000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.11000000, 1.00000000, 0.64000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.87000000, 0.64000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.72000000, 1.00000000, 0.49000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.18000000, 0.96000000, 0.64000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.78000000, 1.00000000, 0.22000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.47000000, 1.00000000, 0.64000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.09000000, 0.91000000, 1.00000000, 0.11000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.62000000, 1.00000000, 0.47000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.64000000, 1.00000000, 0.85000000, 0.06000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.62000000, 1.00000000, 0.26000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.05000000, 0.34000000, 0.70000000, 0.97000000, 1.00000000, 0.36000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.62000000, 1.00000000, 0.33000000, 0.00000000, 0.00000000, 0.00000000, 0.18000000, 0.19000000, 0.45000000, 0.56000000, 0.59000000, 0.95000000, 0.95000000, 0.92000000, 0.70000000, 0.95000000, 0.99000000, 0.16000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.59000000, 0.99000000, 0.93000000, 0.81000000, 0.81000000, 0.81000000, 0.99000000, 1.00000000, 0.98000000, 0.94000000, 0.78000000, 0.56000000, 0.36000000, 0.11000000, 0.02000000, 0.91000000, 0.98000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.47000000, 0.69000000, 0.69000000, 0.69000000, 0.69000000, 0.69000000, 0.38000000, 0.22000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.40000000, 1.00000000, 0.86000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.66000000, 1.00000000, 0.54000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.66000000, 1.00000000, 0.22000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.66000000, 1.00000000, 0.22000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.66000000, 1.00000000, 0.37000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.66000000, 1.00000000, 0.38000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.66000000, 1.00000000, 0.60000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.66000000, 1.00000000, 0.60000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.38000000, 1.00000000, 0.60000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
image5:
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.01000000, 0.07000000, 0.07000000, 0.07000000, 0.49000000, 0.53000000, 0.69000000, 0.10000000, 0.65000000, 1.00000000, 0.97000000, 0.50000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.12000000, 0.14000000, 0.37000000, 0.60000000, 0.67000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.88000000, 0.67000000, 0.99000000, 0.95000000, 0.76000000, 0.25000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.19000000, 0.93000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.98000000, 0.36000000, 0.32000000, 0.32000000, 0.22000000, 0.15000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.07000000, 0.86000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.78000000, 0.71000000, 0.97000000, 0.95000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.31000000, 0.61000000, 0.42000000, 0.99000000, 0.99000000, 0.80000000, 0.04000000, 0.00000000, 0.17000000, 0.60000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.05000000, 0.00000000, 0.60000000, 0.99000000, 0.35000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.55000000, 0.99000000, 0.75000000, 0.01000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.04000000, 0.75000000, 0.99000000, 0.27000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.14000000, 0.95000000, 0.88000000, 0.63000000, 0.42000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.32000000, 0.94000000, 0.99000000, 0.99000000, 0.47000000, 0.10000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.18000000, 0.73000000, 0.99000000, 0.99000000, 0.59000000, 0.11000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.06000000, 0.36000000, 0.99000000, 0.99000000, 0.73000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.98000000, 0.99000000, 0.98000000, 0.25000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.18000000, 0.51000000, 0.72000000, 0.99000000, 0.99000000, 0.81000000, 0.01000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.15000000, 0.58000000, 0.90000000, 0.99000000, 0.99000000, 0.99000000, 0.98000000, 0.71000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.09000000, 0.45000000, 0.87000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.79000000, 0.31000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.09000000, 0.26000000, 0.84000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.78000000, 0.32000000, 0.01000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.07000000, 0.67000000, 0.86000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.76000000, 0.31000000, 0.04000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.22000000, 0.67000000, 0.89000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.96000000, 0.52000000, 0.04000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.53000000, 0.99000000, 0.99000000, 0.99000000, 0.83000000, 0.53000000, 0.52000000, 0.06000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
image6:
image7:
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.45000000, 0.47000000, 0.64000000, 0.99000000, 0.99000000, 0.84000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.25000000, 0.42000000, 0.67000000, 0.98000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.98000000, 0.84000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.10000000, 0.75000000, 0.89000000, 0.89000000, 0.95000000, 0.99000000, 0.99000000, 0.79000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.88000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.27000000, 0.87000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.15000000, 0.07000000, 0.15000000, 0.25000000, 0.88000000, 0.99000000, 0.99000000, 0.72000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.73000000, 0.99000000, 0.99000000, 0.99000000, 0.96000000, 0.42000000, 0.21000000, 0.00000000, 0.00000000, 0.00000000, 0.59000000, 0.99000000, 0.99000000, 0.86000000, 0.08000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.27000000, 0.95000000, 0.99000000, 0.99000000, 0.87000000, 0.23000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.70000000, 0.99000000, 0.99000000, 0.55000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.73000000, 0.99000000, 0.99000000, 0.76000000, 0.26000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.07000000, 0.35000000, 0.94000000, 0.99000000, 0.76000000, 0.26000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.33000000, 0.80000000, 0.75000000, 0.09000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.47000000, 0.99000000, 0.99000000, 0.82000000, 0.09000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.30000000, 0.97000000, 0.99000000, 0.97000000, 0.42000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.99000000, 0.99000000, 0.99000000, 0.40000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.53000000, 1.00000000, 0.99000000, 0.99000000, 0.15000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.02000000, 0.72000000, 0.99000000, 0.99000000, 0.42000000, 0.01000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.04000000, 0.40000000, 0.99000000, 0.99000000, 0.64000000, 0.06000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.05000000, 0.66000000, 0.99000000, 0.99000000, 0.43000000, 0.01000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.16000000, 0.99000000, 0.99000000, 0.85000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.16000000, 0.61000000, 0.99000000, 0.84000000, 0.12000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.65000000, 0.99000000, 0.99000000, 0.42000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.17000000, 0.70000000, 0.99000000, 0.59000000, 0.15000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.54000000, 0.99000000, 0.87000000, 0.15000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.26000000, 0.99000000, 0.31000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
image8: 
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.04000000, 0.80000000, 0.90000000, 0.13000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.10000000, 0.18000000, 0.18000000, 0.12000000, 0.37000000, 1.00000000, 0.84000000, 0.05000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.18000000, 0.60000000, 0.73000000, 0.73000000, 0.87000000, 0.99000000, 0.99000000, 0.52000000, 0.69000000, 1.00000000, 0.74000000, 0.07000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.43000000, 0.99000000, 0.99000000, 0.99000000, 0.96000000, 0.63000000, 0.89000000, 0.99000000, 0.99000000, 1.00000000, 0.36000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.50000000, 0.96000000, 0.99000000, 0.62000000, 0.54000000, 0.08000000, 0.00000000, 0.19000000, 0.91000000, 0.99000000, 0.91000000, 0.03000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.55000000, 1.00000000, 0.87000000, 0.10000000, 0.00000000, 0.00000000, 0.14000000, 0.67000000, 1.00000000, 0.96000000, 0.42000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.22000000, 0.83000000, 0.99000000, 0.63000000, 0.04000000, 0.10000000, 0.70000000, 0.99000000, 0.93000000, 0.44000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.03000000, 0.61000000, 0.99000000, 0.89000000, 0.31000000, 0.87000000, 0.99000000, 0.99000000, 0.43000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.55000000, 0.99000000, 0.99000000, 0.99000000, 1.00000000, 0.99000000, 0.60000000, 0.11000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.43000000, 0.99000000, 0.99000000, 0.99000000, 1.00000000, 0.70000000, 0.15000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.01000000, 0.67000000, 1.00000000, 1.00000000, 1.00000000, 0.70000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.67000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.70000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.10000000, 0.48000000, 1.00000000, 0.99000000, 0.80000000, 0.61000000, 0.99000000, 0.78000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.36000000, 0.99000000, 1.00000000, 0.47000000, 0.05000000, 0.36000000, 0.99000000, 0.62000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.25000000, 0.94000000, 0.99000000, 0.30000000, 0.03000000, 0.13000000, 0.86000000, 0.99000000, 0.49000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.52000000, 1.00000000, 0.75000000, 0.00000000, 0.02000000, 0.42000000, 0.92000000, 1.00000000, 0.42000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.52000000, 0.99000000, 0.75000000, 0.02000000, 0.33000000, 0.99000000, 0.93000000, 0.60000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.60000000, 0.99000000, 0.66000000, 0.75000000, 0.99000000, 0.99000000, 0.30000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.44000000, 0.99000000, 0.99000000, 1.00000000, 0.93000000, 0.51000000, 0.04000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.07000000, 0.46000000, 0.95000000, 0.75000000, 0.44000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
image9:
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.22000000, 0.58000000, 0.82000000, 0.99000000, 0.99000000, 0.44000000, 0.34000000, 0.58000000, 0.22000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.34000000, 0.91000000, 0.99000000, 0.99000000, 0.74000000, 0.82000000, 0.99000000, 0.99000000, 0.99000000, 0.66000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.02000000, 0.22000000, 0.95000000, 0.99000000, 0.75000000, 0.25000000, 0.02000000, 0.05000000, 0.71000000, 0.99000000, 0.99000000, 0.45000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.38000000, 0.99000000, 0.99000000, 0.72000000, 0.05000000, 0.00000000, 0.00000000, 0.36000000, 0.99000000, 0.99000000, 0.88000000, 0.08000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.52000000, 0.99000000, 0.99000000, 0.57000000, 0.05000000, 0.00000000, 0.00000000, 0.00000000, 0.84000000, 0.99000000, 0.99000000, 0.31000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.49000000, 0.99000000, 0.97000000, 0.69000000, 0.04000000, 0.00000000, 0.00000000, 0.03000000, 0.31000000, 0.96000000, 0.99000000, 0.51000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.06000000, 0.91000000, 0.99000000, 0.69000000, 0.00000000, 0.00000000, 0.00000000, 0.14000000, 0.79000000, 0.99000000, 0.99000000, 0.66000000, 0.04000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.09000000, 0.99000000, 0.99000000, 0.12000000, 0.09000000, 0.47000000, 0.77000000, 0.95000000, 0.99000000, 0.99000000, 0.98000000, 0.30000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.06000000, 0.91000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.99000000, 0.89000000, 0.89000000, 0.99000000, 0.91000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.22000000, 0.92000000, 0.99000000, 0.85000000, 0.54000000, 0.16000000, 0.09000000, 0.75000000, 0.99000000, 0.56000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.24000000, 1.00000000, 0.99000000, 0.43000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.28000000, 0.99000000, 0.99000000, 0.08000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.99000000, 0.99000000, 0.08000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.28000000, 0.99000000, 0.99000000, 0.08000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.42000000, 0.99000000, 0.99000000, 0.08000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.18000000, 1.00000000, 0.99000000, 0.08000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.85000000, 0.99000000, 0.22000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.38000000, 0.99000000, 0.74000000, 0.16000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.05000000, 0.72000000, 0.99000000, 0.67000000, 0.04000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.05000000, 0.58000000, 0.99000000, 0.16000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
    .float 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000


  #////////////////////////////////////////////////////////// 8 filters for Convolution Layer /////////////////////////////////////////////

filter1:
    .float -0.3021715581417084, -0.5219495296478271, -0.8275046348571777, -0.7635321617126465, -0.08358605951070786
    .float -0.6808284521102905, -0.5389566421508789, -0.06978537887334824, 0.03661119192838669, 0.47272202372550964
    .float -0.5915364623069763, -0.09034443646669388, 0.31519901752471924, 0.5075920820236206, 0.43331772089004517
    .float -0.13290679454803467, 0.1730104684829712, 0.23429088294506073, 0.08301782608032227, 0.14346738159656525
    .float 0.09642991423606873, 0.07289410382509232, 0.01613827422261238, 0.08963571488857269, -0.008130357600748539
filter2:
    .float -0.3976646959781647, 0.07243328541517258, 0.17796531319618225, 0.3158279061317444, -0.03061327338218689
    .float -0.7922821640968323, -0.03513539955019951, 0.14475052058696747, 0.4069450795650482, -0.003966548480093479
    .float -0.7010685205459595, -0.195740208029747, 0.1875813752412796, 0.19937172532081604, 0.09539283066987991
    .float -0.21912971138954163, -0.27236613631248474, 0.017059145495295525, 0.2383841723203659, 0.1966439187526703
    .float -0.3047935366630554, -0.26542383432388306, 0.05631019547581673, 0.233249232172966, 0.20065777003765106
filter3:
    .float -0.5058345794677734, -0.11047745496034622, 0.3744593560695648, 0.2645780146121979, -0.1141594797372818
    .float -0.506350576877594, 0.2732463777065277, 0.43627992272377014, 0.20766712725162506, -0.5732908248901367
    .float -0.2852497100830078, 0.47801917791366577, 0.38707736134529114, 0.03362441435456276, -0.8110378980636597
    .float -0.11370184272527695, 0.4357048571109772, 0.417684406042099, -0.16270212829113007, -1.010694146156311
    .float 0.21767237782478333, 0.3635902404785156, 0.07909674197435379, -0.33677375316619873, -0.9002063870429993
filter4:
    .float -1.1737827062606812, -1.218120813369751, -1.250355839729309, -0.9605609774589539, -0.7713545560836792
    .float -0.8566417098045349, -0.37871527671813965, -0.28063273429870605, -0.34193313121795654, -0.589036762714386
    .float 0.03962252661585808, 0.3159673511981964, 0.2397918850183487, 0.04922915995121002, -0.26070594787597656
    .float 0.42393410205841064, 0.4725193381309509, 0.5022115707397461, 0.3532373905181885, 0.14413301646709442
    .float 0.37226682901382446, 0.14495070278644562, 0.14529821276664734, 0.31717365980148315, 0.37274178862571716
filter5:
    .float -0.05598093196749687, -0.04746689274907112, -0.3109722137451172, -0.026896797120571136, 0.31424641609191895
    .float -0.1763136088848114, -0.09120946377515793, -0.016533110290765762, 0.27312687039375305, 0.1928989142179489
    .float -0.05651656165719032, 0.146266907453537, 0.19922100007534027, 0.3508509397506714, -0.12861235439777374
    .float 0.06750378012657166, 0.1624288409948349, 0.0652318075299263, 0.12921841442584991, -0.347301721572876
    .float 0.25295358896255493, 0.10161809623241425, -0.03735120967030525, -0.19730061292648315, -0.1538856476545334
filter6:
    .float 0.2053063064813614, 0.1570727825164795, -0.264028400182724, -0.5694141983985901, -0.6235355138778687
    .float -0.05571950227022171, 0.10502171516418457, 0.4118559956550598, 0.17828571796417236, -0.0445619635283947
    .float 0.10382004827260971, 0.4559997320175171, 0.5219376683235168, 0.7184612154960632, 0.1968744546175003
    .float -0.28856056928634644, 0.00629122368991375, 0.2518455982208252, 0.36157751083374023, 0.471097856760025
    .float -0.36171749234199524, -0.6815133690834045, -0.8109551668167114, -0.5517200827598572, 0.007835013791918755
filter7:
    .float 0.19938726723194122, -0.03554271534085274, 0.0012523315381258726, -0.36207810044288635, -0.5317839980125427
    .float 0.15618930757045746, 0.13991329073905945, 0.11875010281801224, 0.15729311108589172, -0.21058908104896545
    .float 0.11568129062652588, 0.12182080000638962, 0.1583261936903, 0.19812358915805817, 0.08120666444301605
    .float 0.2512761056423187, 0.057100940495729446, 0.024182600900530815, 0.09460418671369553, 0.05068109557032585
    .float 0.1325177550315857, 0.1700737476348877, 0.12757639586925507, 0.2108268290758133, 0.2625492513179779
filter8:
    .float -0.40146109461784363, -0.12233902513980865, 0.12766771018505096, 0.42961055040359497, -0.14087915420532227
    .float 0.21626906096935272, 0.35571205615997314, 0.3066645562648773, 0.3928646743297577, 0.010605154559016228
    .float 0.51837158203125, 0.20080401003360748, -0.20074781775474548, 0.02127106860280037, -0.03878828138113022
    .float -0.02099507488310337, -0.5433236956596375, -0.4667297601699829, -0.06299886107444763, 0.03308011591434479
    .float -0.44684046506881714, -0.35074543952941895, -0.16329121589660645, 0.07637804001569748, 0.330104798078537

  #////////////////////////////////////////////////////////// 8 filters  bias values for Convolution Layer /////////////////////////////////////////////

bias1: .float -0.12417933344841003
bias2: .float -0.19517752528190613
bias3: .float -0.14686435461044312
bias4: .float 0.1525413542985916
bias5: .float -0.3670017719268799
bias6: .float -0.25919675827026367
bias7: .float -0.40228351950645447
bias8: .float -0.04272900149226189  


  #////////////////////////////////////////////////////////// 10*1152 Weights Matrix for FC layer Wfc/////////////////////////////////////////////

weights:
# Row 1 of weights matrix
 .float -0.19315732, -0.16441962, 0.15048242, -0.00165857, 0.02078454, -0.09558694, -0.27787380, -0.22545180, -0.07708713, 0.02854570, -0.07630295, 0.03686940
 .float -0.21421432, -0.14934887, 0.03837503, -0.03187454, 0.07260713, 0.05593020, -0.36283213, 0.10212515, -0.10808986, -0.09954599, 0.00248229, -0.06682264
 .float -0.01343328, -0.11530379, -0.28140435, 0.00769539, -0.20840189, -0.16498527, 0.05497671, -0.02501956, -0.06548940, -0.11880178, -0.00264627, 0.03391885
 .float -0.07659315, 0.01044397, 0.06011159, -0.12308413, -0.09168135, -0.02038350, 0.05883408, 0.03675015, 0.08576264, -0.02549832, 0.00389395, 0.07843334
 .float -0.04630465, -0.07588831, 0.06910749, 0.06444945, -0.00545855, 0.02634715, -0.02035098, 0.08787327, -0.06409888, -0.00897929, -0.09523959, -0.00396658
 .float -0.14902648, -0.09719914, -0.05246126, 0.05812227, 0.01455803, -0.07274377, -0.16387430, -0.01773466, -0.04727421, 0.02182859, -0.09410127, -0.04223009
 .float -0.01331938, -0.06749999, -0.12582979, 0.00091952, -0.15179782, -0.08194637, 0.01619815, -0.08910930, -0.06406725, -0.09802606, -0.04285695, -0.03542404
 .float -0.02898901, -0.06680502, -0.09003475, -0.38804394, -0.02189328, -0.16179007, -0.13380380, 0.08784939, -0.01236737, 0.03102898, -0.05818781, -0.14523473
 .float -0.19800650, -0.09991624, 0.03367351, 0.11881182, -0.03442737, -0.08486440, -0.05266554, -0.05234923, -0.07459579, -0.10904965, -0.09367810, 0.05237088
 .float -0.14717454, -0.15340380, 0.02262206, -0.07389697, 0.00019470, -0.05338231, -0.37330508, -0.04134591, 0.00174837, -0.12735045, 0.01476365, -0.02141378
 .float -0.00485172, 0.00114447, -0.25826457, 0.05251256, 0.01838996, -0.13562453, 0.04892204, -0.14565144, -0.06811327, -0.02306803, -0.13568965, 0.05040517
 .float -0.01748886, -0.23265569, 0.09987095, -0.05535816, -0.03148651, -0.05727284, -0.04128993, 0.06352960, -0.01591763, -0.06164107, -0.01195046, -0.05048848
 .float 0.01452127, -0.00152272, -0.00978056, -0.01049262, -0.04084956, 0.04357440, -0.04419616, -0.08062678, 0.00231383, 0.02096560, -0.17887472, 0.07274105
 .float -0.08624574, 0.00397524, 0.07397231, 0.05860235, -0.09340849, -0.07260443, -0.11803129, -0.05584553, -0.03673756, 0.07098243, -0.03383667, -0.03265842
 .float -0.01419673, -0.01153078, -0.18481834, 0.01070895, -0.08243971, -0.00830047, 0.01969926, -0.11315383, -0.09093074, -0.12666197, 0.02227430, 0.05837233
 .float 0.03177540, -0.01760480, 0.03556338, -0.34302133, -0.06112815, -0.17250404, -0.11313362, 0.02960347, -0.04071234, -0.11137414, 0.12231062, -0.04559127
 .float -0.04951756, -0.08386468, 0.12524942, 0.06586558, -0.11585741, -0.12157197, 0.02660586, -0.05627118, -0.06452025, 0.05894332, -0.24149124, -0.01186686
 .float -0.04022996, 0.00627670, 0.00590070, 0.01416865, 0.02869553, 0.02304252, -0.11543947, 0.04881835, -0.02346001, -0.13746782, 0.01803648, -0.02695094
 .float -0.00023435, -0.04921292, -0.21725768, -0.00341879, -0.02077452, -0.18170163, 0.00943506, -0.11517895, -0.00227549, -0.02776510, -0.00835453, -0.06863566
 .float 0.08392478, -0.10059403, 0.02156205, -0.27049387, -0.00929754, -0.03370214, 0.00311599, -0.07556050, -0.04977800, -0.08932226, -0.05184303, -0.08073363
 .float -0.03351502, -0.03990416, -0.15539144, 0.09940437, -0.07206789, 0.04659552, 0.03800440, 0.09943053, -0.05485606, 0.07300816, 0.03960311, 0.01199521
 .float -0.01096452, 0.14010136, 0.02622412, -0.06219661, -0.07837046, 0.19310588, -0.03188971, -0.07488067, -0.12333413, 0.09519426, -0.02895631, -0.05845814
 .float -0.01049952, 0.11171723, -0.12212273, -0.18817876, -0.01861909, 0.10004326, 0.06867139, 0.03048748, -0.09393398, -0.09816212, -0.03742545, -0.11639620
 .float -0.06295384, 0.15368856, 0.01626315, -0.17332181, -0.03698841, -0.11332492, -0.18268480, -0.07560981, -0.10798446, -0.03192874, 0.03204515, -0.17145897
 .float -0.09290400, -0.18835434, -0.08993287, 0.12261792, -0.03322565, 0.07813835, -0.09812604, -0.07502659, -0.00888193, 0.04848098, -0.13538754, 0.11930857
 .float 0.01874377, -0.06209877, 0.06706885, 0.07524338, -0.00745192, 0.07742656, -0.02741209, -0.02928107, -0.00961031, -0.04563009, -0.03327902, -0.03387340
 .float 0.00620037, -0.08319042, 0.13147303, 0.07452271, -0.02809614, 0.01225312, -0.00455794, -0.01369578, -0.16060606, -0.08029541, 0.09758799, -0.00909985
 .float 0.07957105, -0.02744690, -0.04719471, 0.01726554, -0.12519872, 0.06499336, -0.00266510, 0.02850453, -0.09102768, -0.07601695, -0.08308624, -0.04280189
 .float -0.04884764, 0.09042808, -0.02905909, 0.12202325, -0.03662787, 0.04547077, 0.06319988, 0.00194748, -0.16419369, 0.12186393, -0.07174125, -0.03487862
 .float -0.07658055, 0.08228780, 0.04777269, 0.16580585, -0.17504594, 0.20463741, -0.11267091, -0.02075309, -0.18639521, 0.00027410, 0.08874943, 0.06505752
 .float -0.13389711, 0.20851254, -0.10226363, -0.12108021, -0.13354886, 0.07409708, 0.01987169, 0.04888187, -0.12327107, 0.21737666, 0.05864408, -0.25657240
 .float -0.03329637, 0.01645991, 0.07868853, -0.13009486, -0.07353377, 0.08326225, -0.19407173, -0.13397472, -0.04322916, -0.03281514, 0.02564634, -0.17547580
 .float -0.04364989, -0.10120454, -0.02318641, -0.05113185, -0.12645334, -0.12176070, -0.08319099, -0.03721393, 0.07349770, 0.03134989, -0.07207991, 0.00950803
 .float -0.00289964, -0.02684755, -0.00049760, 0.09710483, -0.02039298, 0.06233476, 0.14575467, -0.03261318, 0.13507013, -0.14209166, -0.08899955, -0.05341533
 .float 0.01167156, 0.00537266, -0.00250541, 0.10152306, 0.08703513, -0.23497885, 0.03067043, 0.11662072, -0.01326972, -0.04814038, 0.07669061, -0.00631280
 .float 0.00385189, -0.09120895, -0.07929938, 0.01988038, -0.11987744, -0.14783292, 0.04531620, -0.16198467, -0.06628362, -0.01507507, -0.11615671, -0.25882393
 .float -0.22683877, -0.00450676, 0.05483200, -0.37427956, -0.08782131, 0.02732728, -0.01110136, -0.00361367, -0.30017686, 0.09047803, -0.03943479, -0.26305680
 .float -0.20712253, 0.08678571, -0.03402500, 0.13278463, -0.29770094, 0.15555368, 0.04833590, -0.24255950, -0.26154116, 0.02254126, -0.01456237, 0.06634865
 .float -0.01723544, 0.21740848, 0.04361480, -0.29738020, -0.03024961, 0.06732795, 0.08736539, 0.03699037, -0.08851759, 0.09356312, 0.05917802, -0.22692163
 .float 0.03067815, -0.06723785, 0.14575835, -0.17404665, 0.01474489, 0.24197583, 0.01533457, -0.00464495, -0.09457236, -0.04566441, 0.13007538, -0.19434175
 .float -0.00486347, -0.02787312, -0.00875856, 0.16480839, -0.10797717, -0.06016624, 0.03916853, 0.06653900, 0.07897584, 0.07879274, 0.06152974, -0.12111544
 .float -0.03827895, -0.00257527, -0.03619529, 0.07475878, 0.11603656, 0.02669682, 0.10582364, -0.03674187, 0.11983844, -0.39314693, 0.06739736, 0.05040363
 .float 0.12498672, 0.01088629, 0.08195275, -0.04063445, 0.06970739, -0.37114940, 0.07510751, -0.02940780, 0.03906687, -0.01239786, 0.23147535, -0.03843179
 .float 0.01562633, -0.10146044, 0.02462354, -0.04544555, -0.09958554, -0.09641536, 0.27381784, -0.17375916, -0.06787344, -0.22657270, -0.09183310, -0.26414227
 .float -0.36842250, -0.04591918, -0.00128292, -0.49878490, -0.23096490, -0.15919861, -0.20388722, -0.06268971, -0.16489503, 0.23801585, 0.04684873, -0.56510760
 .float -0.20134630, -0.00770424, -0.15583865, 0.19244638, -0.01475635, 0.14476362, 0.20826843, -0.28671414, -0.04371503, -0.02949764, 0.05360918, -0.03977216
 .float -0.05096819, 0.17705716, 0.20107813, -0.36367974, 0.09723410, 0.01268179, 0.11166935, 0.03414055, -0.04039577, 0.17899542, 0.16767548, -0.29180694
 .float 0.10074160, 0.09499889, 0.15749738, 0.07668009, -0.11964068, 0.20114522, 0.07532373, -0.08717608, 0.11885754, -0.09219441, 0.11636947, -0.12763329
 .float 0.07513639, 0.00241526, -0.10070653, 0.10759062, -0.19781683, -0.04125041, -0.02091423, 0.06040439, 0.14916010, 0.03957104, 0.10966977, -0.08627135
 .float 0.06523024, -0.04565772, -0.12309068, 0.03551006, 0.03051486, 0.14386200, 0.19521897, -0.24937427, 0.09437550, -0.09779166, 0.09220729, 0.08705199
 .float 0.13817589, 0.01355683, 0.14505814, -0.31763650, 0.14896362, -0.22113714, 0.02029043, -0.00680229, -0.04272604, 0.07345512, 0.20116802, -0.29810005
 .float 0.05603036, -0.24255314, 0.06131525, -0.10086431, -0.23978478, 0.02564552, 0.20135179, -0.26596844, -0.12119386, -0.23114050, 0.05861718, -0.22513093
 .float -0.09077065, -0.12816752, -0.14794028, -0.16381563, -0.32907796, -0.27300087, -0.12337734, -0.16290018, 0.00723082, 0.08279820, 0.07149886, -0.12611279
 .float 0.04034252, -0.33164250, -0.09193965, -0.05634337, 0.05751352, 0.19750322, 0.16818290, -0.23097742, 0.03310916, -0.17145135, -0.00796064, -0.04419658
 .float 0.16117232, 0.17449380, 0.08334880, -0.19121608, 0.01842632, -0.01838695, 0.09831686, 0.05298407, -0.00982697, 0.21634203, 0.22258052, -0.49100290
 .float 0.17964135, 0.02135514, 0.06821485, 0.24405685, -0.22600831, 0.07946304, 0.14802629, -0.18880391, 0.07785273, 0.05402355, 0.16328327, 0.15839294
 .float 0.12447892, -0.03843751, 0.03133978, 0.12392297, -0.03252013, 0.03834965, 0.04276164, 0.04223302, 0.08142982, 0.05947954, 0.09909473, -0.23588496
 .float 0.03841156, -0.04698436, -0.00221143, 0.06010523, 0.03220781, 0.06596487, 0.13650464, -0.44505516, 0.09752598, 0.00538096, 0.13340797, 0.06440719
 .float 0.00506701, 0.09032223, 0.11438458, -0.33223110, 0.03928115, -0.03389422, 0.08105871, 0.03442470, 0.02452136, 0.12032231, 0.27506757, -0.16037773
 .float 0.10378951, -0.19029772, 0.16242328, -0.11370626, -0.05002193, 0.05484064, 0.15825036, -0.11341324, -0.10092964, -0.15593596, 0.08919933, -0.26550955
 .float 0.02862352, -0.03445680, -0.02079243, 0.01817348, -0.04225084, -0.01377239, 0.02003840, -0.13727680, 0.09565199, 0.12194383, 0.10502999, -0.02150835
 .float 0.00534985, -0.22050643, -0.04557864, -0.09958106, 0.00390190, 0.09387556, 0.07383503, -0.15004590, 0.14322910, -0.12069570, 0.02615310, 0.08809562
 .float 0.00832905, 0.14978820, 0.02907496, -0.09519940, 0.04150601, 0.00179887, 0.10139034, 0.13664840, 0.07028655, 0.15336879, 0.14245728, -0.53784610
 .float 0.04403168, 0.05709100, 0.09785293, 0.10136588, -0.33968645, 0.10739983, 0.08036679, -0.28497347, 0.02030198, 0.04491596, 0.15399103, -0.02149749
 .float 0.12562300, 0.07283349, 0.03118535, 0.03235501, 0.13241239, 0.12921757, 0.06168836, 0.03084714, -0.01133598, 0.03715490, 0.09705799, -0.31398860
 .float -0.02781663, -0.01926239, -0.05948725, 0.02774715, -0.12370859, 0.12765987, 0.02852651, -0.62384850, -0.02383461, -0.04372300, 0.00042765, 0.15386040
 .float -0.09377141, 0.07088082, 0.15469097, -0.22589925, -0.04988771, 0.10413799, 0.07720402, 0.15620817, 0.16174376, 0.22158289, 0.15899765, -0.12521803
 .float 0.14152747, 0.06916495, 0.18109062, 0.18571159, -0.01345523, 0.05413760, 0.16710259, -0.11567771, 0.10879419, 0.08537229, 0.11773543, -0.07805385
 .float 0.10933027, -0.03665997, 0.18694165, -0.05016238, 0.11368640, -0.11757290, 0.02404587, 0.01964906, 0.02150510, -0.08973304, 0.08253156, -0.09393526
 .float 0.06077643, -0.06770081, -0.09868789, 0.07503128, 0.11906328, -0.08586154, 0.09990419, 0.09155133, -0.02726558, 0.01520440, -0.12076250, 0.04410140
 .float -0.03572428, 0.03966656, 0.02580653, -0.06017642, -0.01454829, 0.06846503, -0.03539087, 0.03876720, -0.04266096, 0.02199878, 0.11470473, -0.40676100
 .float 0.06963504, 0.01131308, -0.02502904, 0.04282727, -0.25612336, -0.05327054, -0.01357478, -0.30355313, -0.01877326, 0.17909823, 0.11018976, -0.00732939
 .float 0.07127105, 0.09492002, -0.08256656, -0.11088587, 0.16054830, 0.16588438, -0.05112712, 0.19317918, 0.06624982, 0.06067278, -0.06486836, -0.38879362
 .float -0.06325813, -0.02366736, -0.01318200, 0.09476968, -0.11206667, -0.00091181, 0.18028606, -0.52381220, -0.08859792, -0.00698597, 0.12082136, 0.03018239
 .float -0.10447680, 0.07434896, 0.20086704, -0.33747143, -0.01659037, 0.00968465, 0.16153707, 0.12258532, 0.01295664, 0.12369587, -0.03470752, -0.14628321
 .float 0.08387296, 0.06974263, 0.16899467, 0.09515575, -0.04666991, 0.00468717, 0.02256038, -0.07427034, 0.06135580, 0.09376042, 0.02745416, 0.11056682
 .float 0.03119149, -0.23034362, 0.00401293, 0.03452430, 0.09401918, -0.05373097, 0.05400237, 0.07168168, 0.07725607, -0.24518102, 0.01233330, 0.10771485
 .float 0.05745672, -0.17067581, -0.01896854, -0.06179532, 0.03242468, -0.20581917, -0.10186168, 0.09067752, -0.01175412, -0.09350353, -0.05091583, -0.17131624
 .float 0.11171538, -0.19074148, -0.14583729, 0.00771652, -0.06859248, -0.10649606, -0.12458138, -0.10600153, -0.09464595, -0.17853749, 0.01347796, 0.03323065
 .float 0.05284680, -0.14488186, -0.16783094, -0.18651074, -0.02842009, -0.10358576, -0.14351820, 0.03146394, -0.06048975, -0.09931115, -0.07010080, -0.07228268
 .float -0.05490015, 0.11807234, 0.01048589, 0.05558236, 0.10089495, -0.19169469, 0.03706728, 0.12474433, 0.09391175, 0.13814835, 0.09779318, -0.18437624
 .float -0.00510699, 0.10794561, 0.00430698, 0.03637094, -0.12375185, 0.06774207, -0.08014667, -0.35015514, 0.05382069, 0.14367800, 0.04001020, 0.06295946
 .float -0.23329791, 0.02645557, -0.15549715, -0.23119515, -0.08426001, 0.17295782, 0.13236551, 0.05558713, -0.38745207, 0.12037482, -0.25698447, -0.06178935
 .float -0.03292476, 0.10734217, -0.01854542, 0.08998014, -0.29028260, 0.01266836, -0.39017236, 0.14823894, -0.01264849, 0.15938903, 0.00138691, 0.00703346
 .float -0.07603549, -0.33166030, -0.33981588, 0.28368944, -0.01926878, 0.06776288, -0.09563123, -0.01892925, -0.06256583, -0.39444260, -0.10245954, 0.24971637
 .float -0.06235938, 0.01066010, -0.14465642, -0.05713263, 0.07673800, -0.43005583, -0.09721010, 0.40212473, -0.12129661, -0.03392028, -0.24084498, -0.08606309
 .float -0.06982689, -0.36241046, -0.27534246, 0.15039120, -0.11996491, -0.15274973, -0.24260378, -0.16401456, -0.16892515, -0.30118784, -0.18241683, 0.06969172
 .float -0.23111096, -0.16378844, -0.17792821, -0.13277277, -0.05103292, -0.21060885, -0.18351870, 0.17509186, -0.07116310, -0.09294645, -0.11557696, 0.02108227
 .float -0.08237153, -0.11245017, 0.05415855, 0.11758644, -0.07273137, -0.16794015, -0.05009554, -0.17092705, -0.14971495, 0.05139025, -0.00390956, 0.04192655
 .float 0.02585474, -0.30552968, -0.17767274, 0.11571509, -0.58300954, 0.00089112, 0.15310940, 0.02593553, -0.02680123, -0.09858032, -0.08904982, 0.04727770
 .float -0.63156730, -0.00952686, -0.06847818, -0.13488083, -0.10345033, -0.05650796, 0.00465480, 0.08809654, -0.68061095, 0.04031402, -0.29808780, -0.24684022
 .float -0.19139752, 0.03380176, 0.07698619, -0.00758301, -0.66810155, -0.03718222, -0.39224908, 0.01271874, -0.10161661, -0.05153071, -0.12770517, -0.04881147
 .float -0.55164860, -0.24126019, -0.49326128, 0.27770102, -0.18120487, -0.09167536, -0.37611184, -0.01959688, -0.40826230, -0.48539063, -0.51833856, 0.29518200
 .float -0.14935067, -0.15843926, -0.27706167, -0.17332631, -0.25424558, -0.51359430, -0.33032480, 0.14109524, -0.27906263, -0.22168696, -0.27670816, -0.21119463
 .float -0.18238020, -0.38322538, -0.48485100, 0.13880572, -0.19478859, -0.38618490, -0.52548707, -0.24968566, -0.02843422, -0.31188875, -0.27161106, 0.12154724
 .float -0.24558310, -0.32259023, -0.33803150, -0.24360052, -0.00710133, -0.25254697, -0.23430108, 0.09306819, -0.08980893, -0.12541151, -0.16566372, -0.19266438

# Row 2 of weights matrix
 .float 0.19872472, 0.15179631, 0.14722998, 0.36744510, 0.11951321, 0.14243690, 0.03951085, 0.22143269, 0.04831471, 0.05129827, -0.02510712, 0.23529227
 .float 0.04358270, -0.11967550, 0.09063907, -0.04867195, -0.09913665, -0.14482670, 0.14901449, -0.01678030, 0.11980294, 0.03440675, -0.16801558, -0.18821174
 .float -0.19215076, -0.11736399, 0.36091897, -0.01141185, -0.07723199, -0.13031875, -0.12305974, -0.02458675, -0.01909394, -0.05109790, 0.40859622, -0.06068060
 .float -0.00369413, -0.08600953, -0.21508344, 0.17034910, -0.08595216, 0.05273480, 0.35381293, -0.02273314, 0.08136072, -0.13319325, -0.05314023, 0.24275064
 .float -0.08492339, -0.03440551, 0.20179476, -0.11320419, 0.13978682, -0.37917018, -0.02531102, 0.13159712, -0.05296751, -0.02601061, 0.13957900, 0.01483656
 .float 0.00905360, -0.18536706, 0.01239532, 0.07733145, -0.00908364, 0.07130535, 0.02071559, 0.00406729, -0.00354719, -0.13940242, -0.09490786, 0.20646311
 .float -0.09560914, -0.00250828, 0.05989715, 0.00891744, 0.04890519, -0.18541986, -0.12654190, 0.17047472, -0.12655826, -0.00581399, -0.09395170, -0.03929184
 .float -0.02271036, -0.13639155, -0.12978198, -0.15554076, -0.13634017, -0.18736951, -0.08698901, -0.00031613, -0.03703811, -0.15477702, -0.09084657, -0.27312887
 .float 0.14976174, 0.03891419, 0.13516110, 0.19463160, -0.00041009, 0.17939925, -0.01824049, 0.02740911, 0.02983023, -0.13195685, 0.06033053, 0.09468622
 .float -0.05064342, 0.09291155, -0.00720892, -0.12677608, -0.19124095, -0.16032220, 0.12406650, -0.02015889, -0.28176500, -0.05071058, -0.10645594, -0.16084379
 .float -0.00731449, 0.02480309, 0.07469311, 0.05244936, -0.22553952, 0.08212771, -0.08553749, 0.15289615, -0.01723252, 0.17371625, 0.25023475, -0.00271552
 .float -0.16166589, -0.11685894, -0.11617374, 0.06304882, 0.08789116, 0.18015458, 0.38178033, -0.05138912, 0.08526270, -0.22009873, -0.05291718, -0.10449599
 .float -0.03535735, 0.08108366, 0.39073157, -0.15376242, 0.02597212, -0.37572357, 0.01404186, -0.16547197, 0.06858143, 0.08748558, 0.20477965, -0.11793043
 .float 0.03301497, -0.40848020, -0.04474612, -0.14255041, -0.02310513, 0.06388305, 0.09543773, -0.02153441, 0.13096856, -0.49010830, -0.04964083, -0.17464994
 .float 0.03206249, 0.02875254, 0.17814952, 0.12994750, 0.07919646, -0.40091684, -0.03440651, -0.08375946, -0.02972669, 0.01200340, 0.03877834, -0.00655332
 .float 0.08826093, -0.35846162, -0.00601769, -0.17510216, -0.10430476, -0.18669190, -0.07572588, 0.01148292, -0.09812406, -0.34670436, -0.07696613, -0.19463135
 .float 0.07839033, -0.04539856, 0.19391450, 0.11765167, 0.16544329, 0.00960530, -0.02152586, 0.05711173, -0.01648752, 0.06828467, 0.10479987, 0.04980624
 .float -0.17764854, 0.01399074, 0.01849492, 0.05731727, -0.13906676, -0.03375185, -0.18764374, 0.04739887, -0.33120695, -0.15574240, 0.03959361, -0.00369968
 .float -0.00536783, 0.03355577, -0.48739895, -0.06918215, -0.22171408, 0.01784824, 0.05905537, -0.02524048, -0.06255060, 0.09167404, -0.17575519, -0.23240662
 .float -0.09986642, -0.10050415, -0.07120888, 0.00164742, 0.08667547, 0.07452059, 0.07738031, -0.24904563, -0.06118432, -0.19957843, -0.09647933, -0.03815564
 .float 0.10768520, 0.14761685, 0.15376653, -0.13826240, -0.01870366, -0.42105138, -0.07792616, -0.20127049, 0.13168639, 0.13390927, 0.02015357, -0.18456283
 .float -0.06708546, -0.40212655, 0.08730358, -0.42400390, 0.06548961, 0.09513424, 0.09532255, -0.21908255, 0.13209882, -0.43334990, -0.00764522, -0.43132547
 .float 0.05436361, 0.08466727, 0.12891772, -0.03486134, 0.07217852, -0.20857196, -0.00073801, -0.23225011, -0.05568866, 0.11490031, 0.25509351, -0.06265703
 .float -0.04596251, -0.24280295, 0.02350564, -0.15821055, -0.18598807, -0.17842175, -0.07041546, 0.03404162, -0.02832753, -0.25129628, 0.00434144, -0.21137658
 .float -0.13430417, -0.05708848, 0.05933431, 0.24321488, 0.06648803, -0.14199059, -0.02526315, -0.03808434, -0.11553799, -0.01986896, -0.05673810, 0.17594597
 .float -0.17738575, 0.08236543, -0.10354357, -0.03197546, -0.15439785, -0.03270492, -0.39928672, 0.16857614, -0.16305490, 0.18074597, -0.02151893, -0.13091077
 .float -0.02522414, 0.10842127, -0.39730948, 0.02790106, -0.15241021, 0.04764968, 0.02101500, 0.10691874, -0.06295080, 0.01408246, -0.34651893, -0.29890764
 .float -0.11133220, -0.04961052, 0.05597621, 0.02437330, 0.10162082, 0.12148625, -0.24152038, -0.22698945, -0.17529540, -0.20572515, 0.02120347, 0.03813585
 .float 0.05376788, 0.12631957, -0.08713283, -0.06295612, 0.01494488, -0.28554600, 0.05255912, -0.06466439, 0.28024748, 0.16441864, -0.07127204, -0.12879550
 .float 0.06370863, -0.21652089, 0.05155025, -0.39764422, 0.15948847, 0.03677425, -0.04049997, -0.42321146, 0.11042894, -0.23932967, 0.04436921, -0.56226224
 .float 0.04477074, 0.10314180, -0.00180213, -0.22547662, 0.07351451, -0.15283947, 0.08834859, -0.18959408, -0.03939359, -0.03671976, 0.13778731, -0.00518874
 .float -0.00156306, 0.01227517, 0.06992929, -0.18130633, -0.16298717, -0.15985869, -0.10970851, 0.18791448, -0.14335182, -0.16343114, -0.03075886, -0.12934321
 .float -0.07028261, 0.00366952, -0.12144196, 0.16199285, -0.07981168, -0.43288430, -0.25466920, 0.02682515, -0.02748689, 0.01388058, -0.07300814, 0.16261917
 .float 0.02120666, -0.12140438, -0.07517783, -0.11126240, -0.14527630, 0.02407605, -0.10087915, 0.17243002, 0.06113363, 0.03386825, -0.08502337, 0.11156464
 .float -0.05441351, 0.05264695, -0.10211145, -0.03778938, -0.06842081, 0.03906303, -0.03990297, 0.07448626, -0.01067040, -0.07306775, -0.29399234, -0.08358835
 .float -0.03412207, -0.07801387, -0.04459860, 0.11229065, 0.09641368, 0.12377655, -0.03688084, -0.01281324, 0.01943917, -0.16988176, 0.03440638, 0.17465360
 .float 0.17534754, 0.08274462, 0.04219562, 0.01289781, -0.00508427, -0.34093696, 0.13124941, -0.05497404, 0.22138856, 0.16219208, -0.01096030, 0.11345327
 .float 0.08269910, -0.26295224, 0.10918100, -0.21561696, -0.00387916, -0.04974041, -0.02100484, -0.21939449, -0.05244211, -0.01636957, 0.03684510, -0.44101697
 .float -0.05022258, 0.03608543, -0.06030029, 0.00261399, -0.00498545, -0.02079088, -0.05940166, -0.19284599, -0.07753983, -0.18851689, 0.03117824, 0.15195110
 .float 0.05300337, -0.00340577, 0.04448367, -0.18690465, -0.06022923, -0.15920410, -0.15258612, 0.25899630, -0.12930608, -0.03259232, 0.04922831, 0.04177244
 .float 0.11486661, 0.05423494, 0.13077340, 0.24924730, 0.07362314, -0.13853644, -0.03469597, -0.00065802, -0.10204969, -0.17254092, 0.09843216, 0.15875514
 .float -0.05370912, -0.01117530, -0.09504543, -0.18717666, -0.00207774, -0.11917496, 0.07825027, 0.19109496, -0.11124401, -0.02913351, 0.05943941, -0.07199742
 .float -0.07626796, -0.12938435, 0.01453028, -0.01566016, 0.11429537, -0.26067582, -0.25201060, -0.11570755, 0.05218142, 0.02789053, 0.03659044, 0.08019558
 .float -0.06076393, -0.25543612, -0.20502860, 0.09957906, 0.13773754, 0.21178721, 0.16088593, 0.13228783, 0.08526608, -0.33287993, -0.01321036, 0.10051642
 .float 0.18607290, 0.20675215, 0.11891272, 0.06399608, 0.00361405, -0.39003930, 0.11630064, -0.04755001, 0.06020907, 0.15338735, 0.01903543, -0.00352437
 .float 0.02666936, -0.19598298, 0.18024382, -0.39880050, -0.02277240, -0.02316184, -0.09083165, -0.29400042, -0.13486089, -0.21237580, 0.00676560, -0.35626575
 .float -0.06740281, 0.10431960, 0.02101213, -0.23417710, -0.09707386, -0.11881143, -0.04610925, -0.09493128, -0.15743700, -0.08939989, 0.03296109, 0.04982316
 .float -0.16604424, 0.04716792, 0.08592658, 0.01295914, -0.12344486, -0.16202593, -0.10730788, 0.28414586, -0.15895534, -0.23702057, -0.03051014, 0.05802325
 .float 0.08240651, 0.07307519, 0.27419930, 0.13616581, 0.09713101, -0.00212489, -0.07098052, 0.07822518, -0.14233583, -0.21061590, 0.13587111, 0.05287457
 .float -0.07943580, -0.24718344, -0.10179070, -0.14254947, 0.01525017, -0.09715321, -0.14954403, 0.05095897, 0.08966165, 0.07950084, -0.09047274, -0.02923832
 .float -0.04598544, -0.01922228, 0.22489190, 0.02413876, -0.03319014, -0.15742289, -0.24647786, -0.09341087, 0.12557329, 0.11186597, -0.07423754, -0.06007403
 .float -0.10621571, -0.54997510, -0.21774453, 0.07906617, 0.18341747, 0.22712898, 0.23869368, -0.03774308, 0.00294746, -0.36242124, -0.02773278, 0.02709281
 .float 0.06910626, 0.15349026, 0.14348759, -0.40462348, -0.03227559, -0.31477797, 0.03018425, -0.18350708, -0.09880926, 0.13270900, 0.00548733, -0.27922340
 .float 0.01364720, -0.18069004, 0.20421638, -0.26695950, -0.00118596, 0.06949502, -0.12169489, -0.28175047, -0.21046291, -0.19070755, 0.00193912, -0.21712996
 .float -0.19851214, 0.00271999, 0.04259536, 0.00305860, -0.17676884, -0.13056514, -0.12217259, -0.33463810, -0.05213093, -0.06784529, -0.09656154, 0.09582532
 .float -0.10577675, 0.02616736, -0.09782677, 0.04222758, -0.09006256, -0.01543317, -0.06364147, 0.20262484, -0.10107375, 0.01768456, -0.21257684, -0.01678687
 .float -0.09955900, -0.33542776, 0.21685535, 0.01309426, -0.09980722, -0.01636049, -0.40011275, -0.24548364, -0.16770117, -0.30882100, -0.13899978, 0.00974639
 .float -0.16670665, -0.03963356, -0.13816762, -0.20329687, -0.02084034, -0.01784935, -0.29167333, 0.03026173, -0.06579740, -0.01350308, -0.06687763, 0.08039527
 .float -0.01273764, -0.03648384, -0.06893399, -0.07783031, -0.06780888, -0.17841361, -0.12862077, -0.09102639, -0.02432764, 0.04216700, -0.08998159, -0.12395459
 .float -0.03262328, -0.43579286, -0.09105134, -0.02111509, 0.05967577, 0.20197557, 0.08602130, -0.46531802, 0.07180759, -0.21147862, -0.13803756, 0.12779514
 .float 0.05758755, 0.17476125, 0.21521637, -0.26093718, -0.04273872, -0.14326957, 0.02629170, -0.12271782, 0.04078509, 0.04777063, -0.08473216, -0.22832897
 .float 0.05249636, -0.06462295, 0.17580535, -0.27645746, -0.06746206, -0.01350659, -0.16926555, -0.09731173, -0.10466079, -0.18613700, -0.06861378, -0.55231180
 .float -0.15374959, -0.23038155, -0.03271617, 0.19115102, -0.15499589, -0.09019428, 0.03210504, -0.26533985, -0.05208804, -0.01966316, -0.10513233, 0.19779300
 .float -0.13318920, 0.07794209, 0.00234207, 0.12914683, 0.11298449, 0.11631222, -0.01080975, 0.17951353, -0.03876774, -0.01707489, -0.06890640, -0.00392527
 .float -0.21970151, -0.23070118, -0.11902287, 0.23259437, -0.12418413, -0.28568393, -0.00349242, -0.16928375, -0.00044398, -0.09626012, -0.04467947, 0.14150620
 .float -0.02400722, -0.08192311, 0.00403448, -0.06931879, 0.05710902, -0.10818437, -0.17126660, 0.11356413, -0.08217095, 0.17776330, -0.09814326, 0.01214963
 .float 0.02958033, -0.03028973, -0.14473984, 0.13570370, -0.07825816, 0.07022255, -0.09192547, -0.12006842, -0.09255512, -0.00556856, -0.03120794, 0.15774949
 .float -0.13982733, -0.11158581, -0.11743066, -0.07768033, 0.15376198, 0.26464778, 0.02975563, 0.10539515, 0.00084707, -0.21742360, -0.02793087, -0.06671924
 .float 0.20666921, 0.28658480, -0.16568057, 0.10261442, -0.05184221, -0.01192620, 0.16134189, -0.19194785, -0.18570377, 0.05575555, -0.28571716, -0.27470347
 .float -0.12707411, 0.09199130, 0.09732339, -0.24941339, -0.02273202, -0.02168555, -0.04866434, 0.11602024, -0.09662864, -0.10648876, 0.00582896, -0.22243586
 .float -0.04161536, -0.16418706, -0.15602419, 0.15265080, 0.00864269, 0.09852370, 0.03846624, 0.01408207, 0.12253079, 0.02225994, -0.28297606, 0.20283693
 .float 0.03979871, 0.06856763, -0.09018601, -0.02173271, 0.02113382, 0.02010540, 0.10643650, 0.22449589, 0.01983407, 0.07589646, -0.02862956, -0.02045266
 .float -0.05737756, -0.01589183, -0.17059967, 0.24808826, -0.11177487, 0.05831095, 0.06621533, 0.05966880, 0.08446043, -0.04122486, 0.12835085, 0.16871765
 .float 0.00824398, 0.11738498, -0.13969450, -0.05002947, 0.09689367, -0.02750007, 0.17369480, 0.12842165, 0.10457557, 0.08956392, -0.19709489, 0.09226754
 .float 0.00046357, -0.01720672, 0.08335926, -0.03576375, -0.03654303, -0.07120971, -0.07721149, -0.01848620, 0.10188504, -0.01628020, 0.05173040, 0.06990030
 .float -0.11781336, -0.16069618, -0.03211218, -0.02262820, 0.19395079, 0.15328518, -0.23318036, 0.11300520, -0.27670747, -0.17900674, 0.00718719, -0.08771100
 .float -0.03602153, 0.14529850, -0.31374730, -0.16095062, -0.11540278, 0.07605708, 0.01637583, -0.13821769, -0.34733970, 0.08181395, -0.30463720, -0.10553889
 .float -0.09153865, 0.15592405, 0.13883099, -0.02375462, -0.10881740, 0.05465126, -0.18191620, 0.06123200, -0.05052928, 0.23729308, 0.12846138, 0.20636363
 .float -0.17169231, -0.12415545, -0.20506552, 0.09336552, -0.01290740, 0.19329469, 0.14191487, -0.01562632, 0.05184653, 0.11512972, -0.23970972, 0.11824638
 .float 0.00088094, 0.00890792, -0.00196934, 0.07467663, -0.10683625, 0.09950263, -0.05845306, 0.05043367, -0.12051939, 0.07299953, -0.01047409, 0.10478760
 .float 0.07304972, -0.13133326, -0.00200741, 0.16340174, 0.04556249, 0.00535770, 0.01112901, 0.01108038, 0.02186064, -0.05169317, 0.13465197, -0.13865377
 .float 0.07742067, 0.13293608, 0.02492174, 0.08793540, 0.09541358, -0.08013125, 0.21584298, -0.06765485, -0.00059537, -0.06595531, 0.00496922, 0.05275258
 .float 0.06489844, 0.02564221, 0.15676132, 0.00106815, -0.02879304, -0.07265144, -0.12564576, 0.01738106, 0.02750811, 0.10841527, 0.04721744, -0.07171647
 .float -0.11027330, -0.06213384, 0.01763235, -0.11314560, 0.09653159, 0.08216552, 0.00398794, -0.21811666, -0.20568761, -0.04627772, 0.13806336, -0.10648502
 .float -0.05238583, 0.07468326, 0.05341585, 0.03781186, -0.14517373, 0.09723267, 0.08051356, -0.04737034, -0.18977967, 0.19550484, -0.21779257, 0.05570317
 .float -0.09287636, 0.07487866, 0.14579916, 0.03961985, -0.03090953, 0.26570055, -0.18214375, -0.08115812, -0.22271244, 0.14145255, 0.13986446, 0.08416615
 .float -0.12446269, 0.03845723, -0.04712259, -0.00193295, -0.20814833, 0.06301226, 0.16530056, -0.10733769, 0.07182554, 0.08920806, -0.33136490, 0.18515746
 .float -0.20269974, -0.04370660, -0.00329939, -0.01751873, 0.09598555, 0.04781148, 0.01206338, 0.27563664, -0.12547094, -0.07255220, -0.19525644, -0.12135235
 .float -0.01876928, -0.07925573, 0.05103006, 0.25634766, 0.01762546, -0.27131623, -0.02270891, -0.05716852, -0.06480122, 0.05926448, 0.05130535, 0.24739286
 .float -0.07063937, -0.30166385, -0.03038655, 0.08245316, 0.17901740, -0.03424813, 0.03296554, 0.26991796, 0.00317933, -0.11728719, -0.02280174, -0.04301768
 .float -0.01115019, -0.00138090, 0.14123689, 0.16276276, -0.00082121, -0.10221458, -0.01831780, 0.00221233, 0.11141464, 0.03173640, -0.04076766, 0.14648049
 .float 0.01756434, 0.06497438, 0.09785172, -0.03673831, 0.00367318, 0.03995484, 0.11729159, 0.25226387, 0.05666139, -0.00875631, -0.08739278, 0.03713273
 .float -0.13013947, -0.05128954, 0.07400970, 0.18706772, -0.10233957, -0.08093715, 0.11336371, 0.10563472, 0.12846863, 0.10678456, 0.03071086, 0.14024659
 .float 0.01164518, 0.01995810, 0.05768707, 0.12262866, 0.11452982, 0.24316671, 0.08311123, 0.10727508, 0.08252928, 0.11014548, -0.02570199, -0.05871325
 .float 0.08624808, -0.01655665, 0.13640505, 0.29742643, -0.04084920, 0.15781802, 0.06937458, -0.03082998, -0.07357727, -0.22814919, -0.18097540, 0.23133427
 .float -0.07049663, 0.05437836, 0.07789499, -0.07819542, 0.19834290, 0.04406689, -0.02164956, 0.33712047, 0.02944758, -0.29863757, -0.16559836, -0.12520500

# Row 3 of weights matrix
 .float -0.13502511, -0.11239482, -0.03805478, -0.05625566, -0.01874831, -0.20451310, -0.08010001, -0.16911474, 0.01868102, -0.03687683, 0.00805191, 0.00017049
 .float -0.06007952, -0.00409409, -0.02983258, 0.00863492, 0.01042185, 0.04448996, -0.15153943, -0.01458366, -0.03031836, 0.24116065, 0.01873667, 0.06132520
 .float 0.16476136, 0.09643123, -0.11851627, 0.12585660, 0.02692717, 0.27445054, 0.12936695, 0.19274136, 0.02969330, 0.15799690, -0.04769259, 0.14307404
 .float 0.05730082, 0.23159172, 0.12313313, 0.10802084, -0.05111535, -0.04151818, -0.01840439, 0.05367154, 0.01568806, 0.18846396, 0.12998874, 0.00513665
 .float -0.00465943, -0.01955070, -0.24798307, 0.00510406, -0.05153257, 0.23955208, 0.07218619, 0.04431952, -0.09060027, -0.01108119, -0.14429980, -0.07584312
 .float -0.02369368, 0.09803319, 0.00876555, 0.03022477, -0.16054355, -0.12479287, -0.06415307, -0.07992929, -0.05911163, -0.15433820, -0.03981961, -0.44838450
 .float -0.02644210, -0.11460419, -0.17313217, -0.11548782, -0.13731290, -0.12920646, -0.05593386, -0.36888966, -0.04228231, -0.11838256, -0.11427468, -0.11322906
 .float -0.07683542, -0.22457685, -0.11185513, -0.50180113, 0.05042643, -0.10724641, -0.07617052, -0.07539197, -0.10076439, -0.01325810, -0.03291058, -0.00865906
 .float 0.01843386, -0.03284021, -0.03307433, -0.03673919, -0.06435058, -0.06148605, 0.10687169, -0.08157737, 0.04212641, 0.00431401, 0.02119204, 0.05827028
 .float 0.03457308, -0.02735053, 0.06448849, 0.03616323, -0.02188105, -0.00286704, -0.20686671, 0.01507339, -0.01613497, 0.05260184, 0.04953399, 0.05325855
 .float 0.06176571, -0.03002744, -0.33250698, -0.02450899, -0.08012269, 0.02211014, 0.00507210, 0.18607083, 0.00192478, -0.05101474, -0.17566654, 0.07126081
 .float 0.06999131, -0.00703716, 0.09659836, 0.07507074, -0.07937878, -0.04123730, -0.16532136, -0.05445752, 0.09884331, 0.02908694, 0.09318566, 0.07609811
 .float -0.08553529, 0.02744179, -0.24141166, -0.03878884, -0.05435496, 0.03117649, 0.04979179, 0.09389421, -0.12131821, -0.06556531, -0.08140885, -0.08361035
 .float -0.11664880, 0.20628396, 0.07973249, 0.11161528, -0.08477113, -0.21609843, -0.20290550, -0.01389472, -0.10793968, 0.08031412, -0.03715096, 0.02958640
 .float 0.06289376, -0.13779412, -0.21716918, -0.02515723, -0.08994502, 0.10738232, -0.05276974, -0.07449918, -0.05432958, -0.22757967, 0.03600619, -0.11962941
 .float -0.01486154, -0.18302464, -0.00772559, -0.42179143, -0.02029840, -0.05498269, -0.29168686, -0.12810767, -0.12149309, -0.17690590, -0.02438720, -0.43416930
 .float 0.00455411, 0.06993660, 0.06372954, 0.06566547, 0.02382094, 0.01387445, 0.04973174, -0.00349120, 0.02282627, -0.01444730, 0.04552662, -0.01754006
 .float 0.16486935, -0.02513559, -0.06842061, -0.08595815, 0.01092372, -0.07269709, -0.01271850, 0.10988612, 0.13702513, -0.00681702, 0.02526326, 0.04913661
 .float 0.07274918, 0.08894809, -0.09019036, 0.11076258, 0.08488199, 0.03834565, 0.05169649, 0.03552053, 0.02856159, -0.01652303, 0.01819713, 0.13407612
 .float 0.03773094, 0.00995484, 0.06377954, -0.03418986, -0.00915690, -0.02338141, 0.10043178, 0.17195953, 0.01371851, 0.10555089, 0.10316916, 0.07610045
 .float -0.12112600, -0.02164605, 0.05652378, 0.09984984, -0.04927532, 0.10832624, 0.05263311, 0.10665119, -0.20548604, -0.04619368, -0.07203411, -0.08874408
 .float -0.06468043, 0.06633257, 0.00524503, 0.18186654, -0.18395000, -0.05998365, -0.24329357, -0.16195673, -0.24433652, -0.00374988, 0.01961817, 0.13377720
 .float -0.16514670, -0.02516432, -0.22838330, -0.33819965, -0.14768380, -0.06735380, -0.11184643, 0.14334376, -0.27207060, 0.00080211, -0.08267041, -0.20703377
 .float -0.13392346, -0.30341280, -0.04863046, -0.34352478, -0.00084926, -0.17883077, -0.22329324, -0.13952431, -0.01807964, -0.38878855, 0.01904184, -0.25575650
 .float 0.00566088, 0.02571765, 0.01680166, 0.06429352, 0.06859168, 0.06667468, -0.00634147, 0.02181600, -0.00689541, -0.08309311, 0.10417994, 0.13390532
 .float 0.10614942, -0.06563428, 0.01871975, -0.05781268, 0.09511254, -0.05680204, 0.04200784, 0.15427917, 0.03477440, -0.13398610, -0.05778589, -0.02908800
 .float 0.05865148, -0.09802943, -0.08860328, 0.21392946, -0.00781922, -0.05137234, -0.03705400, 0.01808306, 0.05240773, -0.04042281, -0.00744637, 0.13260002
 .float 0.01629417, -0.04623468, -0.03822609, -0.12458160, -0.07165333, -0.13427506, 0.09802061, 0.09608785, -0.07703194, -0.00050423, -0.04959704, 0.07997098
 .float -0.19107075, 0.13427892, -0.05019180, -0.06287058, -0.14257455, 0.05072659, 0.00954068, 0.06584150, -0.19923596, 0.13939926, 0.03622956, -0.10206010
 .float -0.07312125, 0.06230578, -0.07283650, 0.12414869, -0.07772495, 0.03079170, 0.00193113, -0.16689803, -0.16711663, 0.00824246, -0.05397829, 0.06563249
 .float -0.04694507, 0.01274125, 0.00543461, -0.30145760, -0.06764151, -0.16165833, 0.00526847, -0.03047918, -0.15162158, 0.00429579, 0.05204865, -0.29400823
 .float -0.00780304, -0.20145509, -0.06835595, -0.40772190, -0.07705174, 0.02923851, -0.03473975, -0.12076718, 0.08197195, -0.35623267, -0.06676172, -0.06841260
 .float -0.00862845, -0.22929354, 0.00686647, -0.05458498, -0.06256422, 0.03162789, -0.09083685, -0.14781447, -0.02013506, -0.19431077, 0.04276241, 0.14929317
 .float 0.07499978, 0.16223209, -0.16469628, -0.18386902, 0.04761097, -0.13032521, -0.04735446, 0.47929418, -0.03812513, 0.06922158, -0.18116513, -0.11256249
 .float 0.03638730, -0.16281958, -0.22411616, 0.30511102, -0.08871744, -0.07965120, -0.24430783, -0.05850937, -0.20301450, -0.24056701, -0.18085244, 0.19758846
 .float -0.04792794, 0.07552589, -0.08378935, 0.03175759, -0.16484977, -0.20067218, -0.05366541, 0.11622699, -0.09919217, -0.06258343, -0.13905561, 0.07554907
 .float -0.09704425, -0.02474913, -0.14474541, -0.12516860, -0.13466622, 0.03946870, -0.18744305, 0.05064396, -0.07575642, 0.11667957, 0.05229566, -0.23268224
 .float -0.12169481, -0.08648719, -0.09490490, 0.08417387, -0.03362903, 0.10012831, 0.04524380, -0.18442798, -0.00774786, -0.11882951, -0.06304753, 0.11085570
 .float -0.00856877, 0.14821853, 0.16049959, -0.23478128, -0.01956459, -0.18354295, -0.01622406, -0.00308937, 0.07411189, 0.10729387, 0.14883643, -0.20121147
 .float 0.04385680, -0.22917464, 0.01972292, -0.10612355, 0.03796144, 0.15558621, 0.16324973, 0.10963710, 0.06676135, 0.01144514, -0.04979691, 0.07057425
 .float -0.06719247, -0.28063074, -0.02417105, 0.09501488, -0.10205578, -0.17409569, -0.02873762, -0.24343890, -0.06805254, -0.15246955, 0.01753840, 0.37912893
 .float 0.05039119, -0.19432351, -0.04551635, -0.16536130, 0.08543377, -0.21961690, -0.25222070, 0.48750773, -0.01187197, -0.09420292, -0.20976096, -0.06598712
 .float -0.06670293, -0.22249712, -0.47459504, 0.44417647, -0.02550953, -0.07202914, -0.25820032, -0.14043723, -0.04579226, -0.31814447, -0.32464257, 0.39714155
 .float -0.02877661, -0.03009937, -0.19424805, 0.02477713, -0.10643061, -0.28115585, -0.17537303, 0.21327402, -0.05340204, -0.03605189, -0.23009380, -0.01105319
 .float -0.02236662, -0.00476383, 0.00871104, 0.13831271, -0.11657561, -0.26064208, -0.16557692, 0.00897157, 0.02645303, 0.07892064, 0.07300341, -0.10027792
 .float 0.01756517, -0.27725455, -0.13881956, -0.08544133, 0.07318325, 0.20344000, 0.15343160, -0.18323563, 0.07693521, -0.19494674, 0.00026758, -0.03344264
 .float 0.13600327, 0.04966182, 0.25261027, -0.20852193, 0.03145350, -0.26343566, -0.00790565, -0.08873370, 0.17865808, 0.05516684, 0.11718138, 0.03641607
 .float 0.08102230, -0.04904587, 0.04723382, -0.13709706, 0.11797749, 0.17997630, 0.13631926, 0.12449269, 0.19040865, 0.31205884, -0.02625674, 0.36360982
 .float -0.11135715, -0.05588840, 0.01496403, 0.14951138, -0.00193757, -0.00517203, 0.07125947, -0.15993236, 0.01584276, -0.20776180, 0.08349545, 0.26054066
 .float 0.00221696, -0.19784248, -0.10043753, -0.14559267, 0.13107000, -0.17385928, -0.12866457, 0.21593256, 0.05448366, -0.27656730, -0.18674170, -0.24712877
 .float 0.20282702, -0.30079890, -0.20763224, 0.24748808, -0.05268485, -0.39858562, -0.18157950, -0.01198428, 0.23297131, -0.26437643, -0.23168190, 0.14045516
 .float 0.10251021, -0.38475660, -0.14285278, -0.01626250, 0.01884392, -0.13795836, -0.02039140, 0.11908586, 0.00432552, -0.18796422, -0.07349711, -0.09000923
 .float 0.08859975, -0.03178560, 0.14245145, 0.05898764, 0.06261292, -0.13363990, 0.01158218, -0.10601902, 0.20466788, -0.02523452, 0.06104978, 0.02258016
 .float 0.05818205, -0.18184175, -0.07833266, -0.10957363, 0.07403310, 0.01996248, 0.14762302, -0.05391994, 0.08667800, -0.11008657, 0.02627237, -0.12013350
 .float 0.14429316, -0.05133230, 0.12024336, -0.00271655, 0.11017960, 0.03839753, -0.00656477, -0.18932213, 0.27882722, -0.20610161, -0.05591335, 0.10793828
 .float 0.02149177, 0.13394617, 0.09430558, -0.06454647, 0.33315074, 0.14151177, 0.09713261, 0.16187596, 0.09250831, 0.39599872, 0.06086148, 0.38133723
 .float -0.03495904, 0.02391345, 0.04680808, 0.02280874, 0.03017069, -0.02912288, 0.10401501, -0.07575470, 0.05395727, 0.00627139, 0.02663665, -0.01870208
 .float -0.01836747, 0.03621032, 0.02880790, -0.00186256, 0.09383254, -0.08965145, 0.00706324, 0.05759499, 0.04961546, -0.20521595, -0.10978487, -0.19567391
 .float 0.21709470, -0.13119893, 0.02139685, 0.05056870, 0.18454528, -0.34529380, -0.03086385, -0.08622043, 0.26368657, 0.04699246, 0.08655874, -0.08713385
 .float 0.10008631, -0.14613175, -0.06663682, -0.08770891, 0.12591790, -0.03159951, 0.12018681, -0.10125692, 0.14894399, -0.05663714, 0.11148114, -0.03842540
 .float 0.12190769, -0.00248186, 0.17683809, -0.15011350, 0.19579105, 0.00126773, 0.05545610, -0.09715538, 0.06727754, -0.08908187, 0.08948030, -0.06531017
 .float 0.05517726, -0.08513043, 0.05751865, 0.02398332, 0.11874796, -0.14662163, 0.02373727, 0.04487234, 0.02735257, 0.07466425, 0.06967360, -0.13410464
 .float 0.05969697, -0.10374803, 0.04625347, 0.03274850, 0.02396920, 0.19589314, 0.05192681, -0.01539728, 0.21150656, -0.25679448, -0.11650453, 0.22236529
 .float -0.08766712, 0.12719558, 0.09624264, 0.16581006, 0.26137656, 0.04079510, -0.05776066, 0.23066221, 0.08435231, 0.12971836, 0.00648012, 0.42521462
 .float 0.10224561, 0.11985701, 0.09929133, 0.05053552, 0.09622205, 0.02567457, 0.20537174, 0.05843779, 0.10522746, 0.00359695, 0.08776984, 0.02372069
 .float 0.12000240, -0.06277030, 0.10305467, 0.00256506, 0.12148812, 0.02969478, 0.17408969, -0.11675667, -0.02198203, -0.08302563, 0.05259233, -0.05561107
 .float 0.15887628, 0.12374815, 0.09316627, -0.11827832, 0.12073982, -0.10500531, 0.05107804, -0.02230647, 0.12346066, 0.16248941, 0.05596155, -0.15607093
 .float 0.19223928, -0.09470371, 0.04215639, -0.02538815, 0.18188643, 0.02850884, -0.00589116, -0.09507283, 0.08969253, -0.09629162, 0.07710438, 0.02380720
 .float 0.09892426, -0.02370423, -0.03744908, 0.04109568, 0.16912624, 0.05359264, -0.00424487, 0.02273413, 0.12669039, -0.05621743, -0.04313268, 0.24272890
 .float 0.06196652, 0.13398883, 0.16090164, 0.06429049, -0.03299735, -0.07298441, -0.04798488, 0.27513206, -0.03827849, 0.13927577, 0.07057561, 0.06688642
 .float -0.10138527, -0.07523683, -0.08075196, 0.24940084, -0.03737073, 0.15259925, 0.06617606, 0.10600254, 0.06244918, -0.08715853, -0.22497390, 0.37022690
 .float -0.17517838, 0.20638883, 0.04382406, 0.10863946, 0.20394546, -0.00122380, -0.15402302, 0.20942284, -0.05982594, 0.00994589, -0.14800668, 0.12722085
 .float 0.07280050, 0.02716939, 0.29985350, 0.02731247, 0.18161990, 0.20552966, 0.15122640, 0.00015616, 0.02418880, 0.09358485, 0.26706380, -0.17788495
 .float 0.12817864, -0.12086894, 0.07202108, -0.00065678, 0.21853900, 0.05586921, 0.04519322, -0.21739420, 0.06388496, -0.18310703, 0.02670557, 0.05692808
 .float 0.19821039, 0.17861620, 0.09975624, -0.29573990, 0.08393347, -0.08812858, 0.15484163, -0.03533759, 0.14598717, 0.06455614, -0.08247004, -0.14093536
 .float 0.16016077, -0.10503636, 0.10525228, -0.01882683, 0.08224225, -0.03007531, -0.16684190, -0.12255645, 0.01186124, -0.13280357, 0.12414338, 0.00480254
 .float -0.03091538, -0.15478112, -0.13864262, 0.12131622, 0.09752300, -0.01109510, 0.02455681, 0.03276920, -0.12450147, 0.04619480, -0.07443902, 0.27593297
 .float -0.12168338, 0.18925866, 0.17100015, 0.20662823, -0.24025154, 0.02470121, -0.10371719, 0.36087220, -0.09730068, 0.21869971, 0.12668264, 0.13022600
 .float -0.02742959, 0.05869465, -0.23093474, 0.52924460, -0.10877638, 0.15283631, 0.14201264, 0.19432834, 0.15291277, 0.03321675, -0.21948081, 0.36293330
 .float -0.28507495, 0.17230393, 0.01341606, 0.14264618, 0.23100412, -0.01089189, -0.19329889, 0.20470970, -0.10418163, -0.06789391, -0.20911816, -0.08544543
 .float -0.03049952, 0.12388626, 0.31829100, -0.19710636, 0.07336043, -0.01033033, -0.16668443, 0.11158011, 0.10071380, -0.03282554, 0.05052282, -0.46086198
 .float -0.00808924, 0.01065179, -0.18309225, 0.08600123, 0.03442224, 0.08956465, 0.07921783, -0.39246140, 0.06586710, 0.02101895, -0.06953327, 0.05746787
 .float -0.07435332, 0.07435932, -0.09944846, -0.31562307, -0.00907956, 0.02199536, -0.02363197, -0.02087233, -0.07250147, 0.10427427, -0.15679650, -0.04397246
 .float -0.01762471, -0.01400830, -0.04274333, 0.05637511, -0.00144679, -0.09777911, -0.36264628, 0.07840158, -0.00307234, 0.05457806, 0.01162195, -0.02141760
 .float -0.24365078, -0.12344529, -0.34604890, -0.05065343, -0.05007858, 0.04881720, 0.12125949, 0.00185533, -0.37793850, -0.03590758, -0.13778526, 0.11852221
 .float -0.02877521, 0.15456471, 0.08521879, 0.00341300, -0.32444438, 0.03756175, -0.05618165, 0.47582725, -0.11979505, 0.23033257, 0.13010626, 0.10988729
 .float -0.20094717, 0.14056437, -0.26197880, 0.58961030, -0.28907105, 0.18339089, 0.10219663, 0.06065551, 0.06422228, 0.08839776, -0.23030242, 0.26177680
 .float -0.12960279, 0.14486751, -0.09369491, -0.11209830, 0.12352673, 0.01948563, 0.02578679, 0.13743049, -0.08928646, -0.08475342, -0.20225996, -0.01135521
 .float -0.29776525, -0.09371305, 0.13031052, -0.07797366, -0.03971943, -0.57011175, -0.06733710, -0.00816085, -0.12957728, 0.01240009, -0.03349448, -0.03971334
 .float -0.04445698, -0.40259162, -0.09535372, -0.03818149, -0.21604721, -0.00007498, 0.05422537, -0.13990177, -0.03389937, -0.20520623, -0.26384620, 0.02913647
 .float -0.43655157, -0.16239798, -0.14514977, -0.20111826, -0.13600309, -0.04078475, -0.17939718, 0.07558508, -0.26053730, -0.14131816, -0.30844057, 0.02568227
 .float -0.00844552, -0.06590498, -0.26281416, 0.09432300, -0.44439882, -0.13034452, -0.48860788, 0.09290664, -0.08320216, -0.00675505, -0.18378212, 0.06207562
 .float -0.61080456, -0.19460478, -0.58903360, 0.07751808, -0.23673299, 0.09743960, -0.07547921, -0.01418684, -0.49869215, -0.08133960, -0.27603380, 0.21541926
 .float -0.22424363, 0.07786266, 0.11280107, 0.01050298, -0.36506608, 0.01993163, -0.18336815, 0.13546097, -0.31153060, 0.11994804, 0.10882246, 0.04688322
 .float -0.18658581, 0.06420187, -0.20804636, 0.20211565, -0.24175419, -0.09762686, -0.15388623, -0.19335896, -0.03968230, -0.03151905, -0.16200082, 0.00269561
 .float -0.14453998, -0.03803578, -0.16450160, -0.25817665, -0.05563379, -0.00280707, -0.01619457, -0.07741199, -0.08509910, -0.13526110, -0.07716928, -0.09773246

# Row 4 of weights matrix
 .float 0.08580356, 0.17576697, -0.09001210, -0.12350693, 0.04410811, 0.13171181, 0.10630167, 0.01934925, 0.11805926, 0.07459983, 0.06251964, -0.07139686
 .float 0.01217602, 0.11640576, 0.06508541, 0.00374809, 0.03359855, 0.01773925, 0.07026041, -0.02800589, 0.07121064, -0.01886924, 0.03575708, -0.04836067
 .float -0.00435723, -0.04718704, -0.13842538, 0.04413619, 0.02060517, 0.06039687, 0.06707819, -0.07933791, -0.01284639, -0.02769477, -0.17652641, 0.08397402
 .float -0.00866052, 0.10024706, -0.00487598, 0.08043688, 0.03554264, 0.06883861, 0.04557753, -0.02609182, -0.05544370, 0.16265674, -0.04414768, -0.16246378
 .float 0.04522060, -0.03278348, -0.15767181, 0.00787778, -0.04506816, 0.31521210, 0.02669169, -0.24385290, -0.03667036, -0.12696443, -0.11955510, 0.02846524
 .float -0.05287044, 0.17292906, 0.04259757, -0.16592005, 0.04007876, -0.12274354, -0.24306004, -0.03174081, -0.01680606, -0.02647440, 0.09168561, -0.26410067
 .float 0.01676808, -0.06331866, -0.18288390, -0.08505814, -0.07633424, -0.33751592, 0.15669312, -0.60244745, -0.07880155, -0.18229991, -0.08766109, -0.07651415
 .float -0.12053873, -0.22080097, 0.11225364, -0.40907282, -0.08879700, -0.24031001, -0.00833213, -0.07642324, -0.09367992, -0.19277981, -0.13485087, -0.52482970
 .float 0.13967143, 0.14968742, 0.02765133, -0.01318179, 0.06725496, 0.11907794, 0.12152329, 0.14276539, -0.02317130, 0.02622085, -0.06839006, 0.05964597
 .float -0.01115320, 0.06614236, 0.10223282, 0.03186397, -0.03502707, -0.03038497, -0.01964366, 0.02403180, 0.02338514, 0.01626786, 0.04048692, -0.05633897
 .float 0.04255286, 0.00307246, -0.07488788, 0.06584276, -0.06646734, 0.13264346, 0.08045179, 0.08701154, -0.03043761, 0.02666196, -0.14635283, 0.10775899
 .float -0.01457144, 0.18610907, 0.07800575, 0.09778009, 0.10559142, 0.00281076, -0.31443366, 0.05159203, -0.12434736, 0.21773688, 0.03537049, 0.20504814
 .float -0.07044090, 0.02284243, -0.21863627, -0.00162409, -0.13820179, 0.11234402, 0.06452353, 0.11753081, -0.08792330, -0.09464047, -0.09338789, -0.05877774
 .float -0.08304441, -0.07042349, 0.09967083, -0.06252876, -0.16535315, -0.00131050, -0.13555479, -0.03042495, -0.12074304, 0.02254249, 0.02910680, -0.04352709
 .float -0.05272625, -0.14918074, -0.08038157, -0.15063892, -0.02306075, -0.05261178, 0.04921839, -0.11022331, -0.14217460, -0.17364827, -0.17433588, -0.11102113
 .float -0.05411353, -0.09155238, -0.00205898, -0.42722200, -0.04877431, -0.21177015, -0.04172283, -0.10668002, 0.01099395, -0.33188266, 0.08103452, -0.36153263
 .float 0.01778744, 0.06110150, 0.09662699, -0.10471059, 0.07163019, 0.11390538, 0.07084876, 0.09955481, 0.06608971, -0.00771362, -0.08232285, -0.01394446
 .float 0.14855956, 0.06058357, 0.09935912, 0.05196193, 0.11703956, -0.00487229, -0.21998043, 0.04237280, 0.01450345, 0.15215285, 0.07163670, 0.08067606
 .float 0.00084044, -0.04206662, -0.35419747, 0.08525884, 0.07162958, 0.12063965, 0.07249949, 0.09141903, -0.04455135, -0.11219548, -0.40978120, 0.10851631
 .float 0.00947767, 0.11854474, 0.04216209, 0.07716905, -0.01797208, -0.10978878, -0.36302325, 0.11035669, -0.01520519, 0.07693435, 0.09870656, 0.05209139
 .float -0.17746070, 0.08087196, -0.19382395, 0.10777719, -0.06739841, 0.04857978, 0.06499927, 0.02369549, -0.23996444, -0.09183241, -0.07996147, -0.11929599
 .float -0.12589250, -0.13691756, 0.10021155, 0.02451381, -0.16138184, -0.01297884, 0.00099294, -0.19157436, -0.01728317, -0.15877557, 0.02691591, -0.04701399
 .float -0.43073472, -0.08889800, 0.03393040, -0.11402674, 0.08489697, -0.17244376, 0.02478556, -0.06870090, -0.40124440, 0.05099708, 0.05734567, -0.15593103
 .float -0.02780516, -0.21710579, 0.11535981, -0.33428603, -0.21166693, -0.25696390, -0.15838815, -0.11186008, -0.05478972, -0.07172834, 0.11939192, -0.26562270
 .float 0.01313227, -0.00806998, 0.26106690, -0.10642941, 0.04582975, 0.08324679, -0.03225437, 0.01547097, 0.05215701, -0.07834441, -0.04976223, 0.10355740
 .float 0.02829146, 0.14740886, 0.06311575, -0.00094879, -0.00047299, -0.08927988, -0.24250065, 0.23756456, 0.08073727, 0.20106650, -0.03921504, -0.05356824
 .float 0.03719065, -0.11059130, -0.27187672, 0.17642576, 0.00038535, 0.13100982, -0.00484587, 0.00936167, 0.03129740, -0.12850448, -0.31697494, 0.23092489
 .float 0.03667515, 0.05454437, -0.09648683, 0.08672879, 0.04654554, -0.07832704, -0.10348415, 0.12866570, 0.11317170, 0.09072743, 0.01594435, 0.17466491
 .float -0.08008794, -0.13543743, -0.08416381, -0.06281572, 0.15509166, -0.05774896, 0.02046257, 0.13656010, -0.07684552, -0.03286506, 0.04819325, -0.10392103
 .float 0.14984186, -0.05701252, 0.10078783, -0.02891234, -0.07429685, 0.00689102, 0.20216233, -0.25177985, 0.19451615, -0.17551342, 0.06425963, 0.06960861
 .float -0.25251457, -0.08040785, 0.16624764, -0.32310468, 0.13837913, -0.18406506, 0.00382949, -0.18565430, -0.40133720, 0.02595423, 0.17373414, -0.13601935
 .float 0.21871844, -0.09221098, 0.13536520, -0.15152372, -0.35954028, -0.18930466, -0.20186399, -0.20523700, -0.10459412, -0.24821390, 0.16741505, -0.07502197
 .float -0.14251806, -0.08641886, 0.23252256, -0.20732431, 0.12253082, 0.03510527, -0.05982173, -0.04974499, -0.11030757, -0.19175075, 0.06292686, 0.13087313
 .float -0.05869122, 0.04573770, -0.09039909, -0.09381010, 0.02534758, -0.25360617, -0.05262189, 0.28695804, 0.03139497, -0.05529292, -0.28211793, -0.01570714
 .float 0.13531363, -0.19775209, -0.17805536, 0.19974630, 0.08688950, -0.02953830, -0.31116164, 0.00676363, 0.35921094, -0.15035509, -0.23481907, 0.10570141
 .float 0.11831952, -0.13967401, -0.24536386, -0.02405068, 0.40616298, -0.04740951, -0.29997054, 0.06531778, 0.08044714, 0.00981067, -0.02908066, 0.12166524
 .float 0.27195280, -0.05666803, -0.06464499, -0.10968667, 0.27692480, -0.03200203, 0.07364700, 0.04916148, 0.22400373, -0.13045815, 0.13635483, -0.08142576
 .float 0.23424439, 0.04670116, 0.09045345, 0.01307818, 0.07609156, -0.17914984, 0.09377921, -0.01622253, 0.17044927, -0.04842109, 0.00185225, 0.09987033
 .float -0.02960530, -0.18877286, 0.09512570, -0.21651305, 0.12748906, -0.06063167, -0.01482672, 0.10117619, -0.19840316, -0.28749678, 0.11437547, -0.30802554
 .float 0.02427372, 0.13268475, 0.09364115, 0.15264282, -0.22792616, -0.27442786, -0.43242568, -0.42202437, -0.29521054, -0.02746205, 0.00744126, 0.05175109
 .float -0.25806220, -0.26193804, 0.23409446, -0.14656375, 0.00302660, -0.10448405, -0.14458800, -0.20088433, -0.14625870, -0.33019406, 0.04715085, 0.14150278
 .float -0.03685105, -0.07560137, -0.19383347, -0.22495988, 0.16850786, -0.30154890, 0.14006370, 0.21987950, 0.02299303, -0.04612339, -0.47588897, -0.14907245
 .float 0.22493427, -0.20993483, 0.15052426, 0.02748327, 0.08318086, -0.11642716, -0.39350095, -0.12033915, 0.47061960, -0.09906188, -0.06527566, -0.00760264
 .float -0.03407490, -0.16994374, -0.15923716, 0.01018170, 0.39923635, 0.09419891, -0.17910689, 0.14021514, 0.11534699, -0.03638587, -0.07113791, 0.11598844
 .float 0.27284852, -0.02111228, -0.06001968, 0.14016674, 0.15778970, 0.13820237, 0.11431988, 0.03489107, 0.07277408, -0.29139390, 0.01705591, 0.11590079
 .float 0.17725588, 0.17138295, 0.00591915, 0.06484708, -0.02298653, -0.18739358, -0.03092419, 0.03360169, 0.03996821, 0.18407167, -0.00041812, 0.16884350
 .float -0.09900655, -0.28348047, -0.15975048, 0.09430591, -0.08385262, 0.02878560, -0.09040002, 0.05493727, -0.10900359, -0.42613460, -0.32897040, 0.02825104
 .float -0.17209166, 0.05732732, -0.12870015, 0.10518052, -0.13443486, -0.09466092, -0.06647764, -0.08925394, -0.10434033, -0.06181901, -0.20916092, -0.43546590
 .float -0.16947033, -0.04478124, 0.09578451, 0.05751088, -0.15761247, -0.01741293, 0.13191606, -0.10554617, -0.17880988, -0.23917979, -0.14101373, -0.04010171
 .float -0.26563450, -0.17530268, -0.13414656, -0.18639278, -0.07647687, -0.19651948, -0.27619314, 0.04044125, -0.27143586, -0.06651770, -0.11868760, -0.12060485
 .float -0.00647951, -0.13459879, -0.07313100, 0.16907497, -0.17996928, -0.14961351, -0.23554432, -0.17256984, 0.25223807, -0.06842551, -0.29536253, 0.03582155
 .float -0.17532194, -0.05995384, -0.10537873, 0.03866607, 0.17568716, 0.03823096, -0.17240454, 0.17952980, 0.00372502, -0.02542464, -0.07703991, -0.08784211
 .float 0.01274650, -0.15002292, -0.15682001, 0.04194038, 0.03550258, 0.17815192, 0.04169128, -0.06740565, -0.12214763, -0.23954321, -0.16117088, 0.16082901
 .float -0.02654588, 0.08120500, -0.00470997, 0.08705279, -0.25891060, -0.14698915, -0.08988091, 0.04099224, -0.08658299, 0.03052448, -0.10983782, -0.00184502
 .float -0.24396129, -0.14890134, -0.20220585, 0.14138393, -0.06551185, -0.07326944, -0.08364150, -0.15282136, -0.04048810, -0.14782439, -0.16418667, -0.07097396
 .float -0.10269921, -0.28669426, -0.03920593, -0.16635256, -0.07603382, -0.15322948, -0.02339009, -0.02230185, -0.01787966, -0.46879044, -0.07605043, -0.70931700
 .float -0.07775386, -0.06860663, -0.11312098, -0.02340411, -0.21350177, 0.03814122, 0.13653547, 0.06200448, -0.20422156, -0.02750935, -0.13981578, 0.10363322
 .float -0.10001309, -0.04214523, 0.04305410, -0.14787248, -0.13363843, -0.17583399, -0.30971634, 0.19090710, -0.18189733, 0.04460995, 0.02486374, -0.13706636
 .float -0.21616330, -0.07488295, -0.27475560, 0.20602886, -0.25240450, 0.07081121, -0.07695859, -0.20761445, -0.09849830, -0.03472017, -0.29520968, 0.26209697
 .float -0.26348330, 0.12808591, -0.12458823, -0.13634072, -0.13784121, 0.01134630, -0.16154484, 0.13701858, -0.05859281, 0.14733872, -0.03573118, -0.03036084
 .float -0.13707210, -0.09444197, -0.22850695, 0.05022670, -0.02265594, 0.23587908, 0.02620692, 0.09966317, -0.24346720, -0.00934315, -0.10900491, 0.06896909
 .float -0.10778212, 0.04066316, -0.09271491, -0.00595154, -0.14745086, -0.00166527, 0.06074590, 0.01065098, -0.01036670, -0.00283029, -0.04831703, -0.18528862
 .float -0.17479067, 0.01729185, -0.03763359, -0.15385172, 0.02351605, -0.17779681, 0.03328685, -0.12831526, -0.25955603, -0.06834759, 0.01493126, -0.14750588
 .float 0.02361308, -0.19309935, 0.00774383, -0.37056878, -0.19417316, 0.00254680, 0.04237615, -0.23211965, -0.09025009, -0.29239014, 0.04061986, -0.09696765
 .float -0.04233932, 0.00270299, -0.13027139, -0.13961740, -0.11460599, 0.17282243, 0.00826827, 0.00752984, -0.21494240, 0.00299360, -0.09924761, 0.13502637
 .float -0.12732053, 0.14779152, 0.09069164, -0.08064807, -0.15316461, 0.03142052, -0.25575250, 0.22269544, -0.09907415, 0.05045965, 0.02496328, -0.05731256
 .float -0.20400949, -0.09147976, -0.22200142, 0.22807994, -0.15215354, -0.08750916, 0.07770404, -0.13300209, -0.04711423, -0.01617516, -0.21882077, 0.25658324
 .float -0.14834921, 0.09123893, -0.00247407, -0.03485996, -0.02318813, 0.01278363, -0.37366217, 0.17201066, -0.06442765, 0.17404975, 0.04286292, 0.04729675
 .float -0.05500140, -0.04891375, -0.22449876, 0.10262532, 0.07255792, 0.14622359, -0.00915655, 0.02542978, 0.07148626, -0.01637072, 0.03280927, 0.14937581
 .float 0.10427832, -0.04333562, -0.09944701, 0.00828674, -0.07479404, 0.08453959, -0.00716415, -0.03155800, 0.02586983, -0.10121393, 0.04680599, -0.00101249
 .float 0.02939898, 0.00731595, -0.06011761, -0.26996540, 0.02982015, -0.16644083, 0.02172810, -0.10408621, -0.06311247, 0.11899126, 0.09062188, -0.32563344
 .float 0.06203095, -0.21663287, -0.03958791, -0.15427628, -0.09018987, 0.00741573, 0.03442718, -0.18631937, -0.05955960, -0.11974512, 0.06922308, 0.01599455
 .float -0.02437548, 0.02154192, -0.11683631, -0.10001089, -0.04448904, 0.02885581, 0.02642543, -0.06765013, -0.09765159, -0.01037257, -0.28318400, 0.15690327
 .float -0.11677060, 0.03466023, 0.08861554, 0.08554800, -0.15079774, 0.06188709, -0.29622883, 0.14193273, -0.12185574, 0.09446721, 0.07362354, -0.00220156
 .float -0.14048128, 0.03462892, -0.36603907, 0.16720827, -0.10099059, 0.05171203, 0.09889799, 0.03352888, 0.03889699, -0.08287496, -0.36071172, 0.09563962
 .float -0.00790216, -0.03017584, 0.05315289, -0.04017960, -0.00826781, -0.05974610, -0.38624090, 0.06665415, -0.00178359, -0.01126982, 0.04204759, 0.03444863
 .float 0.02632416, -0.07609028, -0.09821110, -0.00623645, 0.05253709, -0.06012360, -0.07111997, 0.01186244, 0.01378975, 0.01848345, 0.03669977, 0.11725229
 .float 0.14814658, -0.00773197, -0.00271482, 0.10761836, 0.02469836, 0.07320119, 0.05858641, -0.02835239, 0.10845516, -0.19170410, 0.06886719, -0.05487398
 .float 0.04703107, 0.05211916, 0.12554726, -0.30682343, 0.19572797, -0.14214554, -0.01173926, -0.08127119, -0.01779846, 0.04972966, 0.02832631, -0.35459080
 .float 0.18887997, -0.04157559, 0.09537822, -0.03321450, -0.09547669, -0.19955716, 0.11851676, -0.23394500, -0.06345473, -0.04271801, 0.04512152, -0.13426118
 .float -0.10258935, 0.02769909, -0.10565894, 0.08753107, -0.12675235, 0.05996020, 0.26593790, -0.08695251, -0.07427200, 0.05480769, -0.12207700, 0.28852550
 .float -0.01865441, -0.06436697, 0.26022363, -0.00764632, -0.04133100, 0.02260574, -0.20030017, 0.42122096, -0.02921031, 0.06656878, 0.07338154, -0.04341810
 .float -0.04849038, 0.05850814, -0.24391925, 0.26039508, -0.02186409, -0.00779082, 0.10508549, 0.04779502, 0.13055557, -0.06306275, -0.37021554, 0.21702474
 .float 0.09096183, 0.02707505, 0.08132733, 0.05509633, 0.13812900, -0.10714869, -0.08645106, 0.17154986, 0.15197325, -0.08882248, -0.07172745, 0.12004274
 .float 0.16042568, -0.09847808, -0.04313571, 0.09834453, 0.17147772, -0.01466308, -0.03883850, -0.05209787, 0.23146267, -0.11794170, -0.05060704, 0.02065546
 .float 0.12902181, -0.09211343, -0.03997868, -0.01730365, 0.16380310, -0.08778767, 0.11342348, -0.02114918, 0.17952158, -0.13487968, -0.01231023, 0.04482466
 .float 0.17012943, -0.12834899, -0.07783891, -0.41200218, 0.05290028, -0.12569340, -0.00836767, -0.09718179, -0.14061170, -0.10582589, 0.12484272, -0.34031308
 .float 0.00333227, -0.07214636, -0.07211047, -0.05286979, -0.05710754, -0.17778459, -0.42824784, -0.22812979, -0.14527550, 0.07320135, 0.11513337, 0.01980350
 .float 0.10923281, -0.02136940, 0.05808739, -0.23893051, 0.10123675, 0.11381424, 0.06603215, 0.06030766, 0.06720069, -0.04583678, -0.05063848, -0.00013234
 .float -0.06265615, 0.27306208, 0.11082096, 0.03945558, 0.18066682, 0.00907599, -0.02045553, 0.13251662, -0.07023456, 0.28079626, 0.24560007, -0.00435659
 .float 0.24913785, 0.05064937, -0.13516048, 0.34649660, 0.08553763, 0.06567721, 0.05246581, 0.06723864, 0.43213960, 0.07708389, -0.05442258, 0.52755950
 .float 0.05268814, -0.03022309, -0.00699399, 0.09020539, 0.28175288, -0.13448371, -0.02175318, 0.27287160, 0.16362292, -0.04919003, -0.06454275, -0.02868781
 .float 0.42194018, -0.12458556, -0.10142861, -0.03967583, 0.08206280, -0.12529461, -0.09214246, -0.03004791, 0.25856915, -0.13698767, -0.04621265, -0.20207788
 .float 0.11798944, 0.01191855, -0.14874938, -0.01692340, 0.17791220, -0.15093890, 0.03981673, -0.17221512, 0.06324461, -0.06441621, -0.11307062, 0.05918476
 .float 0.07265041, -0.06806811, -0.10081276, -0.27126053, 0.08886339, 0.03190823, -0.04179552, 0.00463596, -0.00314372, -0.07079560, -0.18156552, -0.28652585
 .float -0.12584148, -0.09744561, -0.07112291, 0.08271480, 0.00937138, -0.17629415, -0.07377739, -0.17760487, -0.06608123, 0.06046421, -0.11173963, -0.06656358

# Row 5 of weights matrix
 .float 0.03848818, -0.16378264, -0.03748261, 0.01040424, 0.05632918, -0.11701705, 0.04530020, -0.20988853, -0.17624556, -0.18808053, 0.00410088, -0.07884500
 .float -0.03952086, -0.21191630, -0.05326759, -0.03359540, -0.17852750, -0.20711780, -0.32202113, -0.01068095, -0.11189004, -0.38277440, -0.11052987, -0.04878901
 .float -0.29980353, -0.27986103, -0.14552741, -0.08676737, -0.18360000, -0.44913393, -0.33853284, -0.12961002, -0.41423607, -0.18221445, -0.08894379, -0.15534700
 .float -0.18952797, -0.31511060, -0.42115816, -0.12213109, -0.25216842, -0.10977820, -0.04798223, -0.05001450, -0.22161016, -0.33436460, -0.25715550, 0.16678676
 .float -0.08104485, -0.13265412, -0.06647671, -0.01436273, -0.14578357, -0.54352940, -0.20852411, -0.03280803, -0.08592786, -0.08922759, -0.07716291, -0.04715374
 .float -0.08069019, -0.35919756, -0.12833552, 0.03392744, -0.03909082, -0.02916920, -0.06900588, -0.11372168, -0.08574332, -0.22671336, -0.16274990, 0.06404707
 .float 0.00291206, 0.03500145, -0.06792925, -0.02570292, -0.08452422, -0.29911986, -0.09056660, 0.06537651, 0.02633021, 0.13155715, -0.01710185, 0.03716716
 .float -0.01742243, -0.30712870, -0.02614150, -0.01325928, 0.09322122, -0.07449582, 0.01670993, -0.04390588, -0.03811915, -0.09509402, 0.07582875, -0.36692300
 .float -0.17434032, -0.03813061, -0.04486998, -0.03741442, -0.08802919, -0.01527330, -0.03186044, 0.03708304, -0.07894023, 0.00574957, -0.04679554, 0.06011271
 .float -0.10532345, -0.05551674, -0.02756453, -0.06604524, 0.02708756, -0.16502534, 0.16177699, -0.04664654, 0.01570688, 0.10926154, -0.05052403, -0.05093975
 .float -0.19754161, -0.20203483, 0.41793674, -0.01248981, 0.02278775, -0.06265790, -0.22844283, 0.09273029, -0.22838591, -0.08138435, 0.33594307, -0.10863496
 .float -0.10593700, 0.02974256, -0.23511584, -0.01905251, -0.03426058, 0.01930811, 0.42357606, -0.22030906, -0.09851766, 0.00191406, -0.28951988, 0.12892878
 .float -0.10521834, 0.06171574, 0.30333110, -0.19181490, -0.13521954, -0.23633534, -0.19642542, 0.06777660, -0.09804604, -0.03669717, 0.25316107, -0.04148865
 .float -0.07972331, -0.32515770, -0.21081235, 0.06003136, -0.13629964, 0.12910314, 0.21802036, 0.00053991, 0.00434958, -0.33576900, -0.14734131, 0.03412674
 .float 0.02060763, 0.05944458, 0.14253445, -0.04248496, 0.00498344, -0.17325282, -0.16089945, 0.16511810, -0.06536323, 0.10801291, 0.00446570, 0.02391572
 .float -0.04379084, -0.00608381, 0.05073067, 0.22228274, 0.10089914, 0.00390961, 0.11461680, 0.08453694, 0.13439755, -0.01686422, 0.05699483, 0.20999664
 .float -0.03864417, -0.01680958, -0.03722121, 0.09005035, -0.12299851, 0.04018842, -0.00634984, 0.12919345, 0.01411423, -0.08941935, -0.14640386, 0.03424197
 .float -0.05627472, 0.17530742, -0.04528434, 0.02949518, -0.00068581, -0.02266941, 0.28743586, -0.03784769, 0.06572013, 0.15416259, -0.01545377, 0.14491434
 .float -0.10980651, 0.08655186, 0.51076890, 0.00168980, -0.12388698, 0.02761716, -0.05249650, 0.19252367, -0.09040876, 0.04117943, 0.32078058, -0.06835566
 .float -0.13588375, -0.26559670, -0.09421747, -0.03639439, -0.20418246, 0.12523901, 0.34186825, -0.26943790, -0.19090809, -0.60512245, -0.17324294, -0.18903537
 .float -0.14581512, 0.24593760, 0.39828590, -0.22575195, -0.18262546, -0.51300700, -0.11166908, -0.32212093, -0.01218376, 0.33518872, 0.31624022, -0.19093953
 .float -0.01905931, -0.61215100, -0.12528060, -0.37339450, 0.02126961, 0.20764330, 0.29625705, -0.07608210, 0.08327353, -0.62850770, 0.02082002, -0.34769985
 .float 0.05960412, 0.09385575, 0.21541649, -0.03376776, 0.04590727, -0.25880572, -0.00082364, -0.02153045, 0.02318822, 0.15528847, 0.02506352, -0.06546162
 .float 0.11261377, 0.02983115, 0.01308287, 0.18043885, 0.06736508, 0.18281077, 0.17210716, 0.01442015, 0.05525744, 0.12537736, 0.02764410, 0.13146755
 .float 0.10853208, -0.01107732, 0.12672956, -0.02951773, -0.15757462, 0.03305728, -0.06558973, -0.00243675, -0.00404380, 0.07418763, 0.00387882, -0.00996224
 .float -0.16946498, 0.02751387, -0.00936785, 0.07855996, 0.00204301, 0.02881397, 0.15056309, -0.00529716, -0.11027355, 0.06345533, 0.02061839, 0.15807365
 .float -0.01633859, 0.09677322, 0.18177810, -0.02622965, -0.07270487, -0.24001017, 0.04633454, -0.08463951, -0.01017925, 0.20816807, 0.19937134, -0.09203806
 .float -0.09774994, -0.25396480, -0.02867259, -0.16955423, -0.12061090, 0.23027007, 0.24779344, -0.52096500, -0.03999909, -0.34181374, 0.07714918, -0.32518423
 .float 0.19179381, 0.16307190, 0.27530210, -0.54950180, -0.10815068, -0.20961611, 0.03687055, -0.14571850, 0.14560968, 0.15012637, 0.32526930, -0.38675427
 .float -0.01261988, -0.28994700, -0.02888409, -0.23038538, 0.23401177, 0.04407997, 0.31345376, -0.23535335, 0.10162140, -0.40384370, 0.01660688, -0.25360718
 .float 0.12398334, 0.09850898, 0.27489254, -0.21233470, 0.08239275, -0.19184950, 0.04452157, -0.10303380, 0.09002248, 0.03619225, 0.13967432, -0.02850064
 .float 0.08415715, -0.11079814, 0.04451327, 0.05952445, 0.09090272, 0.04835835, 0.03751387, -0.02485721, 0.07299424, 0.16832602, -0.10559582, 0.16526700
 .float 0.02213819, 0.00816010, 0.09850673, 0.08579417, -0.18600442, -0.01075497, 0.02516621, -0.03369439, 0.03428472, 0.06572946, -0.15501809, 0.01730466
 .float -0.13094614, -0.12169079, 0.07879627, 0.00355539, 0.12101083, 0.15548468, -0.05268060, 0.05025003, -0.08910985, -0.12018783, 0.16151418, -0.01225106
 .float 0.14888053, 0.17362872, 0.08288328, 0.00385759, 0.06588066, -0.08776589, 0.15522346, 0.00081079, 0.10706187, 0.17960072, 0.15600543, -0.24538991
 .float 0.12135834, -0.14388030, 0.12887914, 0.07380006, 0.07732721, 0.10608070, 0.17589685, -0.30226943, 0.08802650, -0.29586712, 0.11777909, -0.22529396
 .float 0.07104833, 0.04549628, 0.03714238, -0.25983924, 0.00268134, -0.18275740, 0.00620979, 0.01990795, -0.01511700, 0.23559277, 0.20455833, -0.28647870
 .float -0.02724923, -0.31725913, 0.08606119, -0.11098932, 0.02826923, 0.07878748, 0.09992158, -0.28068325, 0.08110171, -0.18820378, 0.13814244, -0.10384389
 .float 0.04543665, 0.10301672, 0.11472724, -0.14459840, 0.11639961, -0.14950109, 0.01241931, -0.14628689, 0.05526499, -0.12317794, 0.05417328, -0.03127997
 .float 0.01011409, -0.21771221, -0.01360926, -0.18190701, -0.12258505, -0.17583938, -0.18250774, -0.01108868, -0.03382439, -0.16305113, -0.14502110, 0.05332885
 .float 0.19541585, 0.11042232, -0.20689279, 0.04795093, -0.19989191, -0.07556087, 0.07817762, -0.03352880, 0.08219197, 0.09551702, -0.15824440, 0.08071968
 .float 0.03160281, 0.03472188, 0.15378903, -0.02663141, 0.16769195, 0.17444807, 0.02317771, -0.10119113, 0.14552188, -0.00407532, 0.22167279, 0.08673356
 .float 0.02393369, 0.15515920, 0.05575559, -0.25921053, 0.19051460, 0.06144863, 0.12764794, 0.06542763, 0.05693240, 0.09140327, 0.03159260, -0.25014560
 .float 0.14169258, 0.11208925, 0.23892677, 0.11481708, -0.12564844, -0.02225357, 0.15845926, -0.07757035, 0.02111232, 0.19387133, 0.20816626, 0.00320771
 .float -0.03325732, -0.07959997, -0.18869954, 0.00726513, -0.06647358, 0.00356614, 0.01651301, 0.18906000, -0.02204610, 0.21103005, -0.00333487, -0.30251074
 .float 0.01214295, -0.18565623, 0.09005749, -0.13713443, 0.03551862, 0.17340300, 0.11552983, -0.26749860, 0.05382466, 0.00058552, 0.07405218, -0.13960357
 .float 0.12821963, 0.24423924, 0.07847950, -0.01566791, 0.16252601, 0.10850002, 0.10031238, 0.04528291, -0.00172907, -0.27087870, -0.04565794, 0.07433373
 .float -0.03803816, 0.09507223, 0.12275617, 0.06863274, 0.05024152, -0.29344380, -0.32603434, 0.09274529, -0.13119599, 0.08455438, -0.06068276, 0.16264044
 .float 0.04305517, 0.09782741, -0.39551310, -0.11846028, -0.06189244, 0.07902467, 0.05455618, 0.04831942, -0.02828958, 0.09084831, -0.22461234, -0.06373877
 .float 0.00748004, 0.27110577, 0.26766214, 0.07826538, 0.08751143, 0.09071006, -0.15529630, -0.20046570, 0.09925362, 0.15166055, 0.16897742, 0.10000884
 .float 0.00225901, 0.02730468, 0.11232200, -0.16767189, 0.16533865, 0.11209730, 0.11539145, 0.06085326, -0.11432336, 0.00461278, -0.12492494, -0.06619190
 .float 0.07538915, 0.15138220, 0.20543458, 0.04486284, -0.33464566, -0.01569883, -0.13411655, 0.06647176, -0.07285527, 0.35046807, 0.15679927, 0.03329845
 .float -0.12817034, 0.09460524, -0.17937082, 0.02266032, 0.02097481, 0.30778840, 0.06978050, 0.11888546, 0.04234610, 0.13611247, 0.03073109, -0.07442504
 .float -0.00738219, 0.05676812, 0.15773080, 0.10527293, 0.16283946, 0.18870446, 0.04961548, -0.21277644, 0.10962331, 0.11706408, 0.12675615, 0.25782603
 .float 0.14728053, 0.12265327, 0.16222338, 0.04991827, 0.09500011, 0.02317306, 0.07029858, 0.20218791, -0.02110786, -0.25538775, 0.01063787, 0.12341778
 .float -0.11783786, 0.10538146, 0.09491637, 0.05589106, -0.01816197, -0.04921016, -0.19354975, -0.05607216, -0.03084165, -0.07975528, -0.07826812, -0.03962434
 .float -0.02263637, 0.14599177, -0.13438974, -0.06347358, 0.24568920, -0.06128091, 0.02523054, 0.13912527, -0.04101709, 0.08081909, -0.02818889, 0.00742276
 .float 0.09442569, 0.10578098, 0.09833673, 0.06512532, 0.05919999, 0.10982323, -0.04584327, -0.06445921, 0.04574647, 0.00852176, 0.04756317, 0.13224067
 .float -0.08788507, 0.01655739, -0.02822442, -0.15501632, -0.07378817, 0.05610009, 0.07122578, 0.06740091, -0.13515052, -0.00531216, -0.09480899, 0.06963588
 .float -0.16543366, 0.10238268, 0.08222837, 0.00868284, -0.02588802, -0.11021585, -0.11567608, 0.17771070, -0.08242676, 0.16430107, 0.03709632, 0.01325154
 .float 0.06403558, 0.09322456, -0.06498786, 0.11187543, 0.03854754, 0.16247343, 0.03323958, 0.07067064, -0.01362252, 0.10963475, -0.04069845, -0.03212617
 .float 0.00315839, 0.12672561, -0.01935615, 0.13622968, 0.18041064, 0.03132423, -0.10003430, -0.19986522, 0.07553056, 0.17338578, 0.05650827, 0.04324964
 .float 0.14546625, -0.06508061, -0.05253846, 0.02462414, -0.00388431, 0.10966271, 0.06655209, 0.10570386, 0.05023887, -0.05133119, -0.03835448, 0.13188273
 .float -0.11229666, 0.13774918, 0.12404277, -0.11794432, 0.01797157, -0.00265532, -0.06885299, -0.04923477, -0.14945070, -0.11953448, -0.11694572, -0.05996416
 .float -0.02031603, 0.03295577, 0.13076176, 0.08884155, 0.11043949, -0.07260935, 0.00084207, 0.10527299, 0.13560754, -0.00917736, -0.01679031, 0.06764603
 .float 0.08820301, -0.05203788, -0.09468482, -0.02919798, 0.08847740, -0.06179356, 0.02957786, -0.00101568, 0.04840753, -0.09204395, -0.08925047, -0.06348639
 .float -0.00196271, -0.07771647, -0.03354554, -0.05040943, -0.10971876, -0.08440991, 0.02009498, -0.16355942, -0.05025902, -0.14117537, -0.00479733, 0.00603045
 .float -0.06450596, -0.03465039, -0.05428569, -0.14907438, -0.04213096, 0.03157426, 0.06873205, -0.03211821, -0.12184885, 0.06284565, -0.01069481, -0.01983715
 .float -0.01051733, 0.09803364, 0.01695531, -0.01811287, -0.10959979, 0.00040855, -0.05296411, -0.03769014, -0.05715719, -0.01141629, -0.03450720, -0.11495877
 .float -0.04726376, -0.05138566, -0.12326315, -0.08053545, -0.02502714, -0.11644011, -0.13843013, -0.16006601, -0.08444367, 0.05523792, -0.06065552, -0.00533610
 .float -0.08102354, -0.05573069, -0.13802485, -0.09498958, -0.11958308, 0.03139506, 0.06556065, -0.00624154, -0.02860950, 0.03486878, -0.07565776, -0.12625869
 .float -0.01003115, -0.08395544, 0.12208873, 0.01766217, -0.03886676, -0.06296872, -0.12714283, -0.14601402, -0.24166249, -0.04706328, -0.08784008, -0.03479793
 .float -0.00068858, -0.08355415, -0.02639211, -0.08918085, -0.10239848, 0.07496567, -0.07911639, -0.06027038, -0.08691341, -0.15749232, -0.10718843, 0.01010908
 .float 0.10058846, -0.13325003, -0.14853019, -0.04193154, -0.00714219, -0.08754194, 0.19581841, 0.12207286, 0.07403108, -0.19837129, -0.17219030, 0.03306313
 .float 0.03101164, -0.11932898, 0.09816042, -0.02731198, 0.05690003, -0.17635888, -0.17904924, -0.08259703, 0.01777843, -0.06459749, 0.14432776, -0.28610528
 .float 0.02855746, -0.23682074, -0.13148469, -0.13837704, -0.00956553, -0.03226905, 0.12092143, -0.50793330, -0.03029427, -0.25990453, -0.10467383, -0.14184496
 .float -0.01055104, 0.05447906, 0.10235130, -0.66535950, -0.10685249, -0.10161297, -0.10963228, 0.06125706, -0.12206962, 0.01033357, 0.05110919, -0.51922756
 .float -0.09908232, -0.01922515, 0.01459301, -0.03361023, -0.14152730, 0.11055431, -0.06130831, -0.25313878, -0.24408970, 0.04237461, 0.03368272, 0.09553316
 .float -0.20006256, 0.14552169, -0.07301859, -0.12226817, -0.04093378, 0.12022175, 0.07969198, -0.06574409, -0.00675788, -0.04440084, -0.17234099, -0.06089930
 .float -0.03448345, -0.05414370, 0.04157353, -0.05385384, 0.02341933, -0.15314502, -0.07573134, 0.01624137, -0.08411211, -0.20243144, -0.14263143, -0.15410910
 .float -0.30545923, -0.05724214, -0.15461285, -0.00399423, -0.07417908, -0.35109785, -0.13984346, 0.04622213, -0.06327800, -0.16796826, -0.07255404, -0.00575801
 .float -0.24280845, -0.20008677, -0.23261097, -0.10762783, 0.02739254, -0.07980795, 0.12886824, -0.10324036, -0.04743659, -0.14022563, -0.26406497, -0.02431648
 .float 0.07896720, -0.07265966, 0.07750320, -0.37772796, -0.05432409, -0.18998908, -0.20166999, -0.04474357, 0.14033934, -0.07552493, 0.21586960, -0.49334720
 .float 0.06534791, -0.28822124, -0.05776541, -0.17296034, 0.06472968, -0.05158477, 0.11094883, -0.49468470, -0.07973321, -0.26552185, -0.05849495, -0.10960592
 .float 0.11130200, 0.09859996, 0.12898631, -0.53608620, -0.04389866, -0.19568859, -0.09553571, -0.05581825, 0.10126791, 0.12566987, 0.07919103, -0.27794242
 .float -0.15954840, -0.04562669, 0.09908368, 0.02174713, 0.08038121, 0.10449275, 0.00546484, -0.21025820, 0.02962770, 0.05156085, -0.01592372, -0.04320177
 .float -0.06882493, 0.19163173, -0.02064465, 0.13264015, 0.02326374, 0.12158819, 0.04872644, 0.10295232, -0.00587615, 0.01792670, 0.01823017, 0.05285252
 .float 0.02048009, 0.10959032, 0.01086871, 0.17293128, 0.02802486, -0.06293606, -0.22261496, 0.01953078, -0.12528095, -0.06190284, -0.08099081, -0.00476899
 .float -0.00939536, -0.06765772, -0.15414728, 0.06578086, -0.29304534, -0.00001558, -0.01318884, -0.14290369, 0.07279257, -0.10495030, -0.22842947, 0.05326161
 .float 0.03668018, -0.06939223, -0.00203870, -0.12018887, -0.08453747, -0.16790031, 0.06098960, -0.21177395, -0.00678701, -0.23688349, -0.10632589, -0.01169294
 .float 0.04853362, 0.01299218, -0.01283853, -0.25801846, 0.05375034, -0.20565495, -0.09861568, 0.04849540, -0.11975319, -0.14254490, 0.06023313, -0.29479522
 .float -0.04083107, -0.06708464, -0.05449045, -0.04762023, 0.08018352, -0.05215277, -0.02853187, -0.27506030, -0.17844225, 0.00091369, -0.11066578, 0.04496936
 .float 0.04788708, -0.05198527, 0.05662299, -0.30807233, -0.15549746, -0.04267245, -0.06819647, 0.05460233, -0.02817135, 0.06749851, 0.19567361, -0.20514882
 .float -0.05721356, 0.02001625, -0.04548445, 0.09315389, -0.01696650, 0.00987135, 0.19272874, -0.19454230, -0.09713256, 0.03872868, -0.06921580, 0.02444386
 .float 0.00852088, 0.14260160, -0.06535881, -0.11334640, -0.12282507, -0.06087470, -0.06537038, 0.16193560, 0.03651743, 0.01946809, 0.11746699, 0.02917935
 .float -0.16201393, 0.12078109, -0.07668914, 0.06154412, 0.06945749, -0.11876897, -0.21502002, 0.07117971, -0.04709402, 0.06214879, -0.06971718, 0.03909228

# Row 6 of weights matrix
 .float -0.05742681, -0.24619319, -0.07735836, 0.10650695, -0.02989973, -0.19417395, -0.19561501, -0.12220415, -0.16738679, -0.21425510, 0.04418443, -0.13942370
 .float -0.04099230, -0.17123917, -0.17746360, -0.28942654, -0.11616298, -0.19347991, -0.18237000, 0.01369607, -0.14539258, -0.08457831, -0.08391183, 0.03373677
 .float -0.12753056, 0.06141513, -0.23400186, -0.06682987, -0.13473518, -0.04492637, -0.06308363, -0.07046502, -0.05758296, -0.13418552, -0.37240610, -0.11598517
 .float -0.03764983, -0.28328785, -0.06630833, -0.10488047, 0.09751215, -0.12431108, -0.47093780, -0.08292840, -0.03267617, -0.19339946, 0.00884838, -0.04842751
 .float 0.10202134, 0.04635841, -0.06851903, 0.00387361, 0.05412799, -0.05167252, -0.01098224, -0.01467519, 0.09740634, 0.19515881, 0.16410837, 0.16496673
 .float 0.06038877, 0.06497936, 0.05865195, 0.07045219, 0.12194616, 0.10698971, 0.30548617, 0.17394670, 0.25259176, 0.30199522, 0.10841519, 0.01055835
 .float 0.04177184, 0.03625623, 0.06346180, 0.10227263, 0.10262914, 0.05854765, 0.05323672, 0.30981922, -0.04303437, -0.01326048, -0.02145390, -0.00875785
 .float -0.00257561, -0.02537749, -0.03023654, 0.14833647, 0.11249752, 0.00756092, 0.00931225, 0.01398799, 0.02249138, -0.01234127, 0.01365293, 0.03990394
 .float -0.15740761, -0.15434763, 0.15464450, -0.02576292, -0.12220520, -0.14082912, -0.16868402, -0.14405039, -0.10946714, -0.11767992, -0.02383213, -0.12794475
 .float -0.18516287, -0.19252494, -0.04551004, -0.05308425, -0.06860356, -0.02809806, -0.07096434, -0.02300168, -0.09706657, -0.18984605, -0.02323124, -0.06983143
 .float 0.03155815, 0.02708424, -0.12299261, -0.01703365, 0.06206482, 0.01889224, 0.03262143, -0.25268364, -0.04468942, 0.02785833, 0.06835694, -0.09533219
 .float -0.02749902, -0.11349566, -0.05037484, -0.14432990, 0.01852222, -0.01397666, -0.20444927, 0.01318835, 0.04678874, -0.05585347, -0.09101971, -0.07937449
 .float 0.09707847, 0.11097109, -0.16029288, -0.01829645, 0.04519272, 0.02719072, -0.07198729, -0.01329326, 0.14796185, 0.23404100, 0.05214181, 0.11275452
 .float 0.12896612, -0.01763106, 0.00496331, -0.04287699, 0.16310562, 0.10318730, 0.25217068, 0.05188876, 0.12564485, -0.00075344, 0.13010406, -0.00907238
 .float 0.09665500, 0.18264621, 0.15196882, 0.08125307, 0.04223051, -0.00488057, -0.03052155, 0.05282609, 0.12591900, 0.02404209, -0.04019205, -0.03451709
 .float 0.09431277, 0.09196006, -0.11404277, 0.24193013, 0.06062309, 0.19284283, 0.13128866, 0.10663210, 0.09726526, 0.21527250, -0.00061254, 0.39976013
 .float -0.09776793, -0.25092090, -0.14615491, -0.15387711, -0.31581146, -0.20845447, -0.17755736, -0.14569835, -0.15363792, 0.02219786, -0.20892051, -0.14405105
 .float -0.27229413, -0.12028889, -0.01630251, -0.07874199, -0.04467352, -0.03092657, 0.02660739, -0.04376765, -0.15195467, -0.07327808, 0.01959475, -0.10596179
 .float 0.04194334, 0.08760583, 0.04908939, -0.16489236, -0.16452170, -0.04212027, -0.08007343, -0.05058494, -0.04804054, 0.10788368, 0.10087036, -0.16523445
 .float 0.02066371, 0.03787242, -0.04569956, -0.25062802, -0.03260591, 0.00606307, 0.04254577, 0.00544008, 0.02458994, -0.07036600, -0.08062509, -0.20871338
 .float 0.09279411, -0.00791999, -0.20601234, 0.05850580, -0.05870529, 0.10599296, 0.01219852, -0.09194678, 0.19186813, -0.00526427, -0.20846997, 0.17965657
 .float 0.01975962, 0.18664493, 0.00529902, 0.01315822, 0.23511381, -0.12571852, -0.06522236, 0.17749126, 0.19464366, 0.16688958, -0.06160292, 0.08465715
 .float 0.49363256, -0.11387134, 0.00866681, 0.28762692, 0.21164034, 0.14728406, -0.05745828, 0.15029594, 0.36802750, 0.05203911, -0.04562803, 0.19237243
 .float -0.00172981, -0.04809811, -0.15180725, 0.37484533, 0.19530977, 0.13742335, 0.18572521, 0.09616667, 0.05697057, -0.03138213, -0.08186758, 0.27694997
 .float -0.05565759, -0.08913138, -0.42092300, 0.05642978, -0.34794470, -0.46549120, -0.11476801, -0.20175803, -0.04434669, -0.01311728, -0.09848766, -0.21297559
 .float -0.11455596, -0.21984571, -0.04690041, -0.09005143, -0.12176998, 0.02003124, -0.04559337, -0.26974657, -0.13441639, -0.07662972, -0.01711525, -0.01526852
 .float -0.06655369, 0.05843880, 0.04679088, -0.23510756, -0.08704065, -0.09096652, 0.02188693, -0.10449320, -0.06368474, 0.15702443, -0.01315654, -0.09427517
 .float -0.12241162, 0.02107054, -0.00676152, -0.08763204, -0.10093313, 0.10341500, -0.06785438, -0.04042281, -0.03491147, -0.02804390, 0.08286589, -0.25609330
 .float -0.02016717, -0.14180823, -0.35707540, 0.12956420, -0.15853733, 0.12604470, -0.12987036, -0.12705094, -0.00061344, -0.30579110, -0.49036834, 0.21819365
 .float -0.09189279, 0.28238285, -0.14951353, -0.05084683, 0.17492563, -0.48630032, -0.44560800, 0.41343716, -0.03718041, 0.19649558, -0.14419076, -0.00497685
 .float 0.22396915, -0.33536530, -0.57301134, 0.36403802, -0.17607066, 0.13649840, -0.18197492, 0.13868707, 0.27609012, -0.20469569, -0.47831900, 0.22026879
 .float -0.15484580, -0.00976502, -0.15447418, 0.14633143, 0.22519587, 0.05594124, 0.29273272, 0.24296643, 0.12153806, 0.21441408, -0.02139478, 0.18187045
 .float 0.07922340, 0.03519710, -0.34087795, -0.01769040, -0.25486773, 0.07831234, -0.09304352, -0.04028330, -0.03966241, 0.11802707, -0.10524420, -0.15876664
 .float -0.06366123, -0.12094600, 0.04902726, 0.06579237, -0.18740387, 0.12589474, -0.12733842, -0.40061018, -0.18572462, 0.01520517, 0.10828146, 0.09275461
 .float -0.19814667, 0.03600615, -0.07475747, -0.19443652, -0.09345639, -0.03354109, 0.17873400, -0.05045276, -0.15665807, 0.06672873, 0.11740012, -0.06716164
 .float -0.05409189, -0.03178315, 0.03655605, 0.08900828, -0.13258870, 0.02360459, 0.09906566, 0.01889939, -0.15000051, 0.07387941, 0.06417932, -0.19493057
 .float -0.21502812, -0.15759152, -0.19594038, 0.19382435, -0.10759809, 0.15286501, -0.08472054, -0.12684129, -0.23326701, -0.26630670, -0.44741030, 0.39056277
 .float -0.33432016, 0.28125212, -0.12422872, -0.11189836, -0.26059273, -0.44395563, -0.56571250, 0.24974585, -0.48742342, 0.24904574, -0.29584855, -0.16071384
 .float -0.05150344, -0.44332007, -0.73926693, 0.33923180, -0.46365696, 0.07176034, -0.22359091, -0.23567529, 0.08462044, -0.41128560, -0.59843916, 0.38092800
 .float -0.17383076, -0.00416157, -0.32922910, -0.01507993, 0.06484278, -0.01822073, -0.05866628, 0.15942296, 0.08124645, 0.29374403, -0.16497421, 0.32889894
 .float -0.08784541, 0.06828888, -0.40829500, -0.01250413, 0.00544249, -0.13616373, -0.08204005, 0.00042762, -0.06174705, 0.09955019, -0.15958498, -0.27875724
 .float -0.07703122, 0.04661931, -0.01062063, 0.06046214, -0.10063915, 0.09495843, -0.15999767, -0.18875213, -0.13273379, 0.15974346, 0.11692964, -0.03355898
 .float -0.12560268, 0.03675846, -0.07491375, -0.09443208, -0.09409123, 0.05494838, 0.11772544, 0.05119560, -0.13897072, 0.04867543, 0.07840411, -0.13061585
 .float -0.12832838, 0.06703632, 0.05192960, -0.01434837, -0.14612198, 0.06518176, 0.06171885, 0.07427558, -0.10305182, 0.12185314, 0.11892264, -0.08623847
 .float -0.25625450, -0.04435540, -0.06930567, 0.28656470, -0.24269617, 0.13972385, -0.02188041, -0.11296507, -0.46169794, -0.07599805, -0.33380452, 0.30622536
 .float -0.33326915, 0.15429102, -0.03544112, -0.02568183, -0.44414467, -0.07116415, -0.35383040, 0.06701247, -0.45441824, 0.05712049, -0.19549364, -0.21903253
 .float -0.23815319, -0.23837925, -0.53605130, 0.16956533, -0.27461138, -0.03476429, -0.12616052, -0.40222928, -0.10783656, -0.17509031, -0.56489770, 0.27611476
 .float -0.38862500, -0.17326313, -0.27638000, -0.29855427, 0.07363819, -0.00942835, -0.26553497, 0.20422739, -0.13993616, -0.22579308, -0.17584175, -0.20461096
 .float -0.04668755, -0.01032609, -0.28843886, -0.01298785, 0.12330493, -0.05858945, -0.13709360, -0.04049677, -0.02723086, -0.04850669, -0.07541101, 0.17759055
 .float 0.00287737, 0.17214479, 0.04269077, 0.00852639, -0.05810817, 0.04617311, -0.13550754, 0.23947106, -0.12804830, 0.16498177, 0.10713781, 0.08119096
 .float -0.13214184, 0.13658205, -0.16013625, -0.05587160, -0.06175800, 0.12413783, 0.08071424, 0.21925965, -0.17625506, 0.00076837, 0.04378328, -0.05444299
 .float -0.08837235, 0.07125116, -0.01964758, 0.09189055, -0.05531755, 0.05115657, 0.07793025, 0.02103222, -0.07000302, 0.07619509, 0.00441736, -0.01309630
 .float -0.15574060, -0.12768364, -0.05306036, 0.14874010, -0.14284547, -0.03183962, 0.07951415, 0.00073943, -0.22883247, -0.11731446, -0.15298223, 0.14017339
 .float -0.29235430, -0.06615854, -0.09472867, 0.01030782, -0.32363248, -0.05078625, -0.11547966, 0.05305358, -0.27146890, -0.09953578, -0.03569715, -0.12127937
 .float -0.13295655, -0.08712317, -0.19554585, -0.04344107, -0.18637714, -0.08740506, -0.13392362, -0.34749690, -0.02325175, 0.07891985, -0.19234625, 0.06073976
 .float -0.10951721, -0.08720651, -0.06199118, -0.29215974, -0.04788899, -0.16140161, -0.11443237, 0.10283699, -0.04035382, -0.16011669, -0.07112600, -0.53150030
 .float -0.17059030, -0.09534969, -0.09946571, 0.15846962, 0.10441276, 0.02821691, -0.07214625, -0.02712238, -0.04560152, -0.02568729, -0.19694060, 0.34341717
 .float 0.00879242, 0.11915968, -0.00822519, -0.05297498, -0.06810037, -0.16940070, -0.18944230, 0.37002920, -0.01683013, 0.08868367, 0.00188516, -0.10660139
 .float -0.21059479, -0.01137678, -0.22355428, 0.26501632, -0.03245637, 0.24377902, 0.00693515, -0.07556898, -0.25019008, 0.01371579, -0.17812385, 0.10619410
 .float 0.03377749, 0.34074417, -0.08754216, -0.08499691, -0.05739763, 0.01107770, -0.09289528, 0.13198927, -0.08619540, 0.04031981, -0.05386504, 0.01434416
 .float -0.05230258, 0.00296427, 0.00420043, 0.07437691, -0.08378755, 0.00821372, -0.08918526, -0.10430750, -0.04516362, 0.02944564, -0.04009873, 0.07751740
 .float -0.16337220, -0.01387181, 0.01989033, -0.12739661, -0.03166410, 0.01019868, -0.03259769, -0.08342522, -0.14234154, -0.11990053, 0.01618962, -0.01142580
 .float 0.06882678, 0.04407183, -0.06640489, -0.07886434, -0.07026311, -0.09185412, -0.06325907, -0.05946563, 0.17998856, 0.11783697, -0.00049852, -0.10106461
 .float 0.06797565, -0.16584402, 0.02140888, -0.09048852, 0.04279782, 0.01324016, 0.11150439, -0.05215413, 0.10984916, -0.12544066, 0.07402858, -0.11088883
 .float -0.10587377, -0.21449964, 0.03990206, -0.00314503, -0.00848034, 0.02594034, 0.02362727, 0.02495117, -0.04086806, -0.02563994, -0.08036675, 0.21959995
 .float -0.02541288, 0.08582469, -0.03291398, -0.02623713, -0.08599050, -0.11825919, -0.10513621, 0.30477938, -0.16190110, 0.06678318, -0.02977137, -0.03766705
 .float -0.02756127, -0.02161997, -0.25876504, 0.21013804, -0.01410183, 0.02525586, 0.02667371, -0.04985955, -0.15006268, -0.08159676, -0.18216099, 0.22913273
 .float -0.03096194, 0.02058203, -0.01314061, -0.11659805, -0.01327189, -0.07822871, -0.04578334, 0.06243222, -0.05803778, -0.17035808, -0.05187901, -0.09702726
 .float 0.00186615, 0.09521725, -0.02473762, 0.03985047, 0.02193461, -0.09399573, -0.00784792, -0.10374224, 0.04709806, 0.04954826, 0.03749570, -0.07457357
 .float -0.01575298, -0.08539677, 0.02998053, -0.09235715, 0.00033288, 0.01509171, 0.04591366, -0.11991393, -0.00567706, -0.08144964, 0.00738374, -0.07081803
 .float 0.08914142, 0.00164339, 0.00921760, -0.18678024, -0.02069361, -0.14783520, 0.05465645, 0.03685793, -0.02143292, 0.12585962, 0.08288399, -0.20591107
 .float 0.03637901, -0.00030465, 0.05384323, -0.07820576, -0.02104715, 0.00114730, 0.06864518, -0.02657597, 0.12754521, -0.10654980, 0.03365072, -0.18598868
 .float -0.08045971, -0.17892720, 0.07223625, 0.02078694, -0.14622307, -0.09239250, -0.12151296, 0.00274347, -0.11853495, -0.08200124, 0.26582354, 0.07134423
 .float 0.02472265, -0.02041760, -0.04362959, -0.04504687, -0.01735124, -0.04880041, -0.02498809, 0.27638116, -0.05489845, 0.03692623, -0.06019421, -0.07462106
 .float 0.03056995, 0.01903541, -0.18320267, 0.24175353, 0.01120140, 0.06672328, -0.07370676, 0.01731319, 0.16542262, -0.12751411, -0.13581394, 0.23671573
 .float 0.02448896, -0.02220197, 0.02737972, 0.02529136, 0.00518267, -0.08073398, -0.06263068, 0.07615659, 0.01159407, 0.11467236, 0.09518710, -0.09295890
 .float 0.06863482, 0.08142384, 0.00616800, 0.07307495, 0.12730433, 0.04969457, 0.04904006, -0.04606788, 0.04333883, 0.11227951, 0.03604690, 0.02756589
 .float 0.14271979, -0.08298759, -0.00510987, -0.00868081, 0.06316228, 0.04215589, 0.04296608, -0.02908468, 0.13524424, -0.00911701, 0.05528725, -0.03702806
 .float 0.11705709, 0.05933023, 0.12134522, -0.16073127, 0.05683537, -0.05001756, 0.00099464, 0.13407026, -0.05155187, 0.22997603, 0.03237606, -0.19416440
 .float 0.06228617, 0.10949403, 0.08149071, -0.05626227, -0.15224138, 0.23612055, 0.12134607, 0.00380026, 0.21607776, 0.06371674, 0.15244783, 0.05230870
 .float -0.05387809, -0.21079774, -0.01626639, 0.13427828, -0.11999968, 0.01819064, -0.04744886, -0.18965791, -0.17653473, -0.07052374, 0.07812549, 0.16618063
 .float -0.11231762, -0.01783081, -0.03670730, -0.11625906, 0.10220593, -0.07467350, -0.11003411, 0.18408872, -0.08846522, 0.00937354, -0.08348001, -0.09792639
 .float 0.10941138, -0.09102110, -0.10651083, 0.25591275, 0.03657360, 0.03078493, -0.00851115, -0.07502528, 0.06106998, 0.02375875, -0.01689948, 0.27755547
 .float -0.00793119, 0.01325766, 0.03620491, 0.01329809, 0.11775699, -0.02717682, -0.07948963, 0.23044983, 0.20569590, 0.00884945, 0.04068143, 0.05291162
 .float 0.08974998, -0.03927079, 0.03760240, 0.03041243, 0.10539404, 0.05135763, -0.03164605, 0.04350741, 0.10060813, 0.03837890, 0.08473390, 0.11408836
 .float 0.10347075, -0.09971044, -0.05565867, -0.00770360, 0.11546612, -0.02271114, 0.05877344, -0.08885413, 0.14220266, -0.17317490, 0.05609453, 0.03377135
 .float -0.00731638, 0.05200408, 0.08694129, -0.03972254, 0.01417372, -0.11946036, -0.02607703, 0.01382677, 0.11272888, 0.28368830, 0.03515634, -0.11872719
 .float 0.24429595, 0.03985801, 0.00853240, 0.12369060, -0.11617855, 0.17967048, 0.16256794, -0.08029233, 0.15085444, 0.22381814, 0.14235552, 0.02738349
 .float -0.03908519, -0.02544483, 0.02722802, 0.07832257, -0.10405333, -0.08937994, -0.00692320, 0.06925432, -0.11977512, -0.17492926, -0.11037270, 0.02228752
 .float -0.16486104, 0.07375497, 0.09015133, -0.19029498, -0.07840033, -0.20590243, -0.19214135, -0.00609423, -0.11127383, 0.05931317, 0.12119346, -0.05088765
 .float -0.16231237, -0.05781389, -0.06338370, 0.15082650, -0.00059860, 0.09925227, 0.01905857, 0.02328242, 0.00471486, -0.10708554, -0.04995562, 0.22253937
 .float -0.01655933, 0.09286161, -0.01294589, 0.03405638, 0.24355178, -0.10908490, 0.01526417, 0.36563160, 0.16037579, 0.12622413, -0.09595203, -0.01129571
 .float 0.22074613, -0.17002560, -0.03923087, 0.48682028, 0.23141134, -0.00455058, -0.10834974, 0.02980038, 0.36294234, -0.14224315, 0.05452526, 0.29674834
 .float 0.07117964, 0.02375644, -0.15814215, 0.02444931, 0.21228784, -0.03557046, 0.15943365, 0.19537040, 0.06980682, -0.00196455, -0.01625636, -0.02969278
 .float 0.27304590, 0.07615290, 0.11645354, 0.08389392, 0.14498323, 0.01834618, -0.03586259, -0.14396442, 0.19949743, 0.22351815, 0.14606602, 0.04070678
 .float 0.31065607, 0.10371820, -0.01735245, 0.03019730, 0.07995836, 0.22438185, 0.22910005, 0.01942014, 0.13938716, 0.10312337, 0.14449863, 0.19413762

# Row 7 of weights matrix
 .float 0.13669266, 0.17577313, -0.09180990, -0.00939615, -0.01596021, 0.10650215, -0.00474599, 0.22673330, 0.05728208, 0.15674588, 0.08361065, -0.00267949
 .float 0.12413876, 0.19670565, -0.00811314, 0.09017201, 0.11301034, 0.12021111, 0.35663950, -0.00304211, 0.16064797, 0.22319415, 0.06975388, 0.08874108
 .float 0.12904610, 0.08275992, 0.24734218, -0.00948110, 0.18336275, 0.01966222, 0.04841254, -0.05759548, 0.01187776, -0.01819290, 0.23771632, -0.12809007
 .float 0.04006828, -0.14288247, 0.07832453, 0.09373680, 0.13507119, 0.00157048, 0.14737627, -0.03163515, 0.10254529, -0.13208854, 0.07183313, 0.09377345
 .float -0.06453508, 0.09588509, 0.21696913, -0.03963631, 0.05696358, -0.10957933, 0.14090493, 0.19117819, 0.00429654, 0.05858004, 0.05156586, -0.00619069
 .float 0.08331332, -0.06609067, 0.08883576, 0.21887499, 0.17114780, 0.24032462, 0.20953040, 0.07485281, 0.00774028, -0.12122778, 0.03932819, 0.40316963
 .float 0.16707638, 0.16438265, 0.15517667, 0.13990377, 0.17884812, 0.00958165, 0.11199834, 0.23539732, 0.08758033, 0.18195006, 0.19633602, 0.13892262
 .float 0.11324964, 0.12397889, 0.10143727, 0.43264332, 0.15726791, 0.14441985, 0.03679896, 0.02924886, 0.07094872, -0.05507071, 0.08118966, 0.29384956
 .float 0.17511448, 0.05843666, -0.24788791, 0.00178785, -0.00133227, 0.08414538, 0.04615317, 0.10902864, 0.03839010, 0.06444781, 0.22345313, 0.02262934
 .float 0.11585126, 0.00971545, -0.03412264, 0.04521465, -0.01407063, 0.10432045, 0.17513572, -0.16229720, 0.03490771, -0.00912504, 0.04150907, 0.05666050
 .float -0.01605950, 0.11904309, 0.28055987, -0.09766327, 0.02677480, -0.14753810, -0.05715840, 0.03958937, -0.04682149, -0.08188564, 0.36548610, -0.26800360
 .float 0.05090061, -0.41141093, -0.09773947, -0.15335764, -0.02976044, 0.10810602, 0.17981620, -0.16109082, -0.05217539, -0.35407072, -0.05060819, -0.20619649
 .float 0.07462296, 0.01549559, 0.11270014, -0.09749630, -0.04124065, -0.25507146, -0.03103557, -0.22500545, -0.03293863, -0.02299531, 0.07091340, -0.09077273
 .float -0.02503893, -0.03090832, 0.06654609, -0.16768152, 0.02070698, -0.00644900, 0.05599479, -0.09260169, 0.12256918, -0.01155567, -0.03862859, 0.00047176
 .float 0.05847849, -0.07091625, 0.20633572, 0.03008472, 0.13138478, -0.01909321, 0.00375407, -0.02605752, -0.02227414, 0.00222278, 0.05966897, 0.05373021
 .float 0.00826911, -0.02603905, 0.07724419, 0.11370875, 0.01427005, 0.07588724, -0.03280143, 0.03113944, 0.10001833, 0.00466300, 0.06300396, -0.09697841
 .float 0.15427843, 0.04407583, -0.19267985, -0.08218884, 0.05683081, -0.12510477, 0.01952826, -0.00629892, -0.00214803, -0.02430132, 0.24950561, -0.11700892
 .float 0.06761219, -0.09059828, -0.10250743, -0.05917044, -0.09137136, 0.04468728, 0.10353243, -0.12146674, -0.01344077, -0.14153780, -0.09961235, 0.03120891
 .float -0.03648344, 0.09529012, 0.17148578, -0.18088959, -0.05960327, -0.41040120, -0.04846328, -0.03418975, -0.10269983, -0.02646050, 0.18093427, -0.24637085
 .float 0.06150125, -0.44063670, -0.12889095, -0.15456119, 0.06901912, -0.05104604, 0.18202832, -0.19568864, 0.05873468, -0.35009220, -0.13978045, -0.15800394
 .float 0.07187292, -0.06552104, 0.07215128, -0.26627380, 0.15047674, -0.12010742, -0.06247964, -0.01445050, 0.06051636, -0.25372050, 0.04697038, -0.04033295
 .float 0.05132315, -0.00609636, -0.07861132, -0.02708345, 0.03950749, 0.01018753, 0.03292787, -0.01240710, -0.06339204, 0.01991246, -0.02459043, 0.03835978
 .float -0.05665408, -0.14570415, 0.06455542, 0.14781860, 0.00083936, 0.18836533, 0.02059023, 0.07139959, 0.01650010, -0.24430445, -0.01703900, 0.02432067
 .float -0.01328260, 0.21462272, 0.04870485, 0.02725618, -0.15266988, -0.16649170, -0.18608639, -0.21809870, -0.08873392, 0.18760696, -0.13103385, 0.13344572
 .float -0.03716517, 0.09947841, -0.08245324, -0.15316583, 0.04935497, -0.13055119, -0.07269667, 0.04264510, -0.05023834, 0.01198005, 0.20826697, -0.12258863
 .float -0.09101766, -0.24660897, -0.06853270, -0.01026566, -0.04923524, -0.03722548, 0.22156900, -0.26383182, 0.08560795, -0.18157782, -0.15643883, 0.07035851
 .float 0.04379147, -0.01914863, 0.17170939, -0.29685396, 0.09715100, -0.34564847, -0.10827485, -0.14975885, 0.04246357, -0.09567828, 0.10947925, -0.19602634
 .float 0.07407756, -0.49486880, -0.00804368, -0.12442353, 0.04562616, -0.04617414, 0.13447988, -0.20980345, 0.08601262, -0.49248460, -0.15096220, -0.24159668
 .float 0.02897367, -0.12834820, 0.22222169, -0.01490389, 0.03008893, -0.35536212, -0.25501960, -0.14832726, 0.08981682, -0.20816562, -0.15823110, 0.16167367
 .float -0.04444330, -0.19925256, -0.20130860, -0.02758144, 0.09346905, -0.14031811, -0.31225967, 0.32259600, -0.02084494, 0.05783000, -0.22180244, -0.02041172
 .float -0.00623627, -0.31151940, -0.02708122, 0.31856075, -0.09232367, 0.12295633, -0.07090694, 0.01269282, -0.09491529, -0.29158154, -0.22704297, 0.21463767
 .float -0.11313178, 0.11910557, -0.11082818, 0.11453462, -0.18640910, -0.19264597, -0.36496815, -0.14348124, -0.15924698, -0.15421265, -0.20253839, -0.11610194
 .float -0.10001034, 0.05084540, -0.14277782, -0.12530020, 0.05500533, -0.17727663, -0.12803817, 0.00551727, 0.06436907, -0.05924711, 0.18796949, -0.13300188
 .float -0.02270410, -0.12604463, -0.00751469, -0.04040813, 0.07083134, 0.10880669, 0.23543958, -0.22057077, -0.01585005, -0.16270000, -0.17216346, -0.00445192
 .float -0.04357430, 0.08328956, 0.23280694, -0.19442098, 0.05245760, -0.35357797, -0.02961147, -0.10386942, 0.11257404, -0.06514180, 0.11778694, -0.20636527
 .float 0.06265143, -0.37307400, -0.06518002, -0.26112180, 0.03676102, -0.16708565, 0.14886765, -0.28012383, 0.04241590, -0.27783070, -0.05286007, -0.35159495
 .float 0.08312240, -0.20312023, 0.06444026, 0.19927171, -0.07781632, -0.24312648, -0.10537824, -0.24711493, -0.14195187, -0.33959920, -0.14473265, 0.28808483
 .float -0.04377275, -0.02847989, -0.17844787, -0.19035147, -0.10234325, -0.25444236, -0.31722566, 0.26671526, -0.13298875, 0.01497382, -0.21779867, -0.01616703
 .float -0.13413015, -0.18947390, -0.29944545, 0.24594398, -0.23291224, 0.09005903, -0.07371324, 0.06237849, 0.02504926, -0.19375686, -0.24477354, 0.18768112
 .float -0.22586916, -0.00608755, -0.07121321, -0.20182543, -0.04651736, -0.18930586, 0.01054823, 0.02102981, 0.01962950, -0.38057840, -0.16013000, -0.68957260
 .float -0.06624570, -0.01389392, -0.20712600, -0.18667655, -0.09105149, -0.03563211, -0.11193863, 0.12867494, 0.01625695, 0.01010728, 0.31944427, -0.15915899
 .float -0.05470071, -0.13157757, -0.06328685, 0.10702267, 0.07611275, 0.10075361, 0.33161578, -0.26135913, -0.03760006, -0.15246764, 0.00256789, 0.01074342
 .float 0.05535499, 0.11357310, 0.08347386, -0.28504303, 0.04156634, -0.23348981, -0.05807222, -0.03776029, -0.01108519, -0.06736792, -0.00369438, -0.32085860
 .float 0.13528372, -0.45675293, -0.01833825, -0.25546864, -0.00171894, -0.16769749, 0.09784205, -0.12308931, -0.02424906, -0.21970035, -0.05737810, -0.24593918
 .float 0.03171265, -0.21899250, 0.02046024, 0.21318097, -0.00743049, 0.10667612, -0.04918224, 0.00134622, -0.07125829, -0.22144613, -0.23868029, 0.20986427
 .float -0.23444013, 0.18603410, -0.06000310, 0.00660429, -0.21317942, -0.12034210, -0.39737228, 0.27144527, -0.17323753, -0.02732181, -0.03328912, -0.09982113
 .float -0.06071349, -0.14458698, -0.18711951, 0.18127683, -0.20786320, -0.06326056, 0.00254870, -0.07883214, -0.16014130, -0.05169248, -0.04802438, -0.01306566
 .float -0.05721942, 0.03738379, -0.06924001, -0.04700703, -0.09626681, -0.07140175, 0.06841431, 0.05869631, 0.06146209, -0.07886058, 0.07621008, -0.35528332
 .float 0.01586443, 0.00098964, -0.14146994, -0.11517077, -0.20258842, -0.01803217, -0.05616727, 0.13969240, 0.03036445, 0.03099417, 0.19235826, -0.20384257
 .float 0.06603175, -0.02618877, 0.08191570, 0.06184755, 0.04943928, 0.09262401, 0.28326404, -0.41830270, -0.07151858, -0.24527052, -0.06681681, -0.01163331
 .float 0.05050193, 0.10074709, 0.07350011, -0.38254863, 0.05156843, -0.21510167, 0.01165770, -0.00465433, -0.11514785, 0.08390322, 0.05847274, -0.26410335
 .float 0.06902733, -0.28022710, 0.00795223, -0.14135031, 0.01290711, 0.02052129, 0.17963545, -0.05990535, 0.02186150, -0.01906359, 0.05306466, -0.28119340
 .float 0.10876175, -0.12268086, -0.09089614, 0.17110465, -0.10825143, 0.11036982, 0.02032275, 0.00425989, -0.02298179, 0.04473759, -0.04377851, 0.28624600
 .float -0.08528168, 0.06711730, 0.00863129, -0.00045986, -0.19281785, -0.04076860, -0.12955750, 0.13079913, -0.07013483, 0.07518753, 0.01359581, -0.01541894
 .float -0.03185859, 0.00787296, -0.01070870, -0.02851503, 0.07610602, 0.00796432, -0.05591921, 0.00224257, -0.11254518, 0.04744238, 0.14598125, -0.15481138
 .float 0.05133554, 0.02119815, 0.03438589, 0.04932008, -0.11080471, 0.00321116, 0.06844057, -0.16098718, 0.07827191, -0.03000371, 0.05801180, -0.09516754
 .float 0.15189297, -0.03212243, -0.10529766, -0.16620210, -0.13481821, -0.20620096, -0.09833883, -0.04318883, 0.01539457, 0.04102287, -0.15252809, -0.29372552
 .float -0.08862661, 0.09046456, 0.13902910, 0.01856797, 0.01456923, 0.13358380, 0.13395557, -0.34554946, -0.07624021, 0.07092787, 0.03868460, 0.10178401
 .float 0.07875483, 0.19329564, 0.12510285, -0.30506170, 0.03136665, -0.02192779, 0.08626386, 0.07793944, 0.08911450, 0.09987063, 0.03224726, -0.31640407
 .float -0.02367200, -0.20450310, 0.03930331, 0.02903024, 0.15099975, 0.09338308, 0.00432123, -0.18924367, 0.04976715, -0.10645811, 0.21263643, -0.22371958
 .float 0.04724583, -0.00858175, 0.01234903, -0.03281228, 0.08410907, 0.03735963, 0.13940194, -0.06494359, -0.10256603, 0.04632343, 0.04367254, 0.05470473
 .float 0.07531246, 0.00585210, 0.06248028, 0.06071559, -0.02965187, 0.02143497, 0.17269930, 0.12756947, 0.07071216, 0.00569644, 0.01991548, -0.02171267
 .float -0.01134806, 0.02049977, 0.01036689, -0.15087284, 0.10089584, 0.04441776, -0.02811950, 0.14744303, -0.06057686, -0.05120359, -0.01987985, -0.25419720
 .float 0.10724394, 0.05728860, -0.06432141, 0.11659653, -0.17707591, -0.03184992, -0.02057486, -0.20177546, 0.06470895, 0.08544350, 0.03056264, 0.07175708
 .float 0.10489585, -0.02577054, -0.08777215, -0.13055697, 0.01848978, -0.11142904, -0.17175268, -0.09893283, 0.11129367, 0.01347936, -0.20093238, -0.22524562
 .float -0.13621664, 0.03383901, -0.00371752, -0.03005948, -0.08792894, 0.08533823, -0.05877740, -0.52679410, -0.02888260, 0.00188428, 0.07109996, 0.10364152
 .float -0.01974958, 0.19354647, -0.02327577, -0.34151113, -0.09043339, -0.04294619, 0.07332597, 0.13066780, -0.06849941, 0.20706144, 0.04528028, -0.34509450
 .float 0.01321892, 0.06622946, 0.14571110, 0.04338178, 0.07197086, 0.14881390, -0.15752086, -0.22131057, 0.13166851, 0.10420569, 0.18429907, -0.04821631
 .float -0.10711577, 0.03982340, 0.01637251, -0.13665773, 0.03155077, 0.15197009, 0.09551600, -0.08408865, 0.01062157, 0.04458488, 0.14181755, -0.10513836
 .float 0.11037133, 0.17305027, 0.11911191, 0.11619507, -0.03133477, -0.02892505, 0.04389001, -0.05817098, 0.03387217, 0.14863189, 0.00930187, 0.14335719
 .float 0.06046466, -0.07626043, -0.09518816, -0.15285668, 0.03912901, -0.03035841, -0.07470317, 0.17361180, -0.08709516, -0.02828821, 0.03641858, -0.00806859
 .float -0.08045491, -0.03370638, 0.01247424, -0.00622093, -0.17797020, 0.04113841, 0.01502174, 0.05202784, -0.02358461, 0.03756651, 0.04649961, 0.09077366
 .float -0.09844853, -0.03625108, -0.13483548, -0.12351671, -0.05123781, -0.12767853, -0.21557382, 0.01850364, -0.01878930, -0.04183424, -0.01864951, -0.18967181
 .float -0.03778813, 0.00253706, -0.21309577, 0.06704896, -0.08254775, 0.09374275, -0.16917467, -0.54139100, -0.04007366, 0.07209244, -0.02771914, 0.04842411
 .float -0.18019868, 0.04517772, -0.03757609, -0.51587373, -0.02277506, 0.07295819, 0.14265382, 0.08773745, -0.20372011, 0.20774250, -0.15225807, -0.12175965
 .float -0.03175299, 0.22769505, 0.05734776, 0.15017569, -0.22154018, 0.12988296, -0.08857784, -0.07913218, 0.01944002, 0.16909117, 0.15166797, 0.03611570
 .float -0.11889018, -0.02750573, -0.00710491, 0.04212033, 0.12091570, 0.21803968, 0.10799225, 0.12470628, 0.02089030, -0.01561609, -0.05987985, 0.02718298
 .float -0.01950841, 0.19651745, 0.10742842, 0.16754742, -0.01976697, 0.01052263, 0.07567558, 0.06790002, -0.04213300, 0.10124265, 0.02000177, -0.01800331
 .float 0.03742623, -0.06119708, 0.04458940, -0.12224338, -0.02189052, -0.07409624, -0.04478080, -0.07246368, -0.05310539, -0.12555312, 0.03027917, -0.02511761
 .float -0.07199433, -0.13423774, -0.03907396, -0.07320475, -0.06049867, -0.02421865, -0.09193084, -0.02672714, -0.16263907, -0.06685026, -0.12222272, 0.13762777
 .float -0.31398687, -0.13140854, -0.08521889, -0.00563564, -0.07466710, -0.10511109, -0.08743782, -0.10890837, -0.21207668, -0.06568919, -0.04193565, 0.00587701
 .float -0.07217369, -0.44327110, -0.19999158, -0.08905348, -0.02486379, 0.02593853, -0.15681475, -0.09438618, -0.02199371, -0.16417356, -0.24893291, 0.04107987
 .float -0.23898480, 0.08172143, 0.01922899, -0.22697254, 0.03476452, -0.03410103, -0.08073386, 0.03811059, -0.39583364, 0.18298776, 0.00292516, -0.24266460
 .float 0.10486446, 0.11982191, 0.08531704, 0.13802257, -0.54833560, 0.13470992, -0.04835248, -0.10600942, -0.03094030, 0.13106857, 0.11551159, 0.06889573
 .float -0.16023860, -0.13074042, -0.07010023, 0.22807224, -0.11147028, 0.13002454, 0.05540421, 0.09610983, 0.07701328, -0.14973737, -0.09610706, 0.20915370
 .float 0.00082510, 0.07876688, 0.03233167, -0.08009820, 0.06471938, -0.15457481, -0.04589884, 0.11483530, 0.02072305, 0.08974068, -0.02623721, -0.02944688
 .float -0.17537084, -0.22772165, -0.02102154, 0.10645293, -0.07893547, -0.07950018, -0.12149037, -0.06786653, -0.29697620, -0.23219119, -0.23651461, -0.02455416
 .float -0.22024842, -0.19960481, -0.11702965, -0.02286285, -0.02399362, -0.17692366, -0.32673190, -0.06476520, -0.19034575, -0.22082682, -0.28325173, -0.31014810
 .float -0.08941035, -0.14855298, -0.17475247, 0.01627941, -0.13356297, -0.08141203, 0.03258857, -0.15818422, -0.15026169, -0.37474903, -0.23830904, 0.06956158
 .float -0.28044530, -0.15928802, -0.04414176, -0.48422100, -0.14773685, -0.08020945, -0.16567524, 0.07853638, -0.08931721, -0.36556554, -0.11685263, -0.09924448
 .float -0.50609875, -0.06680343, 0.17008054, 0.10297561, -0.09657110, -0.20124420, -0.15934540, -0.07078914, -0.60435430, -0.08325431, -0.17105366, 0.08033094
 .float -0.10524715, -0.17344305, -0.11238191, -0.07495748, -0.64677560, 0.02843857, -0.13230580, 0.13216300, -0.14695078, 0.04602342, -0.20733494, -0.07009309
 .float -0.69030386, -0.07360780, -0.19215548, 0.21175504, -0.15121217, 0.08086542, -0.13054037, -0.08348566, -0.55801090, -0.18486655, -0.33270386, 0.20079362
 .float -0.10134722, 0.03683765, -0.06960840, 0.00818598, -0.13782617, -0.19970979, -0.22486702, 0.18519810, -0.13746364, 0.00296587, -0.12533380, 0.01769900
 .float -0.19004771, -0.29231918, -0.37692060, -0.01115973, -0.20294936, -0.10125151, -0.08506906, -0.06225871, -0.05145070, -0.36415425, -0.38175830, 0.06481598
 .float -0.26374844, -0.07284667, -0.05826801, -0.36866930, 0.01074476, -0.26996902, -0.17740032, 0.05363252, -0.07540345, -0.27790500, -0.18277104, -0.33862590

# Row 8 of weights matrix
 .float -0.02290276, -0.16858205, 0.04060171, 0.03066005, -0.01568161, -0.04782635, 0.05005439, 0.01613184, -0.16333793, 0.04019173, -0.19866829, 0.03394656
 .float -0.04008937, -0.24880680, 0.02100702, 0.05783281, -0.11508127, 0.06827200, -0.20208590, 0.00857186, -0.02380447, -0.47378463, 0.09823856, 0.07076506
 .float -0.26031074, -0.12999259, -0.06924462, -0.03702190, 0.05920488, -0.28853413, -0.00381048, -0.13773257, -0.12115684, -0.06706531, -0.30072474, -0.03588895
 .float -0.09240552, -0.37963430, -0.02803982, 0.01539174, -0.05628442, -0.10472371, -0.37890280, -0.03836674, -0.06958583, -0.47782955, 0.02795548, -0.24860692
 .float -0.03738371, -0.10702359, -0.10201808, -0.06113468, 0.01237119, -0.39276922, 0.07151344, -0.27076796, -0.10406051, -0.28140482, -0.16199063, -0.15109466
 .float -0.12172015, -0.39357140, -0.01409751, -0.73374385, -0.12311057, -0.53903680, -0.27803620, -0.22714430, -0.09985866, 0.04661608, -0.14530736, -0.43135208
 .float -0.22608790, -0.31606874, -0.48351374, -0.21298926, -0.30647805, -0.05515313, -0.39295300, -0.48616830, -0.20678256, -0.40594736, -0.40818378, -0.05660214
 .float -0.31519717, -0.11145075, -0.26740050, -0.33702552, -0.28684375, -0.19440657, -0.74403020, 0.04368692, -0.30176604, -0.07773412, -0.36454353, -0.08632317
 .float -0.02328402, 0.03657900, 0.13119848, 0.06222404, 0.05625984, 0.09187100, -0.08972260, -0.03711514, 0.00572301, -0.01415941, -0.01579669, 0.03074346
 .float 0.00635396, 0.17779307, -0.02308606, 0.07044255, -0.00093635, 0.06284352, 0.13905191, 0.08185192, -0.06404197, 0.29337808, 0.07252229, -0.01102953
 .float -0.05206823, 0.10954902, 0.11853987, 0.04310281, 0.04406163, 0.35101852, -0.04285083, 0.15308982, 0.06807286, 0.12847630, -0.15240374, 0.04755778
 .float 0.06531238, 0.29912996, -0.04370536, -0.10688288, 0.10401960, 0.02748730, -0.32479640, 0.04398158, -0.02867806, 0.22650644, 0.02825460, -0.34328580
 .float -0.05126082, -0.22652493, -0.30244282, 0.03742001, 0.01191635, 0.21938838, 0.09911697, -0.31583884, -0.03924638, -0.13017783, -0.14024843, 0.02391966
 .float 0.03316001, 0.04217971, 0.05249815, -0.22084686, -0.10282961, -0.13063544, -0.05216895, 0.00170662, -0.07374104, -0.10532893, 0.00386405, -0.30496624
 .float 0.00425643, -0.08828565, -0.01332513, -0.03965412, -0.04792800, -0.26478165, -0.03606354, -0.07746590, -0.11138043, -0.26371887, 0.03251820, -0.10279720
 .float 0.05507937, -0.47239650, -0.09194108, -0.35435230, -0.13924433, -0.17479780, -0.31151864, -0.11276007, -0.19488847, -0.33372880, -0.21535204, -0.60053870
 .float 0.06575573, 0.06279101, 0.15723540, 0.05221207, 0.03830580, -0.00024387, 0.00084621, -0.07358676, 0.08462664, 0.00843564, -0.16499594, 0.03350431
 .float -0.23498194, 0.17835380, 0.14857662, 0.05711280, 0.04099226, 0.08618353, -0.01657294, 0.13665871, -0.07421033, 0.14999874, 0.04997332, 0.02594611
 .float 0.07210081, 0.04553478, -0.01090650, 0.13558902, -0.17037313, 0.31172445, 0.13372853, 0.13117398, 0.01504815, 0.04087221, -0.12712045, 0.18569697
 .float 0.02438350, 0.23133814, 0.18834680, -0.04520810, -0.02750401, 0.04768254, -0.29004544, 0.20143539, 0.02215044, 0.09227160, 0.07939995, -0.09098693
 .float -0.00390899, -0.04159680, -0.21575549, 0.01906721, 0.03574515, 0.05853060, 0.05024040, 0.01129647, -0.03211704, -0.06505571, -0.04795523, 0.09249602
 .float 0.17254944, 0.01307962, 0.04398790, 0.00104141, -0.01104185, -0.14226167, 0.10201329, 0.01899046, 0.08076061, 0.04398637, 0.10170380, -0.13843770
 .float 0.05999788, 0.11409363, 0.13713837, -0.01683298, 0.05715424, -0.22736391, -0.01191925, -0.10553039, -0.05556824, 0.08159074, 0.10366746, -0.04456793
 .float 0.07361058, -0.22420670, 0.01027814, -0.21257702, -0.00533319, -0.07013795, -0.08037353, -0.01647183, 0.01049161, -0.34988880, -0.10138463, -0.05866022
 .float 0.01611197, 0.09608252, -0.20366470, 0.14935880, -0.06305821, 0.09950659, 0.17776994, 0.05988330, -0.09343638, -0.01647043, -0.43065855, 0.04663313
 .float -0.17220071, 0.11264332, 0.11165674, 0.01474260, -0.00845946, 0.12176811, -0.19153698, 0.14701973, -0.18363899, 0.25528000, 0.16263350, -0.03052862
 .float -0.07745062, 0.12880819, -0.18285342, 0.21543990, -0.10990240, 0.28568396, 0.32327282, 0.06064982, -0.03989005, 0.10011116, -0.06791852, 0.17255868
 .float -0.13321780, 0.15721940, 0.19442992, -0.07187163, -0.06179481, 0.06404617, -0.06169522, 0.20490249, -0.03552134, 0.13285361, 0.23218758, -0.02062702
 .float -0.13815074, -0.05373831, -0.00894381, 0.08094621, -0.03701485, 0.11758319, 0.12924127, 0.00816266, 0.10144498, 0.06305336, 0.16198276, 0.02578521
 .float -0.01117722, 0.15633209, 0.03924214, 0.05502844, 0.07945429, 0.07888881, 0.10497466, 0.07526381, 0.03986230, -0.02299120, 0.13233887, -0.01716757
 .float 0.09305088, 0.04465600, 0.14056683, -0.02328350, 0.09435423, -0.22687712, 0.05051230, -0.06468627, 0.16620752, -0.02498551, 0.18731710, -0.02456720
 .float 0.15889783, -0.18388640, 0.03458031, -0.15445809, 0.02558194, -0.11005477, -0.02595178, 0.08502790, 0.11687740, -0.13821024, -0.03254386, 0.08987519
 .float 0.02939037, 0.11660290, -0.17211185, 0.18958890, -0.09552989, 0.04248332, 0.07632905, 0.03462621, -0.03039916, 0.02676581, -0.27878570, 0.05730024
 .float -0.03912657, 0.03664337, 0.09486316, -0.06409410, -0.09882719, 0.05388239, -0.25004765, 0.23605959, -0.08576388, 0.18828475, 0.10679220, -0.00982348
 .float -0.14075900, -0.04056584, -0.08603298, 0.23010807, -0.07451782, 0.06611352, 0.14334379, -0.12577150, -0.17882752, -0.15208934, 0.04664356, 0.30647737
 .float 0.01767705, -0.01090036, 0.09717901, -0.21036910, -0.20643207, -0.04630223, -0.03124850, 0.21359374, -0.11200296, -0.01041578, -0.06445172, -0.07968520
 .float 0.02086518, 0.06701588, 0.15877017, 0.21585047, -0.15766919, -0.18574613, -0.12220454, 0.11274111, 0.11040411, 0.14466983, 0.10952802, 0.12942275
 .float 0.03734358, -0.06536828, -0.00211778, 0.11008144, 0.14656060, 0.05037743, 0.05148071, 0.09199119, 0.00918172, 0.05145496, -0.00934957, -0.04692639
 .float 0.14832084, 0.07676977, 0.14205104, 0.11223376, 0.08541840, -0.03501072, 0.03943151, -0.06593946, -0.01163465, -0.07851667, 0.13140234, -0.03426355
 .float 0.01979450, 0.02484580, 0.04449476, -0.10854311, 0.05749210, -0.16439688, -0.14019951, -0.02875151, -0.05988401, -0.10627090, 0.00170699, -0.13379700
 .float 0.01444076, 0.09727189, 0.03839629, 0.05498128, 0.19602811, 0.11152398, 0.03187086, 0.09759498, -0.08306095, 0.09031334, -0.09325697, 0.11932863
 .float 0.02883410, 0.01273416, 0.03175900, -0.00028368, -0.13180093, -0.06395274, -0.05651163, 0.04403602, 0.08989990, -0.06119749, -0.00410488, -0.16737351
 .float -0.05522448, -0.11820494, 0.11566773, 0.29623130, -0.04152668, -0.05158531, -0.12408430, -0.01730099, -0.08718848, -0.20457965, 0.18055962, 0.22917100
 .float -0.05100038, -0.25060636, -0.21218511, -0.16466779, -0.01758338, -0.27367973, -0.09289912, 0.14330891, -0.22081204, -0.46725821, -0.24636404, -0.18325305
 .float 0.09609211, 0.11686491, 0.08401789, -0.14936046, -0.23005687, -0.25382440, -0.24943128, -0.17438991, 0.09733304, 0.18525633, 0.13153046, -0.26139846
 .float -0.05740605, -0.05137776, -0.06741287, -0.05908581, -0.14963710, 0.03265983, 0.05247591, -0.05836418, -0.03980527, -0.04819473, -0.06707262, 0.04517540
 .float -0.10870986, 0.09680461, 0.02886538, 0.08357746, -0.03938597, -0.09167199, -0.01426040, -0.01386185, -0.09952455, 0.04770283, 0.01739894, 0.01648070
 .float -0.04675140, -0.07293381, -0.00344786, -0.23962577, -0.04110224, -0.23508008, -0.13939027, 0.04585686, -0.01369544, -0.05488088, -0.00626662, -0.22230455
 .float 0.11807172, 0.06724699, 0.29239923, -0.01219443, 0.17125367, -0.06622259, -0.12527774, 0.01864415, 0.05462397, 0.04248860, 0.09235265, -0.05723451
 .float 0.08219287, -0.22452116, -0.24477811, -0.02798631, -0.04222485, -0.15822595, 0.31859574, 0.30025718, 0.09989977, -0.33235595, -0.25529220, -0.02892071
 .float 0.19719700, -0.10773254, 0.32843380, 0.42496312, 0.08394816, -0.20440440, -0.22082832, -0.08969331, 0.25230715, -0.10583785, 0.09032854, 0.25531873
 .float -0.19242340, -0.18605416, -0.29134774, -0.12597010, 0.03164359, -0.22102731, -0.34500240, -0.01306356, -0.26874515, -0.45251077, -0.33918694, -0.26079744
 .float 0.12949352, 0.15137352, -0.01775129, -0.25005463, -0.14125624, -0.37566343, -0.15508214, -0.31239754, 0.00694689, 0.10576179, -0.01295845, -0.31097220
 .float -0.09344857, 0.08878999, -0.04097994, -0.12594223, -0.16911002, 0.11797220, 0.09829549, 0.14289589, 0.11198858, 0.34201255, 0.11444821, 0.01221032
 .float -0.04053827, 0.15870553, 0.07028376, 0.08035526, 0.03077641, 0.12761860, 0.11885627, 0.07151515, -0.11980712, -0.01011159, 0.14433694, -0.02505563
 .float 0.06926516, -0.15489018, 0.08207627, -0.12285099, -0.05706108, -0.21920680, -0.16402072, -0.12830092, -0.13978441, -0.26699640, -0.11014451, 0.05641381
 .float -0.01056959, -0.05549756, 0.31386850, -0.06285112, 0.00087081, -0.12806444, -0.10853633, -0.09711243, -0.06406753, -0.09284855, 0.30614102, 0.02858002
 .float 0.00477847, -0.40687513, -0.34766440, -0.03819882, 0.07838988, -0.15974954, 0.26162168, 0.15343691, 0.06840893, -0.19030459, -0.18725671, -0.12196460
 .float 0.36534128, -0.19976416, 0.43093535, 0.20438886, 0.02416183, -0.15800083, -0.30330327, -0.09954646, 0.14173414, -0.16451922, -0.09168212, 0.18793060
 .float -0.19209675, -0.30129537, -0.28728360, -0.07469589, -0.01950262, -0.13781077, -0.11979019, 0.01383919, -0.14383543, -0.21789733, -0.29170640, -0.27304360
 .float 0.10155876, 0.09959512, 0.05507442, -0.11251133, -0.01289583, -0.03964581, -0.08437687, -0.18146043, 0.06859390, 0.12608390, 0.03302161, 0.00168593
 .float 0.02123839, -0.07446542, -0.05108484, 0.02706962, 0.05420400, 0.00348275, 0.07457763, 0.19848454, 0.09248058, 0.16525610, 0.11937611, 0.12586209
 .float -0.04443722, 0.04208660, 0.17623499, 0.07976586, 0.08155045, 0.08729882, 0.04248796, -0.00641372, 0.02223072, -0.15999807, 0.11814173, -0.08654510
 .float -0.03997042, -0.01487073, -0.03847795, -0.07762388, -0.04605153, -0.04043583, -0.19240141, 0.06438539, -0.21108882, -0.10812763, -0.11986674, -0.16256268
 .float -0.00518610, -0.12076578, 0.12108227, -0.03041353, -0.03596712, -0.12959814, -0.22745572, -0.06756482, -0.14252204, -0.22570723, 0.12930582, 0.00676365
 .float 0.02620970, -0.28588610, -0.20888555, -0.17755015, 0.12786867, -0.24168818, 0.14761485, 0.11491109, -0.04550146, -0.20354302, -0.12177958, -0.10637568
 .float 0.11623173, -0.29513994, 0.21656474, 0.03999377, -0.12717693, -0.28296986, -0.20724317, -0.14449099, 0.03222662, -0.09577990, 0.06080381, 0.02674453
 .float -0.12805500, -0.29599540, -0.17722640, -0.14925724, -0.00790856, -0.01760704, 0.15963517, -0.13234437, -0.05969206, -0.22201443, -0.12418884, -0.19744658
 .float -0.03572158, 0.15927646, 0.03364093, -0.10539716, 0.04677475, -0.21434239, -0.11494026, -0.04926369, 0.04821136, 0.07339717, 0.00959283, -0.26130456
 .float 0.05383508, 0.02959934, -0.01568285, -0.09705902, 0.13857076, -0.02657751, 0.02358361, 0.00507875, 0.06308488, 0.02343811, -0.00712463, 0.04336909
 .float 0.01025963, -0.14968948, 0.13881297, 0.10529293, 0.05385848, 0.00018459, -0.12721866, -0.02195023, -0.07070381, -0.48674548, -0.11139813, 0.01705638
 .float -0.08496129, -0.01353151, -0.09697545, -0.02981078, -0.05338926, -0.21233894, -0.08242034, -0.03074211, -0.17688471, -0.04317217, -0.17592946, -0.09546650
 .float -0.12170915, -0.19076005, 0.00354106, 0.04580849, -0.03064578, -0.26116523, -0.06358064, -0.12044663, 0.02261980, -0.03422961, -0.04603106, 0.11485768
 .float -0.12882763, -0.30971680, -0.08392617, -0.02381735, -0.09898703, -0.16873685, -0.02587873, 0.01207676, 0.11925017, -0.37854630, -0.09773101, -0.14170223
 .float 0.00272448, -0.13493223, 0.08544044, -0.11974783, -0.07512206, -0.38434416, -0.14910804, -0.16466472, 0.02531211, -0.02699462, 0.10271805, -0.26032433
 .float -0.01725076, -0.36757230, -0.09045377, -0.14545055, -0.07711105, 0.07819216, 0.20678361, -0.29461917, -0.03523182, -0.32732360, -0.07596165, -0.11750638
 .float 0.00513204, 0.01364275, 0.03944143, -0.10285680, -0.09305519, -0.16302031, -0.06874070, -0.10876159, -0.12849608, 0.09986006, -0.10174231, -0.33881977
 .float -0.08166587, -0.17350915, -0.09830496, -0.22203805, -0.12802951, -0.07313509, 0.01030020, -0.13826490, 0.01996907, -0.18903613, -0.15184395, -0.19765143
 .float -0.17341691, -0.22132728, -0.03745112, -0.12597702, 0.02579969, -0.14345162, -0.17652449, -0.28646383, -0.12755506, -0.53405577, -0.17603125, 0.07051457
 .float -0.24867140, -0.12838183, -0.05017142, -0.15335971, -0.19770695, -0.19152308, -0.24801159, 0.07944952, -0.29396486, -0.19825993, -0.22327136, -0.22159150
 .float 0.03591208, -0.10191378, -0.26611190, 0.17999253, -0.11418200, 0.03652368, 0.06716118, -0.06922377, 0.06811247, -0.00748741, -0.07220934, 0.15889372
 .float 0.02999937, -0.07537488, 0.00369283, -0.05401035, 0.12240963, 0.00134573, 0.19039081, -0.01208569, 0.15276055, -0.14008287, -0.00353012, -0.19343999
 .float 0.05719638, -0.11597916, 0.14774035, -0.20422868, 0.17480756, -0.37723730, -0.14701413, -0.05185313, 0.11672777, -0.02120705, 0.14471194, -0.40951252
 .float 0.04896922, -0.30838966, -0.11005627, -0.09211324, -0.03421489, 0.04965066, 0.16485517, -0.59346724, -0.10486237, -0.13276155, 0.05601074, -0.10347386
 .float -0.08625261, 0.18857382, 0.00810589, -0.39899647, -0.14436340, -0.08952165, 0.02204036, -0.04030336, -0.15332882, 0.10690675, 0.03817508, -0.38658485
 .float -0.03682583, -0.10097481, 0.02498298, -0.15415476, -0.13664202, 0.04085486, -0.05118141, -0.41528338, -0.00194565, -0.15430038, -0.02497438, -0.23317584
 .float -0.17817861, -0.24112375, -0.16393852, -0.23480462, 0.01253969, -0.15687920, 0.02186007, -0.19903372, -0.22053316, -0.31515050, -0.14918664, -0.07255939
 .float -0.19323574, -0.33831722, -0.09424763, -0.19612642, -0.11980178, -0.08214266, -0.17709234, -0.03181748, -0.15415373, -0.45472172, -0.24202567, -0.37680808
 .float -0.00735043, 0.05196004, -0.11100553, -0.01897499, 0.01673352, 0.06365930, -0.02469316, -0.03036006, 0.17648678, 0.01289635, 0.12098183, -0.11924933
 .float 0.12296261, 0.04887145, -0.08517842, 0.02787218, 0.17493412, 0.09109686, 0.17449878, -0.07050702, 0.15975481, 0.09795088, -0.14910519, 0.06112976
 .float 0.17112258, 0.00145543, 0.20593628, -0.07449601, 0.05766476, -0.11404747, -0.08460263, 0.04262788, 0.07930606, 0.02820582, 0.25312850, -0.22243994
 .float -0.00324530, -0.06867200, 0.08073766, -0.07422572, 0.15146743, 0.04549036, 0.10498750, -0.23854896, -0.01646659, -0.00660753, 0.13284930, 0.01895549
 .float 0.04167214, 0.29722032, 0.15865412, -0.09127233, 0.08361331, -0.04817220, 0.09219385, 0.04219021, 0.07159887, 0.28446263, 0.08742478, -0.21461050
 .float 0.04095152, -0.05769141, 0.12188572, -0.00613246, -0.09752605, 0.12820080, 0.05083233, -0.14524338, 0.05360878, -0.05101646, 0.09731848, -0.21890582
 .float 0.05963372, -0.03452386, -0.13794576, -0.00113264, -0.05743971, -0.11634008, 0.04038361, -0.15197454, -0.07584774, -0.27027860, 0.10078099, -0.00189828
 .float -0.06812445, -0.34397380, -0.06530792, -0.31220844, -0.00953077, -0.14491244, -0.01591649, 0.09250016, 0.01100195, -0.32564574, -0.06080065, -0.32656200

# Row 9 of weights matrix
 .float -0.15199396, -0.26948640, -0.06856475, -0.14009142, -0.02699354, -0.15774825, -0.14444473, -0.41049027, -0.08054185, -0.07292934, -0.15324765, -0.17736351
 .float -0.08719967, -0.15353216, 0.00299351, 0.06933418, 0.08731722, 0.14047842, 0.01859894, -0.03408513, -0.13073982, -0.29900193, 0.06549679, 0.04742921
 .float -0.01576103, -0.06237420, -0.20289311, -0.02328652, 0.04504478, -0.33655953, 0.05830465, -0.02827090, 0.04767034, 0.03278966, -0.09262046, -0.01976825
 .float 0.15264289, 0.03947368, 0.01898114, -0.01714597, -0.02738597, 0.06684688, 0.05084533, 0.01298436, 0.04855987, 0.07363403, 0.00121193, 0.04124111
 .float 0.05370233, 0.00606484, 0.05743852, 0.09947854, -0.03140520, 0.05967966, -0.06106484, 0.08310967, 0.06274193, 0.07639375, 0.09065374, 0.08364584
 .float 0.12176057, -0.03698266, -0.00753822, 0.20899355, -0.01614794, 0.07903688, -0.09205981, -0.02144789, -0.05744262, 0.12708698, 0.02674634, 0.04319664
 .float 0.02074968, 0.09932445, -0.00745902, -0.08746858, -0.04994088, 0.07125520, -0.04108019, -0.08299936, 0.09013106, -0.14033520, -0.09687928, -0.05059369
 .float 0.04400920, 0.18647474, -0.03084198, -0.11800417, -0.12080844, 0.07293536, -0.06992090, -0.04991826, -0.04584830, 0.04498966, -0.02703702, 0.05247472
 .float -0.18249784, -0.09894451, -0.17821082, -0.17806928, -0.11787221, -0.47595623, -0.04945902, -0.15257731, -0.03626674, 0.04544530, -0.27199173, -0.10902077
 .float -0.16287144, -0.10512655, 0.00783176, 0.02758616, -0.02125390, 0.07410350, 0.06655044, 0.03749514, 0.08348893, -0.11006483, -0.08787680, -0.02803623
 .float -0.00470093, -0.01798176, 0.22557895, -0.00292308, -0.00997075, -0.21990106, -0.07097392, -0.07610624, 0.02014125, -0.03676984, -0.11057880, -0.03116728
 .float 0.03836000, 0.01225998, -0.06630053, 0.06017624, -0.06009629, -0.09470517, -0.07439020, 0.03447927, 0.04583521, 0.05819814, -0.10989031, 0.00075994
 .float 0.05613405, -0.03362968, 0.00241813, -0.02754913, -0.04123966, 0.09465511, -0.06300826, 0.02896157, 0.08165292, -0.00472246, 0.09949813, 0.05337836
 .float -0.02809054, 0.06846625, 0.03891805, 0.08125693, 0.08393624, 0.12669960, 0.04054708, 0.08263569, 0.00154672, 0.07835296, -0.09590866, 0.07386558
 .float 0.01178103, 0.11759432, -0.02086525, 0.03570290, 0.01005916, 0.22521974, -0.09539249, 0.03224924, 0.01762216, 0.04341180, -0.01169945, -0.08659603
 .float -0.05385598, 0.11055298, -0.03454867, 0.08880984, -0.02908853, 0.06829818, 0.01171763, -0.09523979, 0.02639281, 0.09251296, -0.09977481, -0.06698841
 .float -0.08723282, -0.06624089, -0.18335691, -0.08315069, -0.16860442, -0.00231442, -0.06486760, -0.11201681, 0.01606060, 0.05379894, 0.09344403, -0.04487704
 .float 0.00117510, -0.15134937, -0.05079997, -0.07701464, 0.07745835, 0.02455547, 0.17767547, -0.08160844, 0.14758393, -0.07419941, -0.09397946, 0.01598092
 .float -0.01014126, -0.04435419, 0.06951642, -0.08148498, 0.06092245, 0.04549181, -0.02547082, 0.02128903, 0.07913697, 0.01007117, -0.16583607, -0.13880672
 .float 0.06927726, 0.00295870, -0.05636026, 0.03752068, -0.01664164, -0.10245296, -0.01039084, -0.09927760, 0.07793543, -0.03999845, 0.00976534, 0.08485777
 .float -0.02035943, -0.11714055, -0.07311555, -0.07644287, 0.05888371, 0.10860536, -0.05935931, 0.01486624, 0.03088769, -0.09842244, 0.00693829, -0.03221396
 .float -0.05975890, 0.10020769, -0.10867910, 0.17379236, 0.04240157, 0.00885131, 0.00945433, 0.09177028, -0.04572795, 0.09109441, -0.00658464, 0.07532571
 .float 0.02595148, 0.00379490, 0.00794854, -0.01086854, -0.03696301, 0.01664363, 0.00799146, 0.14129843, -0.05344000, -0.02218852, -0.01588143, 0.00178073
 .float 0.00370592, 0.14034498, -0.04787922, 0.09906768, -0.03059735, 0.06734027, 0.07949428, 0.03902075, 0.04001284, 0.01711829, 0.02300631, -0.00525723
 .float 0.04078473, -0.03081570, -0.25361570, -0.09569711, -0.24258564, -0.11804602, 0.04531981, 0.05076370, -0.02477894, 0.02882314, -0.00293437, -0.06673652
 .float 0.15936917, -0.06637243, -0.09919068, 0.02226805, 0.07667763, -0.04463854, 0.07057866, -0.08479349, 0.05245466, -0.04186970, -0.01701940, 0.00624538
 .float -0.10213357, -0.05521385, 0.09563290, -0.14481568, 0.00793940, -0.03509500, -0.06149059, -0.02288884, -0.05964746, -0.04342958, 0.02286776, -0.12049973
 .float 0.08205991, 0.13941462, -0.01416015, 0.17921829, -0.03449217, -0.03688773, -0.02329901, 0.02963661, -0.04558833, 0.16318522, -0.07153103, 0.07230655
 .float -0.07332362, -0.12227815, 0.02936491, -0.09502193, -0.04017576, 0.15411024, 0.01774368, 0.08773023, -0.08130871, -0.03765404, 0.00375143, -0.15162943
 .float 0.00466618, 0.18425664, -0.10462313, 0.06589640, -0.26244706, 0.04423969, -0.01354682, -0.13513435, 0.00226168, 0.13475560, -0.02987700, 0.18600178
 .float -0.06334880, -0.00221059, 0.01451006, 0.08688091, -0.10510923, 0.08945574, 0.02689723, 0.23693790, -0.09902162, 0.17044973, 0.00586724, 0.02785256
 .float 0.03433833, 0.07186266, -0.05533561, 0.20303802, -0.02921023, 0.03831600, 0.07975916, -0.09558845, 0.02339430, -0.03754922, 0.02562592, -0.12252760
 .float 0.02963957, 0.03898608, -0.15127802, -0.01039192, -0.01102755, 0.08081150, -0.00271546, 0.07592777, -0.08834122, -0.00989705, -0.01238029, -0.17932780
 .float 0.07363246, -0.02820698, 0.01982028, 0.03673850, -0.05277193, -0.04541517, 0.01823801, -0.15029426, -0.02291398, 0.13218611, 0.02311417, -0.00871608
 .float -0.08972880, -0.02945351, 0.04378935, -0.27945820, -0.09624302, 0.24774984, 0.09300962, 0.02499494, -0.27146482, 0.07345929, 0.10305494, -0.17720658
 .float -0.09333933, 0.27524700, 0.02630987, 0.09536950, -0.32476070, 0.13526413, -0.00003932, -0.03970987, -0.08031444, 0.23441614, 0.02996310, 0.07162581
 .float -0.24306335, 0.04714425, -0.15326467, 0.00672750, 0.00060891, 0.16911958, 0.01734500, -0.04710757, -0.08207092, -0.08178023, 0.10458265, -0.14028557
 .float 0.11247455, -0.07705189, -0.12097307, 0.07627847, 0.01986570, -0.00879067, 0.23016414, -0.14391685, 0.15275133, -0.04529191, 0.01721682, 0.19339174
 .float 0.03500039, 0.06968346, 0.24130483, -0.16369446, 0.06942280, 0.13240388, 0.00593920, 0.27318615, 0.01622660, 0.27549124, 0.19080633, -0.12908332
 .float 0.20300756, 0.12732103, 0.06635184, 0.24282345, -0.01926674, 0.06107871, 0.24851054, -0.15445286, 0.08655626, 0.02575771, 0.11647465, 0.01051147
 .float -0.01776618, 0.05017235, -0.33904590, -0.22949928, 0.04110113, 0.03685612, -0.09823282, 0.15168713, -0.05982227, 0.04539083, -0.23260193, -0.08121793
 .float -0.05249650, 0.00211801, 0.11736936, 0.01675863, -0.01269166, 0.00195991, -0.14864390, -0.21267570, -0.19931172, 0.17364827, 0.09641596, 0.06329542
 .float -0.35740998, 0.03841218, -0.07512063, -0.12408805, -0.20731577, 0.19082847, 0.10182294, 0.07617636, -0.26678634, -0.01382712, -0.13634922, -0.18163352
 .float -0.14983809, 0.32814535, 0.12153284, 0.05411172, -0.21452685, 0.08609997, -0.06593239, -0.18878004, 0.07837841, 0.20275858, 0.06812274, 0.09014994
 .float -0.10636275, 0.13487202, 0.04490066, -0.30553553, 0.11840645, 0.13479345, 0.13213563, -0.12193812, 0.02783421, -0.16027515, 0.07476130, -0.18151377
 .float 0.17921408, 0.13593115, -0.04177122, 0.10281724, 0.23422714, -0.13786742, 0.02071071, -0.09768124, 0.14127259, 0.16270973, -0.11115126, 0.16912150
 .float 0.17402701, -0.18389723, 0.00317693, -0.05503578, 0.03878643, 0.07674108, -0.04944801, 0.10070202, 0.06829116, 0.10904959, 0.10655556, -0.13864058
 .float 0.19083196, -0.01648042, -0.13190070, 0.18057403, -0.22275098, -0.09036707, 0.10312755, -0.34573470, 0.01358194, 0.06828482, 0.10624400, 0.18628190
 .float 0.05035808, -0.04556454, -0.08189622, -0.13394819, -0.02723229, -0.02616640, -0.09301068, -0.08181627, -0.04213567, -0.00107142, -0.21326113, 0.06689306
 .float -0.07430398, 0.06881046, 0.13873205, -0.06063690, -0.00745032, 0.03448606, -0.24201746, -0.08179430, -0.07580580, 0.07189988, 0.17963448, -0.00623553
 .float -0.19965112, -0.01523097, -0.03014662, -0.16073982, 0.03049944, 0.11000059, 0.04903147, 0.05080822, -0.24458490, 0.03373180, 0.12919877, -0.19645235
 .float 0.11199029, 0.10443557, 0.08122627, 0.06798951, -0.19444777, 0.08760306, 0.01291231, -0.21625054, 0.18573968, 0.11672887, 0.06164506, 0.17936565
 .float 0.04778027, -0.03152863, 0.01315504, -0.00796695, 0.11024165, 0.08794477, 0.04884943, -0.04048586, 0.11707053, -0.20220393, -0.08451289, -0.10694894
 .float 0.03946123, 0.07410588, -0.11835133, 0.04611076, 0.14054415, -0.24738397, 0.01190165, -0.03560068, -0.04730882, 0.18827207, -0.16792247, 0.23346837
 .float -0.21091677, -0.24702083, -0.27413854, 0.00905879, -0.04900781, 0.19251715, -0.19341940, 0.21224792, -0.18391810, -0.14452581, -0.23698439, -0.02733458
 .float -0.09415186, 0.20043021, -0.22147892, 0.12007698, -0.02565502, -0.16727276, -0.21375947, -0.10129905, -0.06962936, -0.05891255, -0.14795572, 0.02165600
 .float -0.00638988, -0.12390466, -0.05649237, -0.15344514, -0.11826087, -0.03498440, -0.02199178, -0.18529785, -0.00233592, 0.02708952, -0.08116950, -0.10166036
 .float 0.04329179, -0.11809576, -0.10233608, -0.02975642, -0.03905944, 0.00569522, 0.07069338, -0.10228123, 0.11187506, -0.04491676, -0.05792238, 0.03858487
 .float -0.05504697, -0.03665906, 0.24243219, -0.12710373, 0.16674036, -0.01993827, -0.08997218, 0.01373291, 0.02526915, 0.04141079, 0.25007760, -0.41195870
 .float 0.11704955, -0.01749441, 0.03017524, 0.04892612, 0.08718799, 0.05160079, 0.08531614, -0.12647445, 0.05394983, 0.12725991, 0.06205393, 0.07560390
 .float -0.08128747, -0.12401006, -0.04358752, -0.13327508, -0.09897812, 0.15657040, 0.11974546, 0.04476162, -0.08334118, -0.09254525, -0.09544007, -0.08829957
 .float -0.12840065, 0.23379440, -0.07490237, 0.09722289, -0.30467615, -0.06378431, -0.21019961, -0.01055033, -0.21275508, 0.06010122, -0.14829634, 0.15842523
 .float -0.10152162, -0.05722882, -0.31042296, 0.11725404, -0.24861118, 0.02810148, -0.11486661, 0.06294809, -0.16580406, 0.06974113, -0.27253935, -0.03080645
 .float -0.18949881, -0.12378761, -0.13177593, -0.13343656, -0.02538750, -0.05187883, -0.13420135, -0.07117549, -0.02332594, -0.08072621, -0.10529973, -0.35559827
 .float -0.05289499, -0.11293933, -0.16887050, -0.18301989, -0.16711567, -0.04528565, -0.04327092, 0.07908317, 0.02704921, 0.03652570, 0.00793160, -0.09644341
 .float 0.12316632, -0.22786647, -0.07583733, -0.04192308, 0.15669581, 0.04347351, 0.21628568, -0.19866602, 0.20296581, -0.16808395, -0.10109097, -0.10920344
 .float 0.03093258, 0.01262901, 0.16970521, -0.33727193, 0.21084106, -0.22811271, -0.05663822, -0.04747205, 0.14716520, 0.00328547, 0.11812819, -0.46808153
 .float 0.09375596, -0.20278327, -0.04888012, 0.04376991, -0.05949150, 0.07805214, 0.01217207, -0.29835877, -0.18737486, -0.01203975, 0.01615992, 0.06580906
 .float -0.22018808, -0.11788447, -0.08803068, -0.10553527, -0.12871510, 0.19766164, 0.05709716, 0.06871525, -0.12368224, -0.09887227, -0.04582707, -0.09007977
 .float -0.19743780, 0.11327610, 0.02702838, 0.03419900, 0.00273048, -0.02726608, -0.05580528, -0.02269720, -0.09343762, -0.01501876, -0.04772552, 0.04227806
 .float 0.04260623, 0.01058222, -0.06292466, -0.18347688, 0.00018191, -0.10970950, -0.04388773, -0.19014995, -0.10101985, 0.04435737, 0.01164760, -0.25757253
 .float 0.03733854, -0.14486510, -0.06195213, -0.01606218, -0.11224719, -0.06845505, 0.03338548, -0.27138105, -0.02093424, 0.05590680, 0.01675950, -0.02624548
 .float 0.08012292, -0.07211531, -0.38417268, -0.14282148, -0.04915832, -0.12749435, -0.05270803, 0.00459302, 0.07377092, 0.09336788, -0.24920304, -0.14788891
 .float 0.03044752, 0.03406296, -0.01356742, -0.07904636, 0.06035826, -0.05196077, -0.04313954, -0.34021464, 0.03776437, 0.00215581, -0.05906425, -0.01670070
 .float -0.00963033, 0.01312681, 0.07684583, -0.21763618, -0.05872105, 0.03087383, 0.02328262, 0.02093995, 0.03389927, 0.06168571, 0.07553855, -0.36126986
 .float -0.03562726, 0.11311834, -0.03138558, 0.02256377, -0.17121914, -0.02425088, 0.06644370, -0.15096407, -0.03041241, 0.24633728, -0.02492100, 0.16485400
 .float -0.15215458, -0.14426368, 0.12417308, -0.15437163, 0.05687356, 0.25021765, 0.12858506, 0.13175255, 0.00823079, -0.09983858, 0.06461622, -0.24008375
 .float 0.10002796, 0.12279744, 0.06959412, 0.21913216, 0.03915834, -0.03861300, 0.09843643, -0.20199336, 0.00582413, -0.13643731, -0.03774269, 0.06779586
 .float 0.08486846, 0.04983759, 0.05790604, -0.37022886, 0.01193618, -0.08355691, 0.00702784, -0.04064836, -0.00350297, 0.09467631, 0.16444603, -0.29152095
 .float 0.02721500, -0.22542766, 0.08833057, 0.08096319, -0.11998445, -0.19416650, -0.00566402, -0.24450713, 0.03351841, -0.13432530, 0.04833144, 0.04835485
 .float 0.03808637, -0.11008540, -0.35510680, -0.21473737, 0.08848610, 0.06807846, -0.26569176, -0.07689849, 0.03157811, 0.04820855, -0.07348449, -0.49090707
 .float 0.07429663, -0.00974016, -0.11753891, 0.13676196, 0.00667515, 0.02474810, -0.16504537, -0.57798750, -0.10583516, 0.21154146, 0.05512562, 0.01217356
 .float -0.17932217, 0.08766042, -0.33863232, -0.46769500, -0.17834619, 0.17425694, 0.04641048, 0.03208932, -0.15597390, 0.04690779, -0.23386113, -0.29867557
 .float -0.09660172, 0.24696296, 0.13860735, 0.07566050, -0.25297183, 0.08895355, -0.10373302, -0.14632817, 0.09119291, 0.19569346, 0.05517695, 0.15731650
 .float -0.07015713, -0.00027905, 0.05329860, -0.11119279, 0.11294721, 0.06872781, 0.07175700, 0.13503976, 0.04971611, -0.05407613, 0.07784514, -0.28279072
 .float 0.01580244, 0.00210217, 0.00938882, 0.04154439, 0.05375067, -0.23763368, 0.09176165, -0.26680446, 0.07198890, -0.03389205, 0.01868264, -0.02498742
 .float 0.11136761, -0.11118289, 0.07525409, -0.37871143, 0.13845268, 0.04413093, 0.10167955, 0.12994692, -0.22820061, -0.11588460, 0.18161115, -0.17329246
 .float 0.10223924, -0.02608961, 0.02602194, 0.02420490, -0.10277888, -0.26817262, -0.31745526, -0.08016443, -0.00653925, -0.02117357, -0.03335917, -0.07209148
 .float -0.06755549, -0.03636399, -0.19722526, -0.23219725, 0.04218028, -0.56087990, -0.17086782, -0.06649480, -0.27425742, -0.04593105, -0.05173647, -0.22324660
 .float -0.12450141, -0.13633956, -0.18750305, 0.06434120, -0.04487301, 0.08874620, -0.21930327, -0.30020780, -0.10322989, -0.03713553, 0.12561314, 0.03961208
 .float -0.33200002, 0.00700752, -0.26555333, -0.23991945, -0.12907395, 0.18431800, 0.08962813, -0.10448829, -0.37773200, 0.01643972, -0.18990801, -0.27163870
 .float 0.06296260, 0.19248752, 0.14794835, 0.12597045, -0.42507970, 0.06496153, -0.11407440, -0.49123672, 0.06940577, 0.17475110, 0.10429684, 0.08303677
 .float -0.32507917, -0.08970672, -0.03465614, -0.57006985, 0.07481656, 0.07433509, 0.07995430, -0.00322407, -0.25151145, -0.03517656, 0.14235504, -0.27936274
 .float 0.03277303, -0.05565828, -0.07129954, 0.02309287, -0.21552900, -0.11868659, -0.01220575, -0.23100494, 0.13824004, -0.10063382, -0.02831858, 0.07768045
 .float -0.25884310, -0.19656385, 0.00637040, -0.10436358, -0.02684417, 0.03908206, 0.06735461, 0.07848406, -0.16691388, -0.05549169, -0.05537425, -0.11305075
 .float -0.16608160, -0.00556906, -0.04777757, 0.13942924, -0.12538467, -0.22342908, -0.27638210, -0.23115352, -0.15929186, -0.02747131, -0.12823972, -0.08056238

# Row 10 of weights matrix
 .float -0.09010229, -0.14513269, -0.08086255, -0.01306392, 0.04868444, -0.08819147, -0.17915320, -0.20215292, -0.05774823, -0.40332872, -0.13118072, -0.15866606
 .float -0.06293040, -0.14754754, -0.31590915, -0.17845751, -0.17947280, -0.14549644, -0.36307100, -0.19818576, -0.14072204, -0.36344936, -0.17828640, -0.15781318
 .float -0.02302559, -0.04749603, -0.63979554, 0.00317426, -0.11732180, -0.44619830, 0.06005312, -0.01789513, -0.04152071, 0.11315230, -0.45086825, 0.01792083
 .float -0.16430645, -0.66912300, 0.06958500, 0.02582492, 0.05854073, -0.04323874, -0.28927127, -0.02678940, -0.00268946, -0.60422800, 0.14590143, -0.38659972
 .float 0.01224595, -0.11152659, -0.27564085, 0.01796231, 0.01370313, -0.45439816, 0.00820544, -0.38861176, 0.07439105, -0.05417139, -0.32238388, -0.08397648
 .float -0.08099743, -0.28971466, 0.01164046, -0.33021775, -0.02576744, -0.09578331, -0.08136132, -0.14301209, -0.01388511, -0.32321018, 0.09522341, -0.43218133
 .float -0.14148451, -0.22703576, -0.03503419, -0.09786554, -0.03014312, -0.12935801, -0.03763662, -0.53121245, -0.11457155, -0.41312605, -0.21248743, -0.07267603
 .float -0.18733127, -0.18335001, -0.11862627, -0.27850142, -0.15758573, -0.16232277, -0.77224904, -0.08936761, -0.25738680, -0.02370007, -0.27313270, -0.05523703
 .float -0.28663990, -0.40807477, -0.23050329, -0.12525056, -0.04116713, -0.39478827, -0.34701973, -0.36052588, -0.01761492, -0.06829326, -0.38154660, -0.05771210
 .float -0.29206568, -0.49975127, -0.06946848, -0.07445330, 0.01381754, 0.06248239, -0.82816476, 0.02476682, -0.25684333, -0.75251460, 0.03490035, -0.00987227
 .float 0.06567003, -0.09720310, -1.10060970, -0.04889618, -0.05255224, -0.48761395, 0.00223890, -0.18963744, 0.02607619, -0.07073164, -0.74648910, 0.06014030
 .float 0.06271344, -0.17271920, -0.05016168, -0.10517847, -0.05913090, -0.03552516, -0.31724048, 0.02632453, 0.04904378, 0.04482662, 0.03644578, -0.00244117
 .float -0.01934825, -0.02866345, -0.19006965, 0.06380645, 0.04412991, 0.23666494, -0.05472286, 0.04930249, 0.02791043, -0.02092113, -0.12247048, 0.06367505
 .float -0.02913681, 0.22785728, -0.00840938, -0.04079716, 0.06928422, -0.07697886, -0.12289244, -0.04719408, -0.01002347, 0.18381725, -0.00419161, -0.08091413
 .float -0.02795941, -0.08395538, -0.12183131, -0.02594263, -0.10797724, -0.15407941, -0.02904987, -0.21804197, 0.04005825, -0.15377670, -0.17571133, -0.02302654
 .float -0.05425810, -0.43915826, 0.09355082, -0.82466450, 0.00282377, -0.28199705, -0.19724935, -0.15151455, -0.04852779, -0.25168142, -0.09536998, -0.79884845
 .float -0.26506940, -0.15062289, -0.33051810, -0.03900468, -0.27241210, -0.53155947, -0.17021905, -0.12364225, -0.05585565, -0.08877509, -0.42190245, -0.00778139
 .float -0.10619120, -0.34371108, 0.00821886, -0.11142205, 0.06741892, -0.08126596, -0.22246280, -0.02475797, 0.10295139, -0.23496178, 0.02222737, -0.24034043
 .float 0.04061296, -0.09217053, -0.12484895, -0.03696263, 0.11973013, -0.13965054, -0.14072770, -0.08821947, 0.00012787, -0.01984718, 0.05876458, -0.05695059
 .float 0.05771913, -0.16550370, -0.13038313, 0.25104240, 0.09433785, -0.10738329, -0.20307605, 0.07194593, 0.00472771, 0.12882243, 0.05027911, 0.18582778
 .float 0.05583968, -0.18637112, -0.20104246, 0.08919360, 0.03044604, 0.15446782, 0.02871107, 0.17238805, 0.03361697, -0.11416146, -0.12324946, 0.17198847
 .float 0.00416944, 0.14148574, 0.02247647, 0.17430910, -0.00555433, -0.05749126, -0.26818004, 0.13550237, -0.19691348, 0.09970203, 0.00693981, 0.20352039
 .float -0.03486513, -0.18819940, -0.11919739, -0.04148577, -0.05712265, 0.17334695, 0.01683951, -0.15710437, 0.01349549, -0.24294030, -0.20271035, 0.02765209
 .float -0.07171581, 0.05384796, 0.07693388, -0.12877184, -0.12654634, -0.32708466, -0.21095681, -0.10486158, -0.10305715, -0.19022219, -0.02446283, -0.35632940
 .float 0.00663791, -0.05308541, -0.24539709, 0.01693973, -0.09962646, -0.04125792, -0.09629451, -0.11267416, -0.03607748, -0.05247587, -0.00264433, -0.06787591
 .float 0.15984076, -0.14082098, -0.09101269, -0.00258717, 0.08002027, -0.00151316, 0.19976676, -0.08560365, 0.21506725, -0.21578906, -0.08388045, -0.08140002
 .float 0.11196892, -0.03667514, 0.03672645, -0.10146564, 0.08596326, -0.09417009, -0.15003360, 0.17324153, 0.17949694, -0.10009378, 0.04158313, -0.05263896
 .float 0.16548397, 0.02497262, -0.15678890, 0.23703581, 0.20749810, -0.04066795, 0.07469008, -0.04718973, 0.16252790, 0.13811733, -0.02132553, 0.21758434
 .float 0.05013130, 0.01938357, 0.06201871, 0.04069896, 0.13382116, 0.09032177, 0.03854061, 0.14996308, -0.06500978, -0.06020798, -0.13885756, 0.05186357
 .float -0.04320276, -0.06018844, 0.05106613, -0.01706453, -0.19790849, -0.04393181, -0.02344401, -0.02513749, -0.03724946, -0.03768204, 0.06459855, -0.05860902
 .float -0.20817670, -0.06730933, -0.08381039, -0.08687754, -0.08024380, -0.01863703, -0.03200248, -0.02242499, -0.14977676, -0.06527425, -0.23756374, -0.03662302
 .float -0.10553196, 0.01762469, 0.01322400, -0.12797382, -0.20468959, -0.12267625, -0.24572861, -0.08189772, -0.18055028, -0.11128377, -0.07504862, -0.35031807
 .float 0.00031078, -0.04596850, -0.05305853, 0.00795070, 0.06105126, -0.08495600, -0.01106261, -0.04341870, 0.08445023, 0.04571589, 0.24644507, -0.12564340
 .float 0.06847880, -0.07914086, -0.11422930, 0.06142849, 0.09445652, -0.05481521, 0.20812295, -0.17665736, 0.13588040, -0.08225222, 0.01850045, -0.00944197
 .float 0.09743360, -0.03180856, 0.10753518, -0.26050508, 0.06382643, 0.03913790, -0.02381610, 0.07730711, 0.19394529, 0.07458004, -0.06244295, -0.18148126
 .float 0.11714079, 0.25107256, 0.06627577, -0.01229261, 0.15296805, 0.07798837, -0.07695268, -0.16210747, 0.16784252, 0.20574874, 0.14870867, 0.05293047
 .float 0.00612327, -0.03069960, -0.12556507, -0.19846244, 0.11505669, 0.21125151, 0.13470125, 0.08972906, 0.02418029, -0.01251128, -0.03725412, -0.08986028
 .float 0.17036363, -0.02243916, 0.16385023, -0.06546330, -0.07108829, 0.10813024, 0.00244055, -0.01460660, 0.13229112, 0.03456172, 0.19321118, -0.03778856
 .float -0.12789637, 0.00412561, -0.12839167, -0.15145011, -0.01920639, -0.06432088, 0.10513413, -0.12240273, -0.14284721, 0.07994945, -0.02492655, -0.09482499
 .float 0.00292819, -0.04336491, 0.02234399, -0.12490279, -0.10945390, -0.03740951, -0.25508916, -0.04349674, -0.04835036, -0.10127813, 0.06465051, -0.20556755
 .float 0.06624258, 0.00892273, -0.02846682, -0.13231340, 0.09733403, -0.00621425, 0.05918480, 0.07398134, 0.06345903, -0.03257858, 0.11676985, -0.25779000
 .float 0.04286752, 0.10146893, -0.01434584, 0.08356467, -0.14382045, 0.10635941, -0.03908559, -0.24110144, 0.04948218, 0.18864733, -0.01078508, 0.05162314
 .float -0.01073286, 0.09431566, -0.05361787, -0.32858917, -0.02966199, 0.19185211, 0.10013761, -0.02616856, -0.20617881, 0.18235670, -0.06023116, -0.24561560
 .float 0.01726049, 0.24061131, 0.10711139, -0.07202045, -0.19426534, 0.01515446, -0.26288610, -0.24272107, 0.10950406, 0.29991192, 0.12381607, 0.16458485
 .float 0.07711813, 0.09949215, -0.06230619, -0.05441777, 0.11421969, 0.17551138, 0.13898154, 0.36783257, 0.07736837, -0.03672010, 0.06688940, 0.06549432
 .float 0.14653768, -0.09000879, 0.20159600, 0.11489521, 0.01791793, -0.05689778, -0.05645863, -0.05973272, 0.06595200, -0.20807300, 0.18713088, -0.10734369
 .float -0.15406348, 0.02447902, 0.05604190, -0.25543400, 0.12445887, -0.06784953, 0.05905835, -0.06691997, -0.05720988, 0.12244299, 0.09290070, -0.26752034
 .float 0.05550763, -0.02390138, 0.02682447, -0.08577553, -0.10669178, -0.02064616, -0.17117302, -0.21949380, -0.07629383, -0.08237236, 0.04450284, -0.01809862
 .float -0.06397448, 0.07264803, -0.21050021, -0.19197600, 0.08186707, -0.03882671, 0.01451244, 0.11427249, 0.06502283, -0.06336068, 0.06529091, -0.32268410
 .float -0.08589537, 0.08574328, 0.05729959, -0.01592541, -0.04305686, 0.05942811, -0.06634384, -0.40831226, -0.09247613, 0.10229109, 0.07975107, -0.07176236
 .float -0.23774692, 0.09571240, -0.07924610, -0.27073306, -0.15243143, 0.18325712, 0.09085209, -0.07886792, -0.35370687, 0.10187126, -0.10733667, -0.00029373
 .float -0.06422205, 0.30099470, 0.12957624, 0.00296831, 0.00864072, -0.04611608, -0.29060286, -0.07476126, 0.03776864, 0.11799041, 0.12330998, 0.30942774
 .float -0.04365060, 0.04232065, -0.00877165, -0.06461514, 0.05373127, 0.01027884, 0.06652964, 0.39678618, -0.06254612, 0.02169246, -0.03064492, 0.10323444
 .float 0.00214464, -0.21640357, 0.03496100, 0.15721253, 0.13959245, -0.05273409, 0.01826272, -0.17793162, 0.06161608, -0.36050025, 0.05223328, -0.13601513
 .float 0.06170685, 0.09877723, 0.15833928, -0.25052482, 0.00143933, -0.14463890, -0.03408760, -0.08128253, -0.14137982, -0.09356684, 0.13181797, -0.19652118
 .float 0.08984730, 0.02899826, -0.00068427, -0.12074657, -0.24509004, -0.30948360, -0.34130806, -0.16230537, -0.18490198, -0.09710942, -0.07603097, -0.22312373
 .float -0.06212605, 0.00648780, -0.27376896, -0.16212405, -0.09201425, -0.04804037, -0.11061835, 0.05066447, 0.02966390, 0.02626517, -0.02771023, -0.37626496
 .float -0.12365400, 0.08244258, -0.03401768, 0.03431527, 0.00215134, 0.12380664, -0.10966790, -0.36258290, -0.13421391, 0.15940312, 0.10295901, 0.06051783
 .float -0.40433884, 0.16274948, -0.31583685, -0.03891816, -0.13637170, 0.26859692, 0.21771653, 0.12081474, -0.37554422, -0.04020151, -0.18733232, 0.22654317
 .float -0.01069170, 0.19740424, 0.03657212, 0.12784623, -0.02078870, -0.17494245, -0.18735988, 0.15869972, 0.05645400, 0.10918097, -0.04834324, 0.27010587
 .float -0.04283036, -0.09055422, 0.00380152, -0.02162556, -0.03846810, -0.16439520, -0.14137161, 0.22706132, -0.03967301, 0.07774672, -0.03458269, -0.01669211
 .float -0.12680747, -0.35735926, -0.02351938, 0.07878257, 0.13404867, 0.00490804, 0.06287815, -0.42412537, 0.04276299, -0.33083153, -0.03933780, -0.07792708
 .float -0.06738865, -0.00606814, 0.13098466, -0.49298838, -0.01050629, -0.30974580, -0.04586316, -0.21313852, -0.18491274, 0.09019557, -0.02349055, -0.23429827
 .float -0.05535289, -0.01194894, -0.02375097, -0.21260253, -0.01964706, -0.17492561, -0.20976269, -0.16805884, -0.11322085, -0.25574570, -0.12455620, -0.34555572
 .float -0.07335611, -0.01636336, -0.19841404, -0.16564280, -0.15650962, -0.19517864, -0.33963950, -0.05014386, -0.02639741, 0.06910933, -0.05908246, -0.29920990
 .float -0.01879158, -0.03656099, 0.02402517, -0.05897361, -0.20432487, 0.00713267, -0.08253849, -0.08365461, -0.07525318, 0.10156777, 0.15643528, 0.02451148
 .float -0.32530698, 0.04395268, -0.41508016, 0.16157797, -0.15920568, 0.38248950, 0.18456016, 0.10563600, -0.23793466, -0.20333648, -0.08731071, 0.20316815
 .float -0.02974409, 0.23184097, -0.03407572, -0.02924861, -0.05599098, -0.25189602, 0.01679540, 0.04598687, -0.04929994, -0.01637466, -0.22354625, 0.17693806
 .float -0.07932926, -0.12918700, 0.03299740, 0.03054202, -0.04445600, -0.25508410, -0.14876604, 0.05672976, -0.05634584, 0.00735840, -0.02239527, -0.08386967
 .float -0.08973543, -0.39040506, -0.11645298, 0.00033813, -0.20662749, 0.12048039, -0.01692795, -0.63694880, -0.04022719, -0.19679826, -0.07718073, -0.07592526
 .float -0.03647316, -0.03288005, 0.01772385, -0.41647044, -0.13260348, -0.17543153, -0.05710113, -0.20093833, -0.01022570, -0.05391245, -0.10429702, 0.02262903
 .float -0.02999347, -0.12990578, -0.01419397, -0.19316228, -0.02457803, -0.02636842, -0.01731799, 0.05731232, -0.04262380, -0.22380380, -0.07411298, -0.30913654
 .float 0.05088159, -0.13865969, -0.14647932, -0.20446634, -0.14256090, -0.17501496, -0.23123254, -0.08152282, -0.15927237, -0.04085796, -0.11444777, -0.09872702
 .float 0.13058543, -0.03573833, -0.10553992, -0.10244877, -0.13547368, -0.04451488, -0.16634068, 0.11661889, -0.11841780, 0.10258900, 0.03769082, -0.07890800
 .float -0.13441835, -0.06839002, -0.19033694, 0.21731330, 0.06624033, 0.03728886, -0.08958888, -0.05943789, -0.05614579, -0.11587799, 0.00353589, 0.24892867
 .float -0.07038721, -0.01457963, -0.16386516, -0.05378929, -0.01962935, -0.13142556, 0.06701567, 0.21632092, -0.14566550, -0.19583753, -0.19280855, -0.07994982
 .float 0.04625775, 0.09324960, 0.03603124, 0.19070100, -0.19293454, -0.30309400, -0.16667691, -0.23245478, -0.05181762, 0.03417910, 0.02384760, 0.06191062
 .float -0.11682536, -0.28703910, -0.18642968, -0.25371027, -0.01661979, 0.06461339, 0.01966217, -0.17695875, -0.01462715, -0.19769081, -0.10665616, 0.00996413
 .float -0.07112216, -0.03376464, -0.01251946, -0.28130990, -0.06379490, -0.01756787, -0.02295226, 0.01647807, -0.00008162, 0.03901952, -0.01234301, -0.09269713
 .float 0.01068883, 0.07311695, -0.02515180, -0.04052667, -0.01656312, -0.04071904, 0.08305606, 0.00588220, 0.02236481, 0.12585862, 0.09270960, -0.03773832
 .float 0.01140681, -0.02327652, -0.01026242, -0.04711995, 0.07892980, -0.10623933, 0.02020703, -0.02195653, -0.00810254, -0.01041965, -0.00846383, 0.08405264
 .float -0.06045367, -0.06892037, 0.09413131, 0.03640110, 0.01284013, -0.03470616, 0.01572678, 0.02895084, 0.01301058, -0.05993651, 0.05365489, 0.01087279
 .float -0.04558846, -0.04095098, 0.16336769, 0.14968139, 0.08414403, -0.14656124, 0.02527365, -0.09873544, 0.08514593, -0.17504948, 0.15799017, 0.02040209
 .float -0.00832123, -0.13318026, -0.08358924, -0.27018046, 0.04317535, -0.00246511, 0.07004478, 0.03130296, -0.12349777, -0.14672618, -0.14626189, -0.10656424
 .float 0.04008083, 0.07900841, 0.03171541, -0.01316400, -0.12741860, -0.15054673, -0.05294887, -0.15232524, -0.01390254, 0.08930681, 0.00507021, 0.02331859
 .float -0.09460484, -0.06261183, 0.01211991, -0.17851858, -0.06040345, -0.00399059, 0.00379835, -0.02883681, -0.14432113, -0.05161224, 0.03858637, -0.00773547
 .float -0.20017484, 0.09903591, -0.01086483, -0.07896005, -0.04198665, -0.07417578, 0.04830798, -0.16781425, 0.11743125, 0.15235767, -0.04653298, -0.01067047
 .float 0.03015069, 0.04264413, 0.16794980, -0.06636981, 0.14481863, 0.06938727, 0.22209350, 0.00848175, 0.07926864, 0.19915555, 0.07090329, 0.04794420
 .float 0.13750175, 0.04597688, -0.08020501, 0.02645070, -0.10981619, 0.19076960, 0.15546083, -0.01133802, 0.12101401, 0.09731068, 0.04066047, 0.10755780
 .float 0.04432805, 0.07894380, 0.06707010, -0.00705456, 0.06567062, 0.01144304, 0.22425434, 0.06027973, 0.06933472, -0.00241888, -0.15742259, -0.05124711
 .float 0.19440569, 0.01799179, 0.07876785, -0.08458680, 0.05074759, -0.07086223, 0.01971910, -0.00857617, 0.16013215, 0.05375404, 0.15957624, -0.23537363
 .float -0.00453250, -0.03481023, -0.03629718, -0.12667403, 0.08724828, 0.08631545, 0.17238633, -0.27082214, -0.08672335, -0.07320594, 0.12916514, -0.09708516
 .float -0.03542179, 0.26956823, 0.02447217, -0.37636080, -0.12228665, -0.06163838, 0.15938586, -0.17792669, 0.00954943, 0.18390132, 0.05656720, -0.09709480
 .float -0.09719609, 0.05159062, 0.11390304, -0.08869129, 0.08971778, 0.23733053, -0.03042476, -0.00955553, -0.10454103, 0.15296458, 0.14806485, -0.03559879
 .float -0.09515617, 0.08845836, 0.12157021, -0.12884554, -0.00658677, 0.17559760, 0.22887497, -0.00403954, 0.03334153, 0.19673678, 0.01514277, 0.03998908
 .float 0.13671783, 0.16430902, 0.28625828, 0.10419934, 0.04845480, 0.20112390, 0.07201066, -0.02590391, 0.04619461, 0.10401170, 0.08259600, 0.17562105

#////////////////////////////////// Bias values for FC layer //////////////////////////////////////////////////////

bias:.float 0.08619133,0.22852448,-0.008509398,-0.17195569,-0.00046828244,0.07456757,0.0031446414,0.06568861,-0.13358837,-0.04876125

#////////////////////////////////// Softmax layer utilities //////////////////////////////////////////////////////
one:
    .float 1.0

half:
    .float 0.5

sixth: 
    .float 0.16666667



#////////////////////////////////// 8 outputs of convolution layer //////////////////////////////////////////////////////
R1:  .space 576*4    
R2:  .space 576*4    
R3:  .space 576*4    
R4:  .space 576*4
R5:  .space 576*4    
R6:  .space 576*4    
R7:  .space 576*4    
R8:  .space 576*4

#////////////////////////////////// 8 outputs of Max_pooling Layer//////////////////////////////////////////////////////

R_1:  .space 144*4    
R_2:  .space 144*4    
R_3:  .space 144*4    
R_4:  .space 144*4
R_5:  .space 144*4    
R_6:  .space 144*4    
R_7:  .space 144*4    
R_8:  .space 144*4


#////////////////////////////////// output of flatten function (1152 elements) //////////////////////////////////////////////////////
flat: .space 1152*4
#////////////////////////////////// output of FC layer (10 values) //////////////////////////////////////////////////////
fcOut: .space 40
#////////////////////////////////// output of SoftMax layer (10 probabilities) //////////////////////////////////////////////////////
softmaxOut:  .space 40


######################################################## The End ################################################################################################