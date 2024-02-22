## 00. Pytorch Fundamentals

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)

"""## Introduction to tensors

### Creating Tensors

PyTorch tensors are created using torch.Tensor() = https://pytorch.org/docs/stable/tensors.html
"""

# Scalar
scalar = torch.tensor(7)
scalar

scalar.ndim

# Get tensor back as Python int
scalar.item()

# Vector
vector = torch.tensor([7,7])
vector

vector.ndim

vector.shape

# MATRIX
MATRIX = torch.tensor([[7,8], [9,10]])
MATRIX

MATRIX.ndim

MATRIX[0]

MATRIX[1]

MATRIX.shape

# TENSOR
 TENSOR = torch.tensor([[[[1,2,3],[3,6,9],[2,4,5]]]])
 TENSOR

TENSOR.ndim

TENSOR.shape

TENSOR[0]

"""### Random Tensor

Why random tensors?

Random tensors are important becuase the way many neural network learn is that they start with tensors full of random numbers and then adjust those random numbers to better represent the data.

Start with random numbes - > look at data - > update random numbers - > look at data - > update random numbers

Torch random tensors - https://pytorch.org/docs/stable/generated/torch.rand.html
"""

# Create a random tensor if size(3,4)
random_tensor = torch.rand(3,4)
random_tensor

random_tensor.ndim

# Create a random tensor similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(224,224,3)) # height, width, color channels (R,G,B)
random_image_size_tensor.shape, random_image_size_tensor.ndim

"""### Zeros and ones"""

# Create a tensor of all zeros
zero = torch.zeros(size=(3,4))
zero

zero * random_tensor

# Create a tensor of all ones
ones = torch.ones(size=(3,4))
ones

ones.dtype

random_tensor.dtype

"""### Creating a range of tensors and tensors-like"""

# Use torch.range() and get deprecated message, use torch.arange()
one_to_ten = torch.arange(start=1, end=11, step=1)
one_to_ten

# Creating tensors like
ten_zeros = torch.zeros_like(input=one_to_ten)
ten_zeros

"""### Tensor datatypes

**Note:** Tensor datatypes in one of the 3 big errors you'll run into  with pytorch & deep learning:
1. Tensors not right datatypes
2. Tensors not right shape
3. Tensors not on the right device
"""

# Float 32 tensor
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None,   # what datatype is the tensor (float32, float64,...)
                               device=None,   # What device is your tensor on?
                               requires_grad=False)   # Whether or not to track gradients with this tensors operations
float_32_tensor

float_32_tensor.dtype

float_16_tensor = float_32_tensor.type(torch.float16)
float_16_tensor

float_16_tensor * float_32_tensor

int_32_tensor = torch.tensor([3,6,9], dtype=torch.long)
int_32_tensor

float_32_tensor * int_32_tensor

"""### Getting information from tensors

1. Tensors not right datatypes - to do get datatype from a tensor, can use 'tensor.dtype'
2. Tensors not right shape - to get shape from a tensor, can use tensor.shape
3. Tensors not on the right device - to get device from a tensor, can use tensor.device
"""

# Create a tensor
some_tensor = torch.rand(3,4)
some_tensor

# Find out details about some tensor
print(some_tensor)
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Device tensor is on: {some_tensor.device}")

"""### Manipulating Tensors (tensor operations)

Tensor operations include:
* Addition
* Subtraction
* Multiplication (element wise)
* Division
* Matrix multiplication
"""

# Create a tensor and add 1o to it
tensor = torch.tensor([1,2,3])
tensor + 100

# Multiply tensor by 10
tensor = tensor * 10
tensor

tensor - 10

"""### Matrix multiplication

# Two main ways of performing multiplication in nueral networks and deep learning:

# 1. Element-wise multiplication
# 2. Matrix multiplication
"""

tensor * 10

torch.mul(tensor, .5)

tensor

torch.matmul(torch.rand(3,2), torch.rand(2,2))

"""One of the most common errors in deep learning: shape errors"""

# One way to fix shape errors is to transpose one of the matrices
# 2x3 x 2x3 = error ...transpose 1st metrix = 3x2 x 2x3 = [3,3] matrix output shape
tensor_A = torch.tensor([[1,2],
             [3,4],
             [5,6]])
tensor_B = torch.tensor([[7,8],
             [9,10],
             [11,12]])
torch.mm(tensor_A, tensor_B)

tensor_A.shape, tensor_B.shape

tensor_B = tensor_B.T

torch.mm(tensor_A, tensor_B)

"""### Finding the min, max, sum, and etc (tensor aggregation)

"""

# Create a tensor
x = torch.arange(0, 100, 10)
x

# Find the min
torch.min(x)

x.min()

total = x.sum()
total

total/len(x)

len(x)

torch.mean(x.type(torch.float32))

torch.mean(x.type(torch.float32))

torch.mean(x.type(torch.float32))

x.type(torch.float32).mean()

"""### Positional Min and Max"""

torch.min(x)

x.min()

x.max()

x.argmax()

"""## .min and .max will retrieve the smallest and largest numbers, however .argmin and .argmax will retrieve the smallest and largest index value"""

## Reshaping, stacking, squeeze and unsqueeze tensors

# View - return a view of an input tensor of certain shape but keep the same memory as the original tensor
# Stacking - combine multiple tensors on tips of each other (vstack) or side by side (hstack)
# Squeeze - removes all 1 dimensions fron a tensor
# Unsqueeze - add a 1 dimension to a target tensor
# Permute - Return a view of the input with dimensions permuted (swapped) in certain way

import torch
x = torch.arange(1.,10.)
x, x.shape

# ADD another dimension
x_reshaped = x.reshape(1, 9)
x_reshaped, x_reshaped.shape

z = x.view(1,9)
z[:, 0] = 5
z, x
# [:,endIndex] = refers to first index to end index
# .view copys contents to new variable

# Stack tensors on top of each other
x_stacked = torch.stack([x,x,x,x], dim=1)
x_stacked
# torch.stacked allows you to stack your vectices in different dimensions

# torch.squeeze() - removes all single dimensions from a target tensor
print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimensions from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")

# torch.unsqueeze() - adds a single dimension to a target tensor at a specific dim (dimension)
print(f"Previous target: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

# Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew Tensor: {x_unsqueezed}")
print(f"New Shape: {x_unsqueezed.shape}")

x_reshaped.squeeze()

x_reshaped.shape

x_reshaped.squeeze().shape

# torch.permute - rearranges the dimensions of a target tensor in a specified order
x_original = torch.rand(size=(224, 224, 3)) # [height, width, colour_channels]

# Permute the original tensor to rearrange the axis (or dim) order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {x_original.shape}")
print(f"New Shape: {x_permuted.shape}")

x_original[0,0,0] = 728218
x_original[0,0,0], x_permuted[0,0,0]

"""# Indexing(selecting data from tensors)

 Indexing with Pytorch is similar to indexing with Numpy
"""

# Create a tensor
import torch
x = torch.arange(1, 10).reshape(1,3,3)
x, x.shape

# Lets index on our new index
x[0]

# Let's index on the middle bracket (dim=1)
x[0][0]

# Let index on the most inner bracket (last dimension)
x[0][2][2]

# You can also use ":" to select "all" of a target dimension
x[:,0]

# Get all values of 0th and 1st dimensions but only index 1 of 2nd dimension
x[:,:,1]

# Get all values of the 0 dimension but only the 1 index value of 1st and 2nd dimension
x[:,1,1]

# Get index 0 of 0th and 1st dimension and all values of 2nd dimension
x[0,0,:]

# Index on x to return 9
x[:,2,2]

# Index on x to return last column
x[:,:,2 ]

"""## Pytorch tensors and Numpy

NumPy is a popular scienitific Python numerical computing library.

And becuase of this, Pytorch has functionality to interact with it.

* Data in Numpy, want in Pytorch tensor - > 'torch.from_numpy(ndarray)'
* Pytorch tensor -> NumPy -> torch.Tensor.numpy()
"""

# NumPy array to tensor
import torch
import numpy as np

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
array, tensor

array = array + 1
array

tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
tensor, numpy_tensor

# Change the tensor, what happens to numpy tensor
tensor = tensor + 1
tensor, numpy_tensor

"""## Reproducbility (trying to take out the random out of random)

In short how a nueral network learns:

Start with random numbers - > tensor operations -> update random numbers to try and make them of data -> again -> again -> again...

To reduce the randomness in NN and pytorch comes the concept of a **random seed**

Essentially what the random seed does is "flavor" the randomness.
"""

import torch

# Create two random tensors
random_tensor_A = torch.rand(3,4)
random_tensor_B = torch.rand(3,4)

print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)

# Let's make some random but reproducible tensors
import torch

# Set the random seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3,4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3,4)

print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)

"""## Running tensors and Pytorch objects on the GPUs (and making faster computations)

GPU's = faster computation on numbers, thanks to CUDA + NVIDIA hardware + Pytorch working behind the scenes to make everything hunky dory (good)

### 1. Getting a GPU

1. Easiest - Use Google Colab for a free GPU (options to upgrade as well)
2. Buy your own GPU
3. Use cloud computing - GCP, AWS, Azure
"""

import torch
torch.cuda.is_available()

from torch.cuda.random import device_count
# Set up device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device

"""## Putting tensors (and models) on the GPU

The reason we want our tensors/models on the GPU is because using a GPU results in faster computations
"""

# Create a tensor
tensor = torch.tensor([1,2,3], device = "cpu")

# Tensor not on GPU
print(tensor, tensor.device)

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
tensor_on_gpu

"""### 4. Moving tensors back to the CPU"""

# If tensor is on GPU, can't transform it to NumPy
tensor_on_gpu.numpy()

"""### Excercises & Extra-curriculum

Here's the link: https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises
"""

