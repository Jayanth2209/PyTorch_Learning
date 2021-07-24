# PyTorch Basics and Linear Regression
## Install PyTorch
You can instead run on Colab/Kaggle
````
# Linux / Binder
# !pip install numpy torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# Windows
# !pip install numpy torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# MacOS
# !pip install numpy torch torchvision torchaudio
````

#### Import Torch
````python
import torch
````
#### What is a Tensor? 
[Tensors](https://pytorch.org/docs/stable/torch.html)
: A Tensor is a number, vector, matrix, or any n-dimensional array. A tensor has an additional property of uniformity in the data type of all entries and that they are always of a proper shape i.e., the lenghts of rows are all same, and the same goes with columns.        
````python
t1 = torch.tensor([[1,2,3],
                  [4,5,6],
                  [7,8,9]])
````
t1 is a 3x3 Tensor with dtype "int64" 
````python
t1 = torch.tensor([[1,2,3],
                  [4,5,6],
                  [7.0,8,9]])
````
Now, t1 is a 3x3 Tensor with dtype "float32". Since one entry (7.0) is a floating point entry, the rest will be converted to FP too.             
To obtain the shape, we use
````python
t1.shape
````
Here is an example of a 3D Tensor of size 2x3x4
````python
t2 = torch.tensor([[[1,2,3,4],
                   [5,6,7,8],
                   [9,10,11,12]],
                   [[9,10,11,12],
                   [13,14,15,16],
                   [17,18,19,20]]])
````
Similarly nD Tensors can also be constructed.         
#### Gradients
We can perform arithmetic operations on Torch tensors
````python
x = torch.tensor(2.,requires_grad=True)
w = torch.tensor(3.,requires_grad=True)
b = torch.tensor(7.,requires_grad=True)
y = w*x + b
````
If you print y, you can see that it is a tensor with one entry = 13. We set ````requires_grad=True```` for making the tensors utilizable for certain functions. For example, we can perform Backprop as follows       
````python
y.backward()
print('dy/dx:', x.grad)
print('dy/dw:', w.grad)
print('dy/db:', b.grad)
````
We obtain derivatives of y w.r.t x,w and b as 3,2 and 1. If we had set ````requires_grad=False````, these would print out as ````None````. Default is ````requires_grad=False````     
##### Some Torch Functions
````python
t2 = torch.ones(3,4)
````
t2 will be a 3x4 Tensor with all entries as "1" and dtype is "float32" by default
````python
t3 = torch.full((3,4),7.384)
````
t3 will be a 2x5 Tensor with all entries as "7.384"
````python
t4 = torch.cat((t2,t3))
````
t4 is the concatenation t2 and t3
````python
t5 = t2.reshape(3,2,2)
````
t5 is reshaped version of t2 with size 3x2x2
### Numpy & Torch
````python
import numpy as np
````
For creating a Numpy Array of size 2x3
````python
n1 = np.array([[1, 2, 3], [3., 4, 5]])
````
Note that dtype is float due to the presence of a one "3.". This is because numpy array has uniformity too.            
Converting Numpy array to Torch Tensor
````python
t6 = torch.from_numpy(n1)
````
Converting Torch tensor to Numpy array
````python
n2 = t6.numpy()
````
#### Why PyTorch over Numpy?
* **Autograd**: The ability to automatically compute gradients for tensor operations is essential for training deep learning models.            
* **GPU Support**: While working with massive datasets and large models, PyTorch tensor operations can be performed efficiently using a Graphics Processing Unit (GPU). Computations that might typically take hours can be completed within minutes using GPUs.

### Linear Regression - PyTorch
````
Y = W*X + B
````
Y = Output, X = Input, W = Weights and B = Bias. Mathematically, Y = X x W Transpose + B.       
Let us consider the following input dataset
| Rainfall (X1) | Temperature (X2) | Yield (Y) |
|      :-:      |      :-:         |      :-:  |
| 67 | 73 | 56 |
| 88 | 91 | 81 |
| 134 | 87 | 119 |
| 43 | 102 | 22 |
| 96 | 69 | 103 |
So our input array X is
````python
X = np.array([[67,73],
              [88,91],
              [134,87],
              [43,102],
              [96,69]],dtype='float32')
````
Our actual output vector Y is
````python
Y = np.array([56,81,119,22,103],dtype='float32')
````
Convert these to Torch Tensors
````python
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)
````
Now, we need to obtain the weights W and biases B, of the best fit to this dataset. Initialize W and B with random values.
````python
W = torch.randn(1,2,requires_grad=True)   # Because W Transpose should be 2x1
B = torch.randn(5,1,requires_grad=True)
````
Define the model
````python
def model(X):
    return X @ W.t() + B
````
**@** represents matrix multiplication in PyTorch and **.t()** returns Transpose of a tensor
Let our predictions be ````Y_````
````python
Y_ = model(X)
````
Define the Loss Function. Considering MSE Loss
````python
def MSE(y,h):
    diff = h - y
    return torch.sum(diff**2)/diff.numel()
````
**torch.sum()** returns sum of elements of a tensor and **.numel()** returns the number of elements in a tensor.                  
Compute the Loss for our model
````python
Loss = MSE(Y,Y_)
````
Back Propogation
````python
Loss.backward()
````
Gradients for weights W and biases B
````python
W.grad     # Same Size as W
B.grad
````
Updating our weights W. Consider a learning rate of "a"
````python
with torch.no_grad():
    W -= a*(W.grad)
    B -= a*(B.grad) 
````
We use ````torch.no_grad()```` to indicate to PyTorch that we shouldn't track, calculate, or modify gradients while updating the weights and biases.        
Now, reset the gradients to zero. We need to do this because PyTorch accumulates gradients. Otherwise, the next time we invoke backprop on the loss, the new gradient values are added to the existing gradients, which may lead to unexpected results.
````python
W.grad.zero_()
B.grad.zero_()
````
Repeat the steps until the loss is significantly minimized. For N Epochs:
````python
for i in range(N):
    Y_ = model(X)
    Loss = MSE(Y, Y_)
    Loss.backward()
    with torch.no_grad():
        W -= W.grad * a
        B -= B.grad * a
        W.grad.zero_()
        B.grad.zero_()
````
Compute the Loss now and verify that it is lower. Compare the actual outputs and predicted outputs.          
###### Note: You will obtain a high loss. For a better fit, we will need a larger dataset. 
