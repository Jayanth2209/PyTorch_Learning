# PyTorch_Learning
Learning PyTorch for DNNs
## Install PyTorch
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
A Tensor is a number, vector, matrix, or any n-dimensional array. A tensor has an additional property of uniformity in the data type of all entries and that they are always of a proper shape i.e., the lenghts of rows are all same, and the same goes with columns.        
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
