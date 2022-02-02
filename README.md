# Introduction to Numc:
Numc is the python package for some fundamental arithmetic computing of matrices, it greatly speeds up the matrix operations in comparison to some naïve implementations of matrix operations.

# Quick start for using `numc`:
To import `numc` and using the `numc` matrices, you should do as the traditional way for most python packages, e.g.  
```python
from numc import Matrix

import numc
numc.Matrix

import numc as nc
nc.Matrix
```
for demonstration purpose, we will be using `import numc as nc` throughout the documentation.  

Then we create a simplest matrix of 2 * 2 of all zeros by calling
```python
>>> nc.Matrix(3, 3) # This creates a 3 * 3 matrix with entries all zeros
[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
```
We can also assign the created matrix to a variable and do some operation with it later e.g.
```python
>>> mat = nc.Matrix(3, 3)
>>> mat + mat # This just adds up the matrix of all zeros with itself
[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
```

There are also other ways to declare a matrix, e.g.
```python
>>> nc.Matrix(3, 3, 1) # This creates a 3 * 3 matrix with entries all ones
[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
>>> nc.Matrix([[1, 2, 3], [4, 5, 6]]) # This creates a 2 * 3 matrix with first row 1, 2, 3, second row 4, 5, 6
[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
>>> nc.Matrix(1, 2, [4, 5]) # This creates a 1 * 2 matrix with entries 4, 5
[4.0, 5.0]
```
The thing to notice here is that for the last way to declare a matrix, the python list passed in has to fit with the size passed in, so in the example, the 1-d list with two elements inside align with the size 1 * 2.

That's the quick introduction to import `numc` and initialize a `numc.Matrix`, hope you enjoy playing with it!




# A Guide to Numc Documentation:
Numc has  
- an instance attribute
- six number methods
- two instance methods

    along with probably the most important functionality of a fast matrix operation package:

- indexing and changing the value of a matrix by indexing  

Here we will be documenting them thoroughly.

## `Instance Attributes`  
#### `.shape`      
Return the shape of the `numc.Matrix`     

**Example:**     
```python
>>> mat = nc.Matrix(3, 3)
>>> mat.shape
(3, 3)
>>> mat = nc.Matrix(3, 1)
>>> mat.shape
(3,)
```

<br>

## `Number Methods`
#### `a + b`  
Element-wise sum of `a` and `b`. Returns a `numc.Matrix` object.  

*Throwing errors*:     
- if `a` and `b` are not of the same type `numc.Matrix`
- if `a` and `b` do not have the same dimensions

**Example:**   
```python
>>> a = nc.Matrix([[1, 2], [3, 4]])
>>> b = nc.Matrix([[4, 3], [2, 1]])
>>> a + b
[[5.0, 5.0], [5.0, 5.0]]
```

<br>

#### `a - b`    
Element-wise subtraction of `a` and `b`. Returns a `numc.Matrix` object.  

*Throwing errors* :    
&nbsp; &nbsp; &nbsp; &nbsp; refer to `a + b`  

**Example:**  
```python
>>> a = nc.Matrix([[1, 2], [3, 4]])
>>> b = nc.Matrix([[4, 3], [2, 1]])
>>> a - b
[[-3.0, -1.0], [1.0, 3.0]]
```

<br>

#### `a * b`  
Matrix multiplication of `a` and `b`. Returns a `numc.Matrix` object.   
*Note*: this is a matrix multiplication, not an element-wise multiplication.  

*Throwing errors* :   
- if `a` and `b` are not of the same type `numc.Matrix`
- if `a`’s number of columns is not equal to `b`’s number of rows.

**Example:**  
```python
>>> a = nc.Matrix([[1, 2], [3, 4]])
>>> b = nc.Matrix([[4, 3], [2, 1]])
>>> a * b
[[8.0, 5.0], [20.0, 13.0]]
```

<br>

#### `-a`
Element-wise negation of `a`. Returns a `numc.Matrix` object.  

**Example:**  
```python
>>> a = nc.Matrix([[1, 2], [3, 4]])
>>> -a
[[-1.0, -2.0], [-3.0, -4.0]]
>>> b = nc.Matrix([[-1, -2], [3, 4]])
>>> -b
[[1.0, 2.0], [-3.0, -4.0]]
```

<br>

#### `abs(a)`
Element-wise absolute value of `a`. Returns a `numc.Matrix` object.  

**Example:**  
```python
>>> a = nc.Matrix([[1, 2], [3, 4]])
>>> abs(a)
[[1.0, 2.0], [3.0, 4.0]]
>>> b = nc.Matrix([[-1, -2], [3, 4]])
>>> abs(b)
[[1.0, 2.0], [3.0, 4.0]]
>>> c = nc.Matrix([[-1, -2], [-3, -4]])
[[1.0, 2.0], [3.0, 4.0]]
```

<br>

#### `a ** pow`  
Raise `a` to the `pow`th power. `a` to the `0`th power is the identity matrix. Returns a `numc.Matrix` object.   
*Note*: This operator is defined in terms of matrix multiplication, not element-wise multiplication.
> Parameters:
> - `a` : a square `numc.Matrix` of size n * n
> - `pow` : an `int` bigger than or equal to 0

**Example:**  
```python
>>> a = nc.Matrix([[1, 2], [3, 4]])
>>> a ** 3
[[37.0, 54.0], [81.0, 118.0]]
>>> b = nc.Matrix([[-1, -2], [3, 4]])
>>> b ** 3
[[-13.0, -14.0], [21.0, 22.0]]
```

<br>

## `Instance Methods`
#### `get(self, i, j)`
Returns the entry at the `i`th row and `j`th column.  
> Parameters:
> - `self` : The `numc.Matrix` object
> - `i` : the ith row of the matrix
> - `j` : the jth column of the matrix

*Throwing Errors*:
- if `i` or `j` is not an `int`
- if `i` or `j` is out of range

**Example:**  
```python
>>> a = nc.Matrix([[1, 2], [3, 4]])
>>> a.get(1, 1)
4.0
```

<br>

#### `set(self, i, j, val)`
Set `self`’s entry at the `i`th row and `j`th column to `val`.  
> Parameters:
> - `self` : The `numc.Matrix` object
> - `i` : the ith row of the matrix
> - `j` : the jth column of the matrix
> - `val` : the value to set at indices (i, j)

*Throwing Errors*:
- if `i` or `j` is not an `int`
- if `i` or `j` is out of range
- if `val` is not a `float` or `int`

**Example:**  
```python
>>> a = nc.Matrix([[1, 2], [3, 4]])
>>> a.set(1, 1, 5)
>>> a
[[1.0, 2.0], [3.0, 5.0]]
```

<br>

## `Indexing`
- You can either index into a matrix or change the value of one entry or slice
- For a 2D matrix, the key could either be an integer, a single slice, or a tuple of two slices/ints. For a 1D matrix, the key could either be an integer or a single slice.

Here is some example usage:  
```python
>>> a = nc.Matrix(3, 3)
>>> a[0] # Key is a single number
[0.0, 0.0, 0.0]
>>> a[0:2] # key is a single slice
[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
>>> a[0:2, 0:2] # key is a tuple of two slices
[[0.0, 0.0], [0.0, 0.0]]
>>> a[0:2, 0] # key is a tuple of (slice, int)
[0.0, 0.0]
>>> a[0, 0:2] # key is a tuple of (int, slice)
[0.0, 0.0]
>>> a[0, 0] # key is a tuple of (int, int)
0.0
>>> b = nc.Matrix(1, 3) # b is a 1D matrix
>>> b[0]
0.0
>>> b[0:2] # Number of rows/cols does not matter now. You are slicing it as if it were a list
[0.0, 0.0]
>>> b[0:1, 0:1] # This is invalid!
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
TypeError: 1D matrices only support single slice!
```

You can also index into the matrix to change the value of one entry or slice. The key could either be an integer, a single slice, or a tuple of two slices/ints.

*Note*: The value passes in should fit the dimensions of the slice, otherwise, an error would be thrown.  

Here is some example usage:  
```python
>>> a = nc.Matrix(3, 3)
>>> a[0:1, 0:1] = 0.0 # Resulting slice is 1 by 1
>>> a[:, 0] = [1, 1, 1] # Resulting slice is 1D
>>> a
[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
>>> a[0, :] = [2, 2, 2] # Resulting slice is 1D
>>> a
[[2.0, 2.0, 2.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
>>> a[0:2, 0:2] = [[1, 2], [3, 4]] # Resulting slice is 2D
>>> a
[[1.0, 2.0, 2.0], [3.0, 4.0, 0.0], [1.0, 0.0, 0.0]]
```  
*Note*: Slices share data with their original matrix. This means that by changing values of the slices, the values of the original matrices should also change.

**Example:**  
```python
>>> a = nc.Matrix(2, 2)
>>> a[0:1, 0:1] = 1.0
>>> a
[[1.0, 0.0], [0.0, 0.0]]
>>> a[1] = [2, 2]
>>> a
[[1.0, 0.0], [2.0, 2.0]]
>>> b = a[1]
>>> b[1] = 3
>>> a
[[1.0, 0.0], [2.0, 3.0]]
```
