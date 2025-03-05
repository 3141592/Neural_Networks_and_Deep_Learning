import numpy as np

import numpy as np

arr1 = np.array([1, 2, 3])  
print("arr1.shape: " + str(arr1.shape))  # (3,)
print("arr1.ndim: " + str(arr1.ndim))   # 1  (1D array)

arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2.shape)  # (2, 3)
print(arr2.ndim)   # 2  (2D array)

arr3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(arr3.shape)  # (2, 2, 2)
print(arr3.ndim)   # 3  (3D array)

arr3 = np.array([
    [[1, 2], 
     [3, 4]], 
    [[5, 6], 
     [7, 8]]
])
print(arr3.shape)  # (2, 2, 2)
print(arr3.ndim)   # 3  (3D array)

try:
    arr3 = np.array([
        [[1, 2], 
         [3, 4]], 
        [[5, 6, 7], 
         [7, 8, 9]]
    ])
except Exception as e:
    print(f"Case 1 Error: {e}")

print(arr3.shape)  # (2, 2, 2)
print(arr3.ndim)   # 3  (3D array)

arr = np.array([
    [[1, 2], 
     [3, 4]],
    [[5, 6, 7], 
     [7, 8, 9]]
])  # Explicitly forcing object dtype

