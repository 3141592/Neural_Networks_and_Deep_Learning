import numpy as np

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

    print(arr3.shape)  # (2, 2, 2)
    print(arr3.ndim)   # 3  (3D array)

except Exception as e:
    print("")

arr = np.array([
    [[1, 2], 
     [3, 4]],
    [[5, 6, 7], 
     [7, 8, 9]]
], dtype=object)  # Explicitly forcing object dtype

print(arr.shape)  # (2, 2)
print(arr.ndim)   # 2  (2D array)


