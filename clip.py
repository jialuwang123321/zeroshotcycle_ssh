import numpy as np

a = np.ones((100, 6))  # 创建全为 1 的数组
a *= np.arange(0, 100)[:, np.newaxis]  # 按行乘以不同的数字


print('a = ',a)


rows = np.arange(0, 100, 5)  # 定义需要取出的行索引
selected_rows = a[rows,:]  # 使用切片操作取出指定行的元素
selected_rows_2 = a[rows,:] 
print('selected_rows==selected_rows_2',selected_rows.shape==selected_rows_2.shape)

