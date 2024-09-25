# 定义要合并的文件路径
file1 = 'dataset.txt'
file2 = 'output9.txt'
output_file = 'dataset.txt'

# 读取第一个文件
with open(file1, 'r') as f1:
    data1 = f1.readlines()

# 读取第二个文件
with open(file2, 'r') as f2:
    data2 = f2.readlines()

# 将两个文件的内容合并并写入新文件
with open(output_file, 'w') as outfile:
    # 写入第一个文件的内容
    outfile.writelines(data1)
    # 换行以确保两个文件内容之间有间隔
    outfile.write('\n')
    # 写入第二个文件的内容
    outfile.writelines(data2)

print(f"文件已合并，输出到 {output_file}")
