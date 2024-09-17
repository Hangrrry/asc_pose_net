import re

# 读取txt文件
with open('output6.txt', 'r') as file:
    lines = file.readlines()

# 定义正则表达式，匹配所有浮点数和整数
number_pattern = r'-?\d+\.?\d*'

# 存储结果
result = []

for line in lines:
    # 提取行中的所有数字
    numbers = re.findall(number_pattern, line)
    if numbers:
        # 将数字按空格拼接
        result.append(" ".join(numbers))

# 将结果写入新的txt文件
with open('output9.txt', 'w') as outfile:
    for item in result:
        outfile.write(item + '\n')

print("提取结果已保存至 output9.txt 文件。")
