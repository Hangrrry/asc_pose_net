# 定义一个函数来处理文件
def process_file(filename):
    x = []  # 用于存储每一行的前两个数字
    y = []  # 用于存储每一行剩余的数字

    # 打开文件并逐行读取
    with open(filename, 'r') as file:
        for line in file:
            # 去除行尾的换行符，并以空格分割每一行
            numbers = line.strip().split()
            # 将前两个数字添加到列表x中
            y.append([float(numbers[0]), float(numbers[1])])
        # 将剩余的数字添加到列表y中
            x.append([float(num) for num in numbers[2:]])
    
    return x,y

