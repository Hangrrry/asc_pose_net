import torch
import torch.nn as nn
from data import process_file
# 定义与训练时相同的模型结构
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionNet(nn.Module):
    def __init__(self):
        super(PositionNet, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 创建模型实例
model = PositionNet().to(device)

# 加载保存的模型参数
model.load_state_dict(torch.load('model_last.pth'))
criterion = nn.MSELoss() 
# 将模型设置为评估模式
model.eval()
print("模型已加载并准备好进行预测")
x_,y_=process_file("dataddd/dataset45.txt")############################################
x= torch.tensor(x_, dtype=torch.float32).to(device)
y= torch.tensor(y_, dtype=torch.float32).to(device)
with open('cleaned.txt', 'w') as file:
        
    for i in range(len(x)):
        out=model(x[i])
        val_loss = criterion(out, y[i])
        print(out,f'loss:{val_loss.item():.4f}')
        if val_loss.item() <100:
            for j in y_[i]:
                file.write(str(j)+' ')
            for j in x_[i]:
                file.write(str(j)+' ')
            file.write('\n')
            
