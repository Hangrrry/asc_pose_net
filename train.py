import torch
import torch.nn as nn
import torch.optim as optim
input_dim=9
# 定义神经网络
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionNet(nn.Module):
    def __init__(self):
        super(PositionNet, self).__init__()
        # 输入是6个特征（相机位置 x, y, z，像素位置 x, y，速度方向信息）
        self.fc1 = nn.Linear(input_dim, 64)  # 输入6个特征，输出64个特征
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)  # 输出二维平面位置 x, y

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # 最后一层不加激活函数，直接输出 x, y 位置
        return x

# 创建模型实例
model = PositionNet()

# 损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)


############################################


from sklearn.model_selection import KFold
import torch
from data import process_file
import numpy as np

best_val_loss = float('inf')
best_model_state = None

file='data222/dataset34.txt'
X_train_val,y_train_val=process_file(file)
X_train_val=np.array(X_train_val)
y_train_val=np.array(y_train_val)
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5折交叉验证

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
    print(f'Fold {fold + 1}')
    
    # 获取训练和验证数据
    X_train_fold, X_val_fold = X_train_val[train_idx], X_train_val[val_idx]
    y_train_fold, y_val_fold = y_train_val[train_idx], y_train_val[val_idx]
    
    # 转换为Tensor（如果使用PyTorch）
    X_train_fold = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
    X_val_fold = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
    y_train_fold = torch.tensor(y_train_fold, dtype=torch.float32).to(device)
    y_val_fold = torch.tensor(y_val_fold, dtype=torch.float32).to(device)
    
    # 初始化模型
    model = PositionNet().to(device)
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs =10000
    for epoch in range(num_epochs):
        model.train()  # 训练模式
        outputs = model(X_train_fold)
        loss = criterion(outputs, y_train_fold)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证模型
        model.eval()  # 验证模式
        with torch.no_grad():
            val_outputs = model(X_val_fold)
            val_loss = criterion(val_outputs, y_val_fold)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_model_state = model.state_dict()

    print(f'Fold {fold + 1} completed with Val Loss: {val_loss.item():.4f}')

torch.save(model.state_dict(), 'model_last.pth')
torch.save(best_model_state, 'model_best.pth')
