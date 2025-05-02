import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
import numpy as np

# 定义神经网络模型（修正为二分类）
class SolventRecommender(nn.Module):
    def __init__(self, input_dim=2048):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出一个概率值
        )
    
    def forward(self, x):
        return self.model(x)

# 分子指纹生成（添加错误处理）
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
        return torch.tensor(fp, dtype=torch.float32)
    except:
        return None

# 数据预处理（添加数据清洗）
def prepare_data(file_path):
    # 加载数据
    df = pd.read_csv(file_path)
    
    # 生成指纹
    df['fingerprint'] = df['smiles'].apply(smiles_to_fingerprint)
    
    # 清洗无效数据
    df = df.dropna(subset=['fingerprint']).reset_index(drop=True)
    
    # 构建标签（使用最常见溶剂）
    all_solvents = df.filter(like='solvent').stack()
    target_solvent = all_solvents.value_counts().index[0]
    df['target'] = df.filter(like='solvent').apply(
        lambda x: target_solvent in x.values, axis=1
    ).astype(int)
    
    # 分割数据集
    return train_test_split(df, test_size=0.2, random_state=42)

# 训练流程（添加批量训练）
def train_model(train_data, input_dim=2048, epochs=10, batch_size=32):
    model = SolventRecommender(input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 转换为张量
    fingerprints = torch.stack(train_data['fingerprint'].tolist())
    labels = torch.tensor(train_data['target'].values, dtype=torch.float32)
    
    # 转换为DataLoader
    dataset = torch.utils.data.TensorDataset(fingerprints, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")
    
    return model

# 预测函数（修正概率计算）
def predict(model, smiles):
    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return "Invalid SMILES"
    
    model.eval()
    with torch.no_grad():
        output = model(fp.unsqueeze(0))
        prob = torch.sigmoid(output).item()
    return f"Recommendation probability: {prob:.2%}"

if __name__ == "__main__":
    # 数据准备
    try:
        train_df, test_df = prepare_data("chemical_data.csv")
        print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
        
        # 检查数据
        if len(train_df) == 0:
            raise ValueError("No valid training data available")
            
        # 训练模型
        model = train_model(train_df, epochs=20)
        
        # 交互测试
        while True:
            query = input("\nEnter SMILES string (q to quit): ").strip()
            if query.lower() == 'q':
                break
            print(predict(model, query))
            
    except FileNotFoundError:
        print("Error: chemical_data.csv file not found")
    except Exception as e:
        print(f"Error: {str(e)}")