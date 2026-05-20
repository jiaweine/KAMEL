"""
TabPFN Offline Classifier
离线TabPFN实现，不依赖HuggingFace，基于本地简化版本
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class SimpleTransformerEncoder(nn.Module):
    """简化的Transformer编码器"""
    
    def __init__(self, input_dim, hidden_dim=64, n_heads=4, n_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1000, hidden_dim))
        
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 输入投影
        x = self.input_projection(x)  # [batch, seq, hidden]
        
        # 添加位置编码
        if seq_len <= 1000:
            pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
            x = x + pos_enc
        
        # Transformer编码
        x = self.transformer(x)
        
        return x

class TabPFNOfflineModel(nn.Module):
    """离线TabPFN模型"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # 特征编码器
        self.feature_encoder = SimpleTransformerEncoder(
            input_dim=1,  # 每个特征单独处理
            hidden_dim=hidden_dim,
            n_heads=4,
            n_layers=2
        )
        
        # 聚合层
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim * input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 输出层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 处理每个特征 [batch, features] -> [batch, features, 1]
        x = x.unsqueeze(-1)
        
        # 对每个特征使用Transformer编码
        encoded_features = []
        for i in range(self.input_dim):
            feature = x[:, i:i+1, :]  # [batch, 1, 1]
            encoded = self.feature_encoder(feature)  # [batch, 1, hidden]
            encoded_features.append(encoded.squeeze(1))  # [batch, hidden]
        
        # 拼接所有特征
        x = torch.cat(encoded_features, dim=1)  # [batch, input_dim * hidden]
        
        # 聚合
        x = self.aggregator(x)  # [batch, hidden]
        
        # 分类
        x = self.classifier(x)  # [batch, output_dim]
        
        return x

class TabPFNOfflineClassifier_ML(BaseEstimator, ClassifierMixin):
    """
    离线TabPFN分类器 - 完全本地实现
    不依赖HuggingFace或在线模型下载
    """
    
    def __init__(self, **kwargs):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=min(50, kwargs.get('max_features', 50)))  # 限制特征数
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型参数
        self.hidden_dim = kwargs.get('hidden_dim', 64)
        self.n_epochs = kwargs.get('n_epochs', 50)
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.batch_size = kwargs.get('batch_size', 32)
        self.patience = kwargs.get('patience', 10)
        self.kwargs = kwargs
        
    def _prepare_data(self, X, y=None):
        """数据预处理"""
        if y is not None:
            # 特征选择
            X_selected = self.feature_selector.fit_transform(X, y)
            # 标准化
            X_scaled = self.feature_scaler.fit_transform(X_selected)
        else:
            # 测试时使用已fitted的transformer
            X_selected = self.feature_selector.transform(X)
            X_scaled = self.feature_scaler.transform(X_selected)
            
        return X_scaled
    
    def train(self, X_train, y_train, X_val, y_val):
        """训练模型"""
        
        # 数据预处理
        X_train_proc = self._prepare_data(X_train, y_train)
        X_val_proc = self._prepare_data(X_val)
        
        # 标签编码
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_val_enc = self.label_encoder.transform(y_val)
        
        n_features = X_train_proc.shape[1]
        n_classes = len(np.unique(y_train_enc))
        
        # 初始化模型
        self.model = TabPFNOfflineModel(
            input_dim=n_features,
            output_dim=n_classes,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # 转换为tensor
        X_train_tensor = torch.FloatTensor(X_train_proc).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_enc).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_proc).to(self.device)
        y_val_tensor = torch.LongTensor(y_val_enc).to(self.device)
        
        # 训练设置
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            # 训练模式
            self.model.train()
            
            # 批次训练
            n_batches = (len(X_train_tensor) + self.batch_size - 1) // self.batch_size
            total_loss = 0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(X_train_tensor))
                
                batch_X = X_train_tensor[start_idx:end_idx]
                batch_y = y_train_tensor[start_idx:end_idx]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # 验证
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        # 验证集预测
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_val_tensor)
            val_proba = F.softmax(val_outputs, dim=1).cpu().numpy()
        
        # 处理二分类概率输出
        if n_classes == 2 and len(val_proba.shape) == 2 and val_proba.shape[1] == 2:
            val_proba = val_proba[:, 1]
        
        return {
            'val_proba': val_proba,
            'training_samples': len(y_train)
        }
    
    def predict(self, X_test):
        """预测"""
        X_test_proc = self._prepare_data(X_test)
        X_test_tensor = torch.FloatTensor(X_test_proc).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            proba = F.softmax(outputs, dim=1).cpu().numpy()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # 反向标签编码
        predictions = self.label_encoder.inverse_transform(predictions)
        
        # 处理二分类概率输出
        n_classes = len(self.label_encoder.classes_)
        if n_classes == 2 and len(proba.shape) == 2 and proba.shape[1] == 2:
            test_proba = proba[:, 1]
        else:
            test_proba = proba
        
        return predictions, test_proba
    
    def get_model_info(self):
        return {
            'algorithm': 'TabPFN Offline (Local Implementation)',
            'hidden_dim': self.hidden_dim,
            'n_epochs': self.n_epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'device': str(self.device),
            'preprocessing': 'StandardScaler + SelectKBest',
            'hyperparameters': self.kwargs
        }

# 导出类
AVAILABLE = True
