import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# 加载增强版数据库（示例结构）
def load_enhanced_database(file_path='recrystallization_enhanced.csv'):
    """
    包含更多特征的示例数据结构：
    name | solubility_water | solubility_ethanol | ... | solvent_1 | solvent_2 | ...
    """
    return pd.read_csv(file_path)

# 特征工程
def create_features(df):
    # 文本特征（使用SMILES字符串）
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
    smiles_features = vectorizer.fit_transform(df['smiles'])
    
    # 数值特征标准化
    numeric_features = df[['molecular_weight', 'logP']].values
    numeric_features = (numeric_features - numeric_features.mean(0)) / numeric_features.std(0)
    
    # 组合特征
    return np.hstack([smiles_features.toarray(), numeric_features])

# 机器学习推荐模型
class SolventRecommender:
    def __init__(self, n_neighbors=5):
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        
    def train(self, features):
        self.model.fit(features)
        
    def recommend(self, query_features, df, top_n=3):
        distances, indices = self.model.kneighbors(query_features)
        
        recommendations = []
        for idx_list in indices:
            # 统计邻近样本的溶剂出现频率
            solvents = df.iloc[idx_list].filter(like='solvent').values.flatten()
            solvents = [s for s in solvents if pd.notna(s)]
            freq = pd.Series(solvents).value_counts()
            recommendations.append(freq.head(top_n).index.tolist())
            
        return recommendations

# 改进的推荐流程
def ml_recommendation():
    # 加载数据
    df = load_enhanced_database()
    
    # 创建特征矩阵
    features = create_features(df)
    
    # 训练模型
    recommender = SolventRecommender()
    recommender.train(features)
    
    # 用户交互
    while True:
        query = input("\nEnter chemical name or SMILES string: ")
        
        # 处理查询输入
        if query in df['name'].values:
            query_features = features[df[df['name']==query].index[0]]
        else:
            # 使用查询作为SMILES输入
            try:
                query_features = create_features(pd.DataFrame([{'smiles': query}]))
            except:
                print("Invalid input format")
                continue
                
        # 生成推荐
        recommendations = recommender.recommend(query_features.reshape(1, -1), df)
        
        print("\nRecommended solvents:")
        for i, solvent in enumerate(recommendations[0], 1):
            print(f"{i}. {solvent}")

if __name__ == "__main__":
    print("ML-Powered Recrystallization Recommender")
    ml_recommendation()