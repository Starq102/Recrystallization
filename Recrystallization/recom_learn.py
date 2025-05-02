import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import joblib
import pubchempy as pcp

MODEL_FILE = 'solvent_model.pkl'
ENCODER_FILE = 'solvent_encoder.pkl'
TOP_K = 3  # 推荐 Top K 个溶剂
CV_FOLDS = 5  # 交叉验证折数

class RecrystallizationRecommender(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("重结晶溶剂预测系统")
        self.geometry("800x750")

        # 1. 加载并校验数据库
        df = self._load_database()
        if df is not None:
            cols = ['name', 'cas',
                    'solvent_1', 'solvent_2', 'solvent_3', 'solvent_4', 'solvent_5',
                    'SMILES']
            try:
                self.df = df[cols].copy()
                self.status = "数据库加载成功"
            except KeyError:
                messagebox.showerror("错误", "Excel 列名不匹配，请检查是否含有：\n" + "，".join(cols))
                self.df = None
                self.status = "数据库格式错误"
        else:
            self.df = None
            self.status = "未加载到数据库"

        # 2. 每次启动都重新训练模型（包含超参数调优和交叉验证评估）
        if self.df is not None:
            self._train_model()

        # 3. 创建界面
        self._create_widgets()

    def _load_database(self):
        default = 'recrystallization_database_with_smiles.xlsx'
        if os.path.exists(default):
            path = default
        else:
            messagebox.showinfo("提示", f"当前目录未找到 {default}，请手动选择数据库文件")
            path = filedialog.askopenfilename(
                title="选择数据库文件",
                filetypes=[("Excel 文件", "*.xlsx *.xls")]
            )
            if not path:
                return None
        try:
            return pd.read_excel(path, engine='openpyxl')
        except Exception as e:
            messagebox.showerror("错误", f"读取文件失败：{e}")
            return None

    def _create_widgets(self):
        main = ttk.Frame(self, padding=10)
        main.grid(sticky='nsew')

        frm = ttk.LabelFrame(main, text="输入化合物名称或 SMILES 进行预测", padding=5)
        frm.grid(row=0, column=0, sticky='ew', pady=5)
        self.search_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.search_var, width=60).grid(row=0, column=0, padx=5)
        ttk.Button(frm, text="预测溶剂", command=self.predict).grid(row=0, column=1, padx=5)

        res = ttk.LabelFrame(main, text="预测结果与原因", padding=5)
        res.grid(row=1, column=0, sticky='nsew', pady=5)
        self.txt = scrolledtext.ScrolledText(res, width=80, height=40)
        self.txt.grid(sticky='nsew', padx=5, pady=5)

        self.status_var = tk.StringVar(value=self.status)
        ttk.Label(main, textvariable=self.status_var, relief='sunken')\
            .grid(row=2, column=0, sticky='ew', pady=5)

        # 布局伸缩
        self.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)
        res.grid_columnconfigure(0, weight=1)

    def _train_model(self):
        """重新训练模型，并通过 GridSearchCV 调参及交叉验证评估"""
        df0 = self.df.dropna(subset=['SMILES', 'solvent_1']).reset_index(drop=True)
        fps = []
        for smi in df0['SMILES']:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(np.array(fp))
            else:
                fps.append(np.zeros(2048, dtype=int))
        X = np.vstack(fps)
        y = df0['solvent_1'].values

        # 标签编码
        self.encoder = LabelEncoder()
        y_enc = self.encoder.fit_transform(y)

        # 划分训练/测试，用于最终评估
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.1)

        # 参数网格
        gbc = GradientBoostingClassifier()
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5]
        }
        grid = GridSearchCV(gbc, param_grid, cv=CV_FOLDS, n_jobs=-1, scoring='accuracy')
        grid.fit(X_train, y_train)

        # 最佳模型&交叉验证分数
        self.clf = grid.best_estimator_
        best_params = grid.best_params_
        cv_scores = cross_val_score(self.clf, X_train, y_train, cv=CV_FOLDS)
        cv_mean = cv_scores.mean() * 100
        test_acc = self.clf.score(X_test, y_test) * 100

        # 保存模型与编码器
        joblib.dump(self.clf, MODEL_FILE)
        joblib.dump(self.encoder, ENCODER_FILE)
        self.status = (
            f"模型训练完毕 | 测试准确率：{test_acc:.1f}% | CV 平均准确率：{cv_mean:.1f}%\n"
            f"最佳参数：{best_params}"
        )

    def _name_to_smiles(self, name: str) -> str:
        """PubChem 查询英文名对应的 SMILES"""
        try:
            comps = pcp.get_compounds(name, 'name')
            if comps:
                return comps[0].canonical_smiles
        except:
            pass
        return None

    def predict(self):
        """输入 SMILES 或名称→解析→生成指纹→预测→显示 Top K 结果与原因"""
        self.txt.delete('1.0', tk.END)
        inp = self.search_var.get().strip()
        if not inp:
            self.status_var.set("请先输入化合物名称或 SMILES")
            return
        if self.df is None:
            self.status_var.set("无可用数据库")
            return

        # 解析 SMILES 或 英文名
        mol = Chem.MolFromSmiles(inp)
        smi = inp
        if mol is None:
            self.txt.insert(tk.END, f"尝试将“{inp}”当作英文名称查询 SMILES...\n")
            smi = self._name_to_smiles(inp)
            if smi is None:
                self.txt.insert(tk.END, "名称查询失败，无法获取 SMILES。\n")
                self.status_var.set("SMILES 获取失败")
                return
            self.txt.insert(tk.END, f"查询到 SMILES：{smi}\n")
            mol = Chem.MolFromSmiles(smi)

        # 生成指纹
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        x = np.array(fp).reshape(1, -1)

        # 预测概率分布
        probs = self.clf.predict_proba(x)[0]
        top_indices = np.argsort(probs)[::-1][:TOP_K]

        # 显示 Top K 结果
        self.txt.insert(tk.END, f"输入 SMILES：{smi}\n")
        self.txt.insert(tk.END, "推荐溶剂及置信度：\n")
        for rank, idx in enumerate(top_indices, start=1):
            sol = self.encoder.inverse_transform([idx])[0]
            confidence = probs[idx] * 100
            self.txt.insert(tk.END, f"  {rank}. {sol} （置信度：{confidence:.1f}%）\n")
        self.txt.insert(tk.END, "\n选择理由：\n"
                                    "1. 使用 Gradient Boosting 对多种参数进行 5 折交叉验证，以提高泛化能力。\n"
                                    "2. Morgan 指纹捕捉分子极性、氢键、芳香性等特征，供模型学习。\n"
                                    "3. Top-K 溶剂按概率排序，置信度越高说明模型越确信该溶剂适合重结晶。\n"
                                    "4. 基于历史数据中相似结构化合物的实验表现统计。\n")
        self.status_var.set("预测完成")

if __name__ == "__main__":
    app = RecrystallizationRecommender()
    app.mainloop()
