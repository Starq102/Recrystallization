import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from threading import Thread
import joblib
import pubchempy as pcp

MODEL_FILE = 'solvent_models.pkl'
ENCODER_FILE = 'solvent_encoder.pkl'
TOP_K = 3  # 推荐 Top K 个溶剂

class MultiLabelCalibratedClassifier:
    """
    简单封装：针对每个标签保持一个校准后的二分类器列表，使用统一接口
    """
    def __init__(self, classifiers, mlb):
        self.classifiers = classifiers
        self.mlb = mlb

    def predict_proba(self, X):
        # 返回 shape (n_samples, n_labels)
        probs = [clf.predict_proba(X)[:, 1] for clf in self.classifiers]
        return np.vstack(probs).T

    def predict(self, X, thresh=0.5):
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

class RecrystallizationRecommender(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("重结晶溶剂预测系统（LightGBM 多标签校准版）")
        self.geometry("800x750")

        df = self._load_database()
        if df is not None:
            cols = ['name', 'cas'] + [f'solvent_{i}' for i in range(1,6)] + ['SMILES']
            try:
                self.df = df[cols].copy()
                self.status_var = tk.StringVar(value="数据库加载成功")
            except KeyError:
                messagebox.showerror("错误", "Excel 列名不匹配，请检查列名：\n" + ",".join(cols))
                self.df = None
                self.status_var = tk.StringVar(value="数据库格式错误")
        else:
            self.df = None
            self.status_var = tk.StringVar(value="未加载到数据库")

        if self.df is not None:
            self._train_model()

        self._create_widgets()

    def _load_database(self):
        default = 'recrystallization_database_with_smiles.xlsx'
        if os.path.exists(default):
            path = default
        else:
            messagebox.showinfo("提示", f"未找到 {default}，请选择文件。")
            path = filedialog.askopenfilename(title="选择数据库文件", filetypes=[("Excel", "*.xlsx *.xls")])
            if not path:
                return None
        try:
            return pd.read_excel(path, engine='openpyxl')
        except Exception as e:
            messagebox.showerror("错误", f"读取失败：{e}")
            return None

    def _create_widgets(self):
        main = ttk.Frame(self, padding=10)
        main.grid(sticky='nsew')

        frm = ttk.LabelFrame(main, text="输入化合物名称或 SMILES", padding=5)
        frm.grid(row=0, column=0, sticky='ew', pady=5)
        self.search_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self.search_var, width=60).grid(row=0, column=0, padx=5)
        ttk.Button(frm, text="预测溶剂", command=self.predict).grid(row=0, column=1, padx=5)

        res = ttk.LabelFrame(main, text="预测结果与原因", padding=5)
        res.grid(row=1, column=0, sticky='nsew', pady=5)
        self.txt = scrolledtext.ScrolledText(res, width=80, height=40)
        self.txt.grid(sticky='nsew', padx=5, pady=5)

        ttk.Label(main, textvariable=self.status_var, relief='sunken').grid(row=2, column=0, sticky='ew', pady=5)
        self.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)
        res.grid_columnconfigure(0, weight=1)

    def _featurize(self, smiles_list):
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(np.array(fp, dtype=int))
            else:
                fps.append(np.zeros(2048, dtype=int))
        return np.vstack(fps)

    def _train_model(self):
        def train():
            self.status_var.set("开始训练并校准模型...")
            rec = self.df.dropna(subset=['SMILES']).reset_index(drop=True)
            X = self._featurize(rec['SMILES'])
            labels = [[rec.at[i, f'solvent_{j}'] for j in range(1,6) if pd.notna(rec.at[i, f'solvent_{j}'])]
                      for i in rec.index]
            self.mlb = MultiLabelBinarizer()
            Y = self.mlb.fit_transform(labels)
            X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=0.2, random_state=42)

            base = LGBMClassifier(device='cpu', n_jobs=1)
            # 针对每个标签独立网格调参并校准
            calibrated = []
            for i in range(Y.shape[1]):
                y_i = y_tr[:, i]
                if np.sum(y_i) < 5:
                    # 样本太少，直接训练
                    model = LGBMClassifier(device='cpu', n_jobs=1)
                    model.fit(X_tr, y_i)
                else:
                    grid = GridSearchCV(LGBMClassifier(device='cpu', n_jobs=1),
                                        {'num_leaves': [31,63], 'max_depth': [10,20,-1], 'learning_rate': [0.05,0.1]},
                                        cv=3)
                    grid.fit(X_tr, y_i)
                    best = grid.best_estimator_
                    cal = CalibratedClassifierCV(best, cv=3, method='isotonic')
                    cal.fit(X_tr, y_i)
                    model = cal
                calibrated.append(model)

            # 保存多标签校准器和编码器
            self.multi_clf = MultiLabelCalibratedClassifier(calibrated, self.mlb)
            joblib.dump(self.multi_clf, MODEL_FILE)
            joblib.dump(self.mlb, ENCODER_FILE)

            # 评估
            y_pred = self.multi_clf.predict(X_te)
            acc = np.mean(np.all(y_pred == y_te, axis=1))
            self.status_var.set(f"训练完成，测试集多标签准确率：{acc*100:.1f}%")
        Thread(target=train, daemon=True).start()

    def _name_to_smiles(self, name: str) -> str:
        try:
            comps = pcp.get_compounds(name, 'name')
            return comps[0].canonical_smiles if comps else None
        except:
            return None

    def predict(self):
        if not hasattr(self, 'multi_clf'):
            self.status_var.set("模型尚未训练完成，请稍后再试。")
            return
        inp = self.search_var.get().strip()
        if not inp:
            self.status_var.set("请输入化合物名称或 SMILES")
            return
        mol = Chem.MolFromSmiles(inp)
        smi = inp
        if mol is None:
            smi = self._name_to_smiles(inp)
            if not smi:
                self.status_var.set("名称查询失败，无法获取 SMILES")
                return
        X = self._featurize([smi])
        probs = self.multi_clf.predict_proba(X)[0]
        top_idx = np.argsort(probs)[::-1][:TOP_K]

        self.txt.delete('1.0', tk.END)
        self.txt.insert(tk.END, f"输入 SMILES：{smi}\n推荐 Top {TOP_K} 溶剂及置信度：\n")
        for i, idx in enumerate(top_idx, 1):
            sol = self.mlb.classes_[idx]
            conf = probs[idx] * 100
            self.txt.insert(tk.END, f"  {i}. {sol} （{conf:.1f}%）\n")

        # 添加解释原因
        self.txt.insert(tk.END, "\n选择理由：\n")
        self.txt.insert(tk.END, "1. 每个溶剂由独立的二分类器预测其适用概率。\n")
        self.txt.insert(tk.END, "2. 对正样本充足的标签使用 Isotonic Regression 校准概率，使置信度更准确。\n")
        self.txt.insert(tk.END, "3. 基于 Morgan 指纹提取分子结构特征，捕捉分子的极性、芳香环等信息。\n")
        self.txt.insert(tk.END, "4. 采用多标签学习同时兼顾首选和常用备选溶剂，推荐更全面。\n")
        self.txt.insert(tk.END, "5. 对每个标签分别网格搜索优化 LightGBM 参数，提升模型性能。\n")
        self.status_var.set("预测完成")

if __name__ == "__main__":
    app = RecrystallizationRecommender()
    app.mainloop()
