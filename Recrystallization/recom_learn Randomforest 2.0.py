import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib
import pubchempy as pcp

MODEL_FILE = 'solvent_model.pkl'
ENCODER_FILE = 'solvent_encoder.pkl'
TOP_K = 3  # 推荐 Top K 个溶剂

class RecrystallizationRecommender(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("重结晶溶剂预测系统（改进版）")
        self.geometry("800x700")

        # 加载并校验数据库
        df = self._load_database()
        if df is not None:
            cols = ['name', 'cas', 'solvent_1', 'solvent_2', 'solvent_3', 'solvent_4', 'solvent_5', 'SMILES']
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

        # 每次启动都重新训练模型
        if self.df is not None:
            self._train_model()

        # 创建界面
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
        self.txt = scrolledtext.ScrolledText(res, width=80, height=35)
        self.txt.grid(sticky='nsew', padx=5, pady=5)

        self.status_var = tk.StringVar(value=self.status)
        ttk.Label(main, textvariable=self.status_var, relief='sunken')\
            .grid(row=2, column=0, sticky='ew', pady=5)

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
        # 准备多标签目标
        records = self.df.dropna(subset=['SMILES']).reset_index(drop=True)
        X = self._featurize(records['SMILES'])
        y_labels = []
        for _, row in records.iterrows():
            sols = [row[f'solvent_{i}'] for i in range(1,6) if pd.notna(row[f'solvent_{i}'])]
            y_labels.append(sols)

        self.mlb = MultiLabelBinarizer()
        Y = self.mlb.fit_transform(y_labels)

        # 分割数据并设置随机种子
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.1, random_state=42
        )

        # 网格搜索调参
        base_rf = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
        grid = GridSearchCV(base_rf, param_grid, cv=3)
        grid.fit(X_train, y_train)
        best_rf = grid.best_estimator_

        # 动态确定校准折数
        pos_counts = np.sum(y_train, axis=0)
        min_count = np.min(pos_counts[pos_counts > 0]) if np.any(pos_counts > 0) else 0
        cv_folds = min(3, min_count) if min_count >= 2 else 0

        if cv_folds >= 2:
            calibrated = CalibratedClassifierCV(best_rf, cv=cv_folds, method='isotonic')
            classifier = calibrated
        else:
            classifier = best_rf  # 样本过少，跳过校准

        # 不使用并行以避免 pickle 错误
        self.clf = OneVsRestClassifier(classifier)
        self.clf.fit(X_train, y_train)

        # 评估并保存
        score = self.clf.score(X_test, y_test)
        joblib.dump(self.clf, MODEL_FILE)
        joblib.dump(self.mlb, ENCODER_FILE)
        self.status = f"模型训练完成（测试集准确率：{score*100:.1f}%）"

    def _name_to_smiles(self, name: str) -> str:
        try:
            comps = pcp.get_compounds(name, 'name')
            if comps:
                return comps[0].canonical_smiles
        except:
            pass
        return None

    def predict(self):
        self.txt.delete('1.0', tk.END)
        inp = self.search_var.get().strip()
        if not inp:
            self.status_var.set("请先输入化合物名称或 SMILES")
            return
        if self.df is None:
            self.status_var.set("无可用数据库")
            return

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

        x = self._featurize([smi])
        probs = self.clf.predict_proba(x)[0]
        top_idx = np.argsort(probs)[::-1][:TOP_K]

        self.txt.insert(tk.END, f"输入 SMILES：{smi}\n推荐 Top {TOP_K} 溶剂及置信度：\n")
        for rank, idx in enumerate(top_idx, 1):
            sol = self.mlb.classes_[idx]
            conf = probs[idx] * 100
            self.txt.insert(tk.END, f"  {rank}. {sol} （{conf:.1f}%）\n")

        self.txt.insert(tk.END, "\n预测原理简述：\n"
                            "1. 多标签学习：同时考虑所有常用溶剂，不局限于首选一项；\n"
                            "2. 特征提取：2048位 Morgan 指纹捕捉分子结构细节；\n"
                            "3. 网格搜索+动态校准：根据每个溶剂样本量动态决定校准折数；\n"
                            "4. One-vs-Rest：为每个溶剂分别训练决策器，输出真正的概率分布。\n")
        self.status_var.set("预测完成")

if __name__ == "__main__":
    app = RecrystallizationRecommender()
    app.mainloop()
