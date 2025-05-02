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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from threading import Thread
import joblib
import pubchempy as pcp

MODEL_FILE = 'solvent_model.pkl'
ENCODER_FILE = 'solvent_encoder.pkl'
TOP_K = 3  # 推荐 Top K 个溶剂

class RecrystallizationRecommender(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("重结晶溶剂预测系统（多标签 LightGBM + 校准）")
        self.geometry("800x750")

        # 加载并校验数据库
        df = self._load_database()
        if df is not None:
            cols = ['name', 'cas', 'solvent_1', 'solvent_2', 'solvent_3', 'solvent_4', 'solvent_5', 'SMILES']
            try:
                self.df = df[cols].copy()
                self.status_var = tk.StringVar(value="数据库加载成功")
            except KeyError:
                messagebox.showerror("错误", "Excel 列名不匹配，请检查列名：\n" + "，".join(cols))
                self.df = None
                self.status_var = tk.StringVar(value="数据库格式错误")
        else:
            self.df = None
            self.status_var = tk.StringVar(value="未加载到数据库")

        # 异步训练模型
        if self.df is not None:
            self._train_model()

        # 创建界面
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
        def train():
            self.status_var.set("开始训练多标签模型...")
            records = self.df.dropna(subset=['SMILES']).reset_index(drop=True)
            X = self._featurize(records['SMILES'])
            y_labels = [
                [row[f'solvent_{i}'] for i in range(1, 6) if pd.notna(row[f'solvent_{i}'])]
                for _, row in records.iterrows()
            ]
            self.mlb = MultiLabelBinarizer()
            Y = self.mlb.fit_transform(y_labels)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

            base = LGBMClassifier(device='cpu', n_jobs=1)
            ovr = OneVsRestClassifier(base)
            param_grid = {
                'estimator__num_leaves': [31, 63],
                'estimator__max_depth': [10, 20, -1],
                'estimator__learning_rate': [0.05, 0.1]
            }
            grid = GridSearchCV(ovr, param_grid, cv=3)
            grid.fit(X_train, y_train)
            best = grid.best_estimator_

            # 对每个标签单独校准或训练
            pos_counts = np.sum(y_train, axis=0)
            calibrated_list = []
            for i in range(Y.shape[1]):
                y_i = y_train[:, i]
                model = base if pos_counts[i] < 3 else best.estimators_[i]
                if pos_counts[i] >= 3:
                    cal = CalibratedClassifierCV(model, cv=3, method='isotonic')
                    cal.fit(X_train, y_i)
                else:
                    clf_i = type(model)(**model.get_params())
                    clf_i.fit(X_train, y_i)
                    cal = clf_i
                calibrated_list.append(cal)

            self.calibrated_clfs = calibrated_list
            self.status_var.set("模型校准完成")
            Y_pred = np.column_stack([clf.predict(X_test) for clf in self.calibrated_clfs])
            acc = np.mean(np.all(Y_pred == y_test, axis=1))
            joblib.dump(self.calibrated_clfs, MODEL_FILE)
            joblib.dump(self.mlb, ENCODER_FILE)
            self.status_var.set(f"训练完成，多标签准确率：{acc*100:.1f}%")
        Thread(target=train, daemon=True).start()

    def _name_to_smiles(self, name: str) -> str:
        try:
            comps = pcp.get_compounds(name, 'name')
            return comps[0].canonical_smiles if comps else None
        except:
            return None

    def predict(self):
        if not hasattr(self, 'calibrated_clfs'):
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
        x = self._featurize([smi])
        probs = np.column_stack([clf.predict_proba(x)[:, 1] for clf in self.calibrated_clfs])[0]
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
        self.txt.insert(tk.END, "2. 概率校准（Isotonic Regression）让输出的置信度更可靠。\n")
        self.txt.insert(tk.END, "3. 模型基于 Morgan 指纹提取的结构特征，捕捉分子的极性、芳香环等化学信息。\n")
        self.txt.insert(tk.END, "4. 训练过程使用多标签学习，兼顾首选和常用备选溶剂，能更全面地推荐溶剂组合。\n")
        self.txt.insert(tk.END, "5. 后台网格搜索自动优化 LightGBM 参数，提升了模型的泛化性能。\n")
        self.status_var.set("预测完成")

if __name__ == "__main__":
    app = RecrystallizationRecommender()
    app.mainloop()
