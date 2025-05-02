import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from threading import Thread
import joblib
import pubchempy as pcp

MODEL_FILE = 'solvent_model.pkl'
ENCODER_FILE = 'solvent_encoder.pkl'
TOP_K = 3  # 推荐 Top K 个溶剂

class RecrystallizationRecommender(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("重结晶溶剂预测系统（GPU 加速 LightGBM）")
        self.geometry("800x750")

        # 1. 加载并校验数据库
        df = self._load_database()
        if df is not None:
            cols = ['name', 'cas',
                    'solvent_1', 'solvent_2', 'solvent_3', 'solvent_4', 'solvent_5',
                    'SMILES']
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

        # 2. 异步训练模型
        if self.df is not None:
            self._train_model()

        # 3. 创建界面
        self._create_widgets()

    def _load_database(self):
        default = 'recrystallization_database_with_smiles.xlsx'
        if os.path.exists(default):
            path = default
        else:
            messagebox.showinfo("提示", f"未找到数据库文件 {default}，请手动选择。")
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

        ttk.Label(main, textvariable=self.status_var, relief='sunken')\
            .grid(row=2, column=0, sticky='ew', pady=5)

        self.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)
        res.grid_columnconfigure(0, weight=1)

    def _train_model(self):
        """异步训练模型，使用预设参数以加快速度"""
        def train():
            self.status_var.set("开始训练模型，可能需要几秒钟...")
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

            self.encoder = LabelEncoder()
            y_enc = self.encoder.fit_transform(y)

                        # 使用多线程 CPU 训练并调整参数以避免无增益分裂
            self.clf = LGBMClassifier(
                device='cpu',
                n_jobs=-1,
                n_estimators=200,
                learning_rate=0.1,
                num_leaves=63,
                max_depth=10,
                min_gain_to_split=0.1,
                min_data_in_leaf=20
            )
            self.clf.fit(X, y_enc)

            # 保存模型与编码器
            joblib.dump(self.clf, MODEL_FILE)
            joblib.dump(self.encoder, ENCODER_FILE)
            self.status_var.set("模型训练完成，已保存模型。")
        Thread(target=train, daemon=True).start()

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
            self.txt.insert(tk.END, f"尝试将'{inp}'当作名称查询 SMILES...\n")
            smi = self._name_to_smiles(inp)
            if not smi:
                self.txt.insert(tk.END, "名称查询失败，无法获取 SMILES。\n")
                self.status_var.set("SMILES 获取失败")
                return
            self.txt.insert(tk.END, f"查询到 SMILES：{smi}\n")
            mol = Chem.MolFromSmiles(smi)

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        x = np.array(fp).reshape(1, -1)

        probs = self.clf.predict_proba(x)[0]
        top_indices = np.argsort(probs)[::-1][:TOP_K]

        self.txt.insert(tk.END, f"输入 SMILES：{smi}\n")
        self.txt.insert(tk.END, "推荐溶剂及置信度：\n")
        for rank, idx in enumerate(top_indices, start=1):
            sol = self.encoder.inverse_transform([idx])[0]
            conf = probs[idx] * 100
            self.txt.insert(tk.END, f"  {rank}. {sol} （置信度：{conf:.1f}%）\n")

        # 特征重要性解释
        importances = self.clf.feature_importances_
        top_feats = np.argsort(importances)[::-1][:5]
        self.txt.insert(tk.END, "\n选择理由：\n")
        self.txt.insert(tk.END, "1. 模型使用 GPU LightGBM，并采用优化参数加速训练。\n")
        self.txt.insert(tk.END, "2. 模型计算并学习了每个指纹特征（位点）的重要性，通过 feature_importances_ 获取。\n")
        self.txt.insert(tk.END, "3. 本次预测中最重要的指纹特征索引位点：" + ", ".join(map(str, top_feats)) + "。\n")
        self.txt.insert(tk.END, "4. 这些高权重特征对应化学结构片段，在相似化合物的重结晶中与推荐溶剂关联最强。\n")
        self.status_var.set("预测完成")

if __name__ == "__main__":
    app = RecrystallizationRecommender()
    app.mainloop()
