import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import pubchempy as pcp
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint

class RecrystallizationRecommender(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Recrystallization Solvent Recommender")
        self.geometry("800x600")
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Load database with SMILES
        try:
            self.df = pd.read_excel('recrystallization_database_with_smiles.xlsx')
            self.status = "Database loaded successfully"
        except:
            self.df = None
            self.status = "Error: Could not load database"

        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Search section
        search_frame = ttk.LabelFrame(main_frame, text="Search by Name, CAS No. or SMILES", padding="5")
        search_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Search type selector
        self.search_type = ttk.Combobox(search_frame, values=['Name', 'CAS No.', 'SMILES'], width=10)
        self.search_type.set('Name')
        self.search_type.grid(row=0, column=0, padx=5)
        
        # Search entry
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=50)
        self.search_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(search_frame, text="Search", command=self.search).grid(row=0, column=2, padx=5)

        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, width=60, height=20)
        self.results_text.grid(row=0, column=0, padx=5, pady=5)

        # Status bar
        self.status_var = tk.StringVar(value=self.status)
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)

        # Configure grid weights
        self.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        search_frame.grid_columnconfigure(1, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)

    def get_smiles_from_cas(self, cas):
        """Get SMILES string from CAS number using PubChem"""
        if pd.isna(cas):
            return None
        try:
            compounds = pcp.get_compounds(cas, 'name')
            if compounds:
                return compounds[0].canonical_smiles
            return None
        except:
            return None

    def calculate_structure_similarity(self, smiles1, smiles2):
        """Calculate Tanimoto similarity between two SMILES strings using Morgan fingerprints"""
        if pd.isna(smiles1) or pd.isna(smiles2):
            return 0
        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            if mol1 is None or mol2 is None:
                return 0
                
            # Use GetHashedMorganFingerprint instead
            fp1 = GetHashedMorganFingerprint(mol1, 2)
            fp2 = GetHashedMorganFingerprint(mol2, 2)
            
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        except:
            return 0

    def search(self):
        if self.df is None:
            self.status_var.set("Error: No database loaded")
            return
            
        query = self.search_var.get().strip()
        if not query:
            self.status_var.set("Please enter a search term")
            return

        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        search_type = self.search_type.get()
        matches = []
        
        try:
            if search_type == 'Name':
                # Name search logic remains unchanged
                exact_matches = self.df[self.df['name'].str.lower() == query.lower()]
                if not exact_matches.empty:
                    matches = [(row['name'], 100) for idx, row in exact_matches.iterrows()]
                else:
                    # Then try fuzzy name matching only
                    name_matches = process.extract(query, self.df['name'].tolist(), scorer=fuzz.ratio, limit=3)
                    matches = name_matches
                    
            elif search_type == 'CAS No.':
                # First get SMILES from CAS number
                query_smiles = self.get_smiles_from_cas(query)
                if query_smiles:
                    # If we got SMILES, do structure similarity search
                    query_mol = Chem.MolFromSmiles(query_smiles)
                    if query_mol:
                        valid_smiles_df = self.df[self.df['SMILES'].notna()]
                        for idx, row in valid_smiles_df.iterrows():
                            struct_score = self.calculate_structure_similarity(query_smiles, row['SMILES']) * 100
                            if struct_score > 50:  # Only include if similarity > 50%
                                matches.append((row['name'], struct_score))
                else:
                    # If no SMILES found, try exact CAS match as fallback
                    exact_matches = self.df[self.df['cas'] == query]
                    if not exact_matches.empty:
                        matches = [(row['name'], 100) for idx, row in exact_matches.iterrows()]
                
            else:  # SMILES search remains unchanged
                query_mol = Chem.MolFromSmiles(query)
                if query_mol:
                    # Only compare with entries that have SMILES
                    valid_smiles_df = self.df[self.df['SMILES'].notna()]
                    for idx, row in valid_smiles_df.iterrows():
                        struct_score = self.calculate_structure_similarity(query, row['SMILES']) * 100
                        if struct_score > 50:  # Only include if similarity > 50%
                            matches.append((row['name'], struct_score))
                    
        except Exception as e:
            self.status_var.set(f"Search error: {str(e)}")
            return
            
        # Sort and take top 3
        matches = sorted(matches, key=lambda x: x[1], reverse=True)[:3]
        
        if matches:
            self.display_results(query, matches)
        else:
            self.results_text.insert(tk.END, "No matches found.\n")
            self.status_var.set("No matches found")

    def display_results(self, query, matches):
        if not matches:
            self.results_text.insert(tk.END, "No matches found.\n")
            return
            
        self.results_text.insert(tk.END, f"Search results for: {query}\n")
        self.results_text.insert(tk.END, "-" * 50 + "\n\n")
        
        for chemical_name, score in matches:
            row = self.df[self.df['name'] == chemical_name].iloc[0]
            
            self.results_text.insert(tk.END, f"Chemical: {chemical_name}\n")
            if pd.notna(row['cas']):
                self.results_text.insert(tk.END, f"CAS: {row['cas']}\n")
            if pd.notna(row['SMILES']):
                self.results_text.insert(tk.END, f"SMILES: {row['SMILES']}\n")
            self.results_text.insert(tk.END, f"Match Score: {score:.1f}%\n")
            
            self.results_text.insert(tk.END, "Recommended solvents:\n")
            solvent_count = 0
            for i in range(1, 6):
                solvent = row[f'solvent_{i}']
                if pd.notna(solvent):
                    solvent_count += 1
                    self.results_text.insert(tk.END, f"  {solvent_count}. {solvent}\n")
            
            self.results_text.insert(tk.END, "-" * 50 + "\n\n")
        
        self.status_var.set(f"Found {len(matches)} matches")

if __name__ == "__main__":
    app = RecrystallizationRecommender()
    app.mainloop()
