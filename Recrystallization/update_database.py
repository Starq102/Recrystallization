import pandas as pd 
import pubchempy as pcp
import time

def get_smiles_from_cas(cas):
    """Get SMILES string from CAS number using PubChem"""
    if pd.isna(cas):
        return None
    try:
        # Add delay to avoid overwhelming PubChem API
        time.sleep(0.5)
        compounds = pcp.get_compounds(cas, 'name')
        if compounds:
            return compounds[0].canonical_smiles
        return None
    except:
        print(f"Failed to get SMILES for CAS: {cas}")
        return None

def update_database():
    print("Loading database...")
    df = pd.read_excel('recrystallization_database.xlsx')
    print(f"Loaded {len(df)} entries")
    
    # Check if SMILES column exists
    if 'SMILES' not in df.columns:
        print("Adding SMILES column...")
        df['SMILES'] = None
        
    # Get SMILES for entries without them
    missing_smiles = df[df['SMILES'].isna() & df['cas'].notna()]
    print(f"Found {len(missing_smiles)} entries missing SMILES")
    
    for idx, row in missing_smiles.iterrows():
        print(f"Processing {idx+1}/{len(missing_smiles)}: {row['name']}")
        smiles = get_smiles_from_cas(row['cas'])
        df.loc[idx, 'SMILES'] = smiles
        
        # Save progress periodically
        if (idx + 1) % 10 == 0:
            print("Saving progress...")
            df.to_excel('recrystallization_database_with_smiles.xlsx', index=False)
    
    # Final save
    print("Saving final database...")
    df.to_excel('recrystallization_database_with_smiles.xlsx', index=False)
    print("Done!")
    
    # Print statistics
    total = len(df)
    with_smiles = len(df[df['SMILES'].notna()])
    print(f"\nDatabase statistics:")
    print(f"Total entries: {total}")
    print(f"Entries with SMILES: {with_smiles}")
    print(f"Coverage: {(with_smiles/total)*100:.1f}%")

if __name__ == "__main__":
    update_database()
