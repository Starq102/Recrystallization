import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def load_database(file_path='recrystallization_database.xlsx'):
    """Load the recrystallization database"""
    return pd.read_excel(file_path)

def find_similar_chemicals(query, df, limit=3):
    """Find similar chemical names using fuzzy string matching"""
    # Get all chemical names from database
    chemical_names = df['name'].tolist()
    
    # Find matches using fuzzy string matching
    matches = process.extract(query, chemical_names, scorer=fuzz.ratio, limit=limit)
    return matches

def get_solvents_for_chemical(chemical_name, df):
    """Get all available solvents for a given chemical"""
    row = df[df['name'] == chemical_name].iloc[0]
    solvents = []
    
    # Collect all non-null solvents
    for i in range(1, 6):
        solvent = row[f'solvent_{i}']
        if pd.notna(solvent):
            solvents.append(solvent)
    
    return solvents

def recommend_solvents():
    # Load database
    df = load_database()
    
    # Get user input
    query = input("\nEnter chemical name for recrystallization: ")
    
    print("\nSearching database...")
    
    # Find similar chemicals
    matches = find_similar_chemicals(query, df)
    
    print("\nResults:")
    print("-" * 50)
    
    for chemical_name, score in matches:
        print(f"\nChemical: {chemical_name}")
        print(f"Match Score: {score}%")
        
        # Get solvents
        solvents = get_solvents_for_chemical(chemical_name, df)
        print("Recommended solvents:")
        for i, solvent in enumerate(solvents, 1):
            print(f"  {i}. {solvent}")
        print("-" * 50)

if __name__ == "__main__":
    print("Recrystallization Solvent Recommendation System")
    print("=" * 50)
    
    while True:
        recommend_solvents()
        
        if input("\nSearch another chemical? (y/n): ").lower() != 'y':
            break
    
    print("\nThank you for using the recommendation system!")
