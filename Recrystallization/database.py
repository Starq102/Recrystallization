import re
import pandas as pd

def extract_cas(text):
    cas_match = re.search(r'\[(\d{1,7}-\d{2}-\d{1})\]', text)
    return cas_match.group(1) if cas_match else None

def clean_solvent_text(solvent):
    if not solvent:
        return None
        
    # Expanded comprehensive solvent map
    solvent_map = {
        # Basic solvents
        'water': 'water',
        'methanol': 'methanol',
        'ethanol': 'ethanol',
        'acetone': 'acetone',
        'chloroform': 'chloroform',
        'benzene': 'benzene',
        'toluene': 'toluene',
        'xylene': 'xylene',
        'hexane': 'hexane',
        'cyclohexane': 'cyclohexane',
        'petroleum_ether': 'petroleum_ether',
        'diethyl_ether': 'diethyl_ether',
        'acetonitrile': 'acetonitrile',
        'dimethylformamide': 'dimethylformamide',
        'dimethyl_sulfoxide': 'dimethyl_sulfoxide',
        'tetrahydrofuran': 'tetrahydrofuran',
        'ethyl_acetate': 'ethyl_acetate',
        'acetic_acid': 'acetic_acid',
        'dioxane': 'dioxane',
        'carbon_tetrachloride': 'carbon_tetrachloride',
        'isopropanol': 'isopropanol',
        
        # Common abbreviations
        'MeOH': 'methanol',
        'EtOH': 'ethanol',
        'Et2O': 'diethyl_ether',
        'Me2CO': 'acetone',
        'EtOAc': 'ethyl_acetate',
        'AcOEt': 'ethyl_acetate',
        'CHCl3': 'chloroform',
        'CCl4': 'carbon_tetrachloride',
        'H2O': 'water',
        '*C6H6': 'benzene',
        '*benzene': 'benzene',
        'Me2CHOH': 'isopropanol', 
        'iPrOH': 'isopropanol',
        'iso-PrOH': 'isopropanol',
        'MeCN': 'acetonitrile',
        'DMF': 'dimethylformamide',
        'DMSO': 'dimethyl_sulfoxide',
        'THF': 'tetrahydrofuran',
        'petroleum ether': 'petroleum_ether',
        'pet ether': 'petroleum_ether',
        'ligroin': 'petroleum_ether',
        'hexanes': 'hexane',
        'ether': 'diethyl_ether',
        'AcOH': 'acetic_acid',
        'aqueous': 'water',
        'aq': 'water',
        'aq.': 'water'
    }

    # Modified patterns to preserve some useful descriptors
    patterns_to_remove = [
        r'\(.*?\)',          # Remove parenthetical information
        r'\[.*?\]',          # Remove bracketed information
        r'hot',              # Remove temperature descriptors
        r'cold',
        r'warm',
        r'cool',
        r'boiling',
        r'dilute',
        r'concentrated',
        r'freshly',
        r'\d+\s*ml[/\s]g',  # Remove volume ratios
        r'after.*$',        # Remove procedural text
        r'then.*$',
        r'followed by.*$',
        r'using.*$',
        r'with.*$',
        r'at.*$',           # Remove temperature/conditions
        r'under.*$',
        r'over.*$',
        r'reflux.*$',
        r'by cooling.*$'
    ]
    
    cleaned = solvent.lower()
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Clean up whitespace and common separators
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip(' .,;')
    
    # Replace abbreviations with standardized names
    for abbrev, full in solvent_map.items():
        cleaned = re.sub(r'\b' + re.escape(abbrev.lower()) + r'\b', full, cleaned)
    
    # Special handling for percentages and aqueous mixtures
    cleaned = re.sub(r'(\d+)%\s*([a-z_]+)', r'\2/water', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'aqueous\s+([a-z_]+)', r'\1/water', cleaned, flags=re.IGNORECASE)
    
    # Convert solvent string to list of valid solvents
    solvents = []
    parts = re.split(r'(?:\s+and\s+|\s*[/,]\s*|\s+or\s+)', cleaned)
    
    for part in parts:
        part = part.strip().lower()
        # Replace any remaining spaces with underscores
        part = re.sub(r'\s+', '_', part)
        
        if part in solvent_map.values():
            solvents.append(part)
            
    # Join multiple solvents with "/" to indicate mixtures
    return ["/".join(solvents)] if solvents else None

def extract_recryst_solvent(text):
    # More comprehensive crystallization patterns 
    patterns = [
        r'(?:re)?crystalli[sz]e[sd]?\s+from\s+(.*?)(?:[\.,]|$)',
        r'crystalli[sz]es?\s+from\s+(.*?)(?:[\.,]|$)', 
        r'crystallisation\s+(?:from|in)\s+(.*?)(?:[\.,]|$)',
        r'crystallised\s+(?:from|in)\s+(.*?)(?:[\.,]|$)',
        r'crystallises?\s+(?:from|in)\s+(.*?)(?:[\.,]|$)',
        r'crystalline\s+(?:from|in)\s+(.*?)(?:[\.,]|$)',
        r'(?:re)?crystalli[sz](?:ation|ed)?\s+(?:from|in)\s+(.*?)(?:[\.,]|$)',
        # Add simple "Crystallise X from Y" pattern
        r'Crystallise\s+(?:the\s+)?(?:acid|salt|compound|ester|alcohol)?\s+from\s+(.*?)(?:[\.,]|$)',
        # Add simpler backup pattern for missed cases
        r'from\s+(.*?)(?:[\.,]|$)'
    ]
    
    all_solvents = []
    
    # Try to find all solvent mentions in the text
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            solvent_text = match.group(1).split('.')[0]
            cleaned = clean_solvent_text(solvent_text)
            if cleaned:
                all_solvents.extend(cleaned)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_solvents = [x for x in all_solvents if not (x in seen or seen.add(x))]
    
    return unique_solvents if unique_solvents else None

def extract_chemical_name(text):
    # Improved pattern to capture more chemical names
    name_pattern = r'^([^[\n]+?)(?:\s*\[[\d-]+\])'
    
    match = re.search(name_pattern, text, re.MULTILINE)
    if match:
        name = match.group(1).strip()
        # Clean up trailing spaces, parentheses and punctuation
        name = re.sub(r'\s+', ' ', name)
        # Keep text in parentheses as it may be part of the name
        name = name.strip(' .,;')
        return name
    return None

def parse_chemical_data(text_file, chunk_size=1000):
    chemicals = []
    
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
        paragraphs = text.split('\n\n')
        
        for i in range(0, len(paragraphs), chunk_size):
            chunk = paragraphs[i:i + chunk_size]
            
            for para in chunk:
                if not para.strip():
                    continue
                
                name = extract_chemical_name(para)
                cas = extract_cas(para)
                solvents = extract_recryst_solvent(para)
                
                if name and len(name) > 4 and solvents:
                    chemical = {
                        'name': name,
                        'cas': cas,
                        'solvent_1': solvents[0] if len(solvents) > 0 else None,
                        'solvent_2': solvents[1] if len(solvents) > 1 else None,
                        'solvent_3': solvents[2] if len(solvents) > 2 else None,
                        'solvent_4': solvents[3] if len(solvents) > 3 else None,
                        'solvent_5': solvents[4] if len(solvents) > 4 else None
                    }
                    chemicals.append(chemical)
                
    return chemicals

def create_database(chemicals):
    df = pd.DataFrame(chemicals)
    
    # Filter out rows with invalid values
    df = df[
        (df['name'].str.len() > 4) & 
        (df['solvent_1'].notna())  # At least one solvent must exist
    ]
    
    # Reset index after filtering
    df = df.reset_index(drop=True)
    
    df.to_csv('recrystallization_database.csv', index=False)
    df.to_excel('recrystallization_database.xlsx', index=False)
    return df

if __name__ == '__main__':
    input_file = 'powders.txt'
    chemicals = parse_chemical_data(input_file)
    df = create_database(chemicals)
    print(f"Created database with {len(chemicals)} compounds")
