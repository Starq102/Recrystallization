import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

class OrgSynCrawler:
    def __init__(self):
        self.base_url = "https://orgsyn.org/"
        self.solvent_data = []
        
        # Patterns for finding recrystallization info
        self.recryst_patterns = [
            r"(?:re)?crystalliz(?:e|ation)\s+from\s+([^\.;]+)",
            r"(?:re)?crystallis(?:e|ation)\s+from\s+([^\.;]+)",
            r"after\s+recrystallization\s+from\s+([^\.;]+)",
            r"purified\s+by\s+recrystallization\s+from\s+([^\.;]+)",
        ]
        
        # Pattern for finding compound names before recryst
        self.compound_pattern = r"([A-Za-z0-9\-\(\),]+(?:\s+[A-Za-z0-9\-\(\),]+){0,5}?)\s+(?:was|is|were|are)\s+(?:re)?crystallized"

    def search_procedures(self, page=1):
        """Search through procedure pages on orgsyn.org"""
        try:
            url = f"{self.base_url}/Search/Search.aspx?q=&p={page}"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all procedure links
            procedures = soup.find_all('a', href=lambda x: x and 'Details' in x)
            
            for proc in procedures:
                try:
                    proc_url = f"{self.base_url}/{proc['href']}"
                    self.extract_recryst_info(proc_url)
                    time.sleep(2)  # Be nice to the server
                except Exception as e:
                    print(f"Error processing procedure {proc_url}: {e}")
                    
            return True
            
        except Exception as e:
            print(f"Error searching page {page}: {e}")
            return False

    def extract_recryst_info(self, url):
        """Extract recrystallization information from a procedure page"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get procedure text
            procedure = soup.find('div', {'class': 'procedure'})
            if not procedure:
                return
                
            text = procedure.get_text()
            
            # Look for recrystallization info
            for pattern in self.recryst_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Get solvents
                    solvents = match.group(1).strip()
                    
                    # Try to find compound name before recryst
                    compound_match = re.search(self.compound_pattern, text[:match.start()], re.IGNORECASE)
                    compound = compound_match.group(1) if compound_match else None
                    
                    if compound and solvents:
                        self.solvent_data.append({
                            'compound': compound.strip(),
                            'solvent': solvents,
                            'source': url
                        })
                        
        except Exception as e:
            print(f"Error extracting from {url}: {e}")

    def save_progress(self):
        """Save current progress to file"""
        if not self.solvent_data:
            return
            
        df = pd.DataFrame(self.solvent_data)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'recryst_orgsyn_{timestamp}.xlsx'
        df.to_excel(filename, index=False)
        print(f"Saved {len(self.solvent_data)} entries to {filename}")

def main():
    crawler = OrgSynCrawler()
    
    page = 1
    while crawler.search_procedures(page):
        print(f"Processed page {page}")
        crawler.save_progress()
        page += 1
        time.sleep(5)  # Delay between pages

if __name__ == "__main__":
    main()
