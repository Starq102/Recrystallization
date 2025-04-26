import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from scholarly import scholarly
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from msedge.selenium_tools import Edge
from webdriver_manager.microsoft import EdgeChromiumDriverManager

class RecrystallizationCrawler:
    def __init__(self):
        self.patterns = [
            r'(?:re)?crystalli[sz](?:ed|es|ation)?\s+from\s+(.*?)(?:[\.,]|$)',
            r'purified\s+by\s+(?:re)?crystallization\s+(?:from|in)\s+(.*?)(?:[\.,]|$)',
            r'(?:re)?crystalli[sz](?:ed|es|ation)?\s+(?:from|in)\s+(.*?)(?:[\.,]|$)'
        ]
        self.solvent_data = []
        self.save_counter = 0  # Add counter for periodic saves
        prin
        
    def search_google_scholar(self, query, num_results=10):
        """Search Google Scholar for recrystallization information"""
        print(f"Searching for: {query}")
        
        try:
            # Use search_pubs() and get all required information
            search_query = scholarly.search_pubs(query + " recrystallization")
            results = []
            
            for i in range(num_results):
                try:
                    paper = next(search_query)
                    # Extract relevant fields using new API
                    results.append({
                        'title': paper.get('title', ''),
                        'abstract': paper.get('abstract', ''),
                        'url': paper.get('pub_url', '')  # Changed from 'url' to 'pub_url'
                    })
                    time.sleep(2)  # Be nice to Google Scholar
                except StopIteration:
                    break  # No more results
                except Exception as e:
                    print(f"Error processing paper: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error searching Google Scholar: {e}")
            
        return results

    def extract_recryst_info(self, text):
        """Extract recrystallization information from text"""
        for pattern in self.patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                solvent_text = match.group(1)
                if solvent_text:
                    return solvent_text.strip()
        return None

    def crawl_chemistry_journals(self, compounds):
        """Crawl chemistry journals for recrystallization data"""
        try:
            # Initialize Edge driver with additional options
            options = Options()
            options.add_argument('--disable-gpu')  # Disable GPU acceleration
            options.add_argument('--no-sandbox')  # Disable sandbox
            options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource issues
            options.add_argument('--log-level=3')  # Minimize logging
            options.add_argument('--silent')  # Silent mode
            
            driver = webdriver.Edge(options=options)
            
        except Exception as e:
            print(f"Error initializing Edge driver: {e}")
            print("Please ensure Edge is installed and up to date")
            return pd.DataFrame()  # Return empty DataFrame on error
        
        try:
            for i, compound in enumerate(compounds):
                print(f"Searching for {compound}... ({i+1}/{len(compounds)})")
                
                # Search Google Scholar
                results = self.search_google_scholar(f"{compound} recrystallization")
                
                for result in results:
                    try:
                        # Visit the paper URL if available
                        if result['url']:
                            driver.get(result['url'])
                            WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.TAG_NAME, "body"))
                            )
                            
                            # Extract text
                            text = driver.find_element(By.TAG_NAME, "body").text
                            
                            # Look for recrystallization information
                            recryst_info = self.extract_recryst_info(text)
                            
                            if recryst_info:
                                self.solvent_data.append({
                                    'compound': compound,
                                    'solvent': recryst_info,
                                    'source': result['url']
                                })
                                
                            # Save progress every 5 compounds
                            if len(self.solvent_data) > 0 and len(self.solvent_data) % 5 == 0:
                                self.save_progress()
                                
                    except Exception as e:
                        print(f"Error processing {result['url']}: {e}")
                        
                    time.sleep(3)  # Be nice to the servers
                    
        finally:
            try:
                driver.quit()
            except:
                pass
            
        return pd.DataFrame(self.solvent_data)

    def save_progress(self):
        """Save current progress to timestamped file"""
        if not self.solvent_data:
            return
            
        # Convert current data to DataFrame
        df = pd.DataFrame(self.solvent_data)
        
        # Save with counter in filename
        self.save_counter += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'recrystallization_progress_{timestamp}_{self.save_counter}.xlsx'
        
        df.to_excel(filename, index=False)
        print(f"Progress saved to {filename}")
        print(f"Processed {len(self.solvent_data)} compounds so far")

    def update_database(self, new_data, existing_db_path='recrystallization_database_with_smiles.xlsx'):
        """Merge new data with existing database"""
        try:
            existing_df = pd.read_excel(existing_db_path)
            print(f"Loaded {len(existing_df)} existing entries")
            
            # Process new data to match existing format
            new_entries = []
            for _, row in new_data.iterrows():
                entry = {
                    'name': row['compound'],
                    'solvent_1': row['solvent'],
                    'source': row['source']
                }
                new_entries.append(entry)
                
            # Append new entries
            new_df = pd.DataFrame(new_entries)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['name', 'solvent_1'])
            
            # Save updated database with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f'recrystallization_database_web_{timestamp}.xlsx'
            combined_df.to_excel(output_file, index=False)
            print(f"Added {len(new_entries)} new entries to {output_file}")
            
        except Exception as e:
            print(f"Error updating database: {e}")

def main():
    # List of compounds to search for
    compounds = [
        # Common Pain Medications
        "aspirin", "acetaminophen", "ibuprofen", "naproxen", "diclofenac",
        
        # Antibiotics
        "amoxicillin", "tetracycline", "erythromycin", "penicillin", 
        "streptomycin", "chloramphenicol",
        
        # CNS Drugs
        "fluoxetine", "sertraline", "paroxetine", "diazepam", "alprazolam",
        
        # Anti-inflammatory 
        "prednisolone", "dexamethasone", "hydrocortisone",
        
        # Common Lab Chemicals
        "benzoic acid", "salicylic acid", "caffeine", "urea", "glycine",
        "citric acid", "tartaric acid", "succinic acid", "oxalic acid",
        "acetic acid", "fumaric acid", "maleic acid",
        
        # Industrial Chemicals
        "adipic acid", "terephthalic acid", "isophthalic acid", 
        "phthalic acid", "naphthalene", "anthracene", "phenanthrene",
        
        # Natural Products
        "menthol", "camphor", "vanillin", "coumarin", "quercetin",
        "rutin", "cholesterol", "testosterone", "progesterone",
        
        # Pesticides/Herbicides  
        "atrazine", "simazine", "malathion", "parathion", "carbaryl",
        
        # Dyes
        "methylene blue", "crystal violet", "rhodamine B", "fluorescein",
        "eosin Y"
    ]
    
    crawler = RecrystallizationCrawler()
    
    # Crawl for data - no need for driver_path anymore
    new_data = crawler.crawl_chemistry_journals(compounds)
    
    if not new_data.empty:
        # Only update database if we got some data
        crawler.update_database(new_data)
    else:
        print("No data collected - check Edge installation and try again")

if __name__ == "__main__":
    main()

