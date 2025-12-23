
import pandas as pd

def check_categories():
    keywords = ['Mobile', 'Phone', 'Laptop', 'Computer', 'Headphone', 'Camera', 'Watch', 'Electronics', 'Fashion', 'Clothing']
    found_categories = set()
    
    print("Scanning file for relevant categories...")
    try:
        chunksize = 100000
        for chunk in pd.read_csv('amz_uk_processed_data.csv', usecols=['categoryName'], chunksize=chunksize):
            unique_cats = chunk['categoryName'].dropna().unique()
            for cat in unique_cats:
                if any(k.lower() in str(cat).lower() for k in keywords):
                    found_categories.add(cat)
                    
        print("\nFound Relevant Categories:")
        for cat in sorted(list(found_categories)):
            print(cat)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_categories()
