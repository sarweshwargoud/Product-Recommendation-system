
import pandas as pd
from recommender import RecommenderSystem

def inspect_data():
    rec = RecommenderSystem()
    print("Loading data...")
    if rec.load_data():
        print(f"\nTotal Products: {len(rec.df_products)}")
        print("\nSample Brands (First 50):")
        print(rec.brands[:50])
        print("\nUnique Categories:")
        print(rec.categories)
        
        print("\nSample Product Titles & Extracted Brands:")
        print(rec.df_products[['name', 'brand', 'category']].head(10))

        # Test specific search
        query = "HP Laptop under 40000"
        print(f"\n--- Testing Search: '{query}' ---")
        results = rec.recommend(query)
        print(f"Results Found: {len(results)}")
        if not results.empty:
            print(results[['name', 'brand', 'category', 'price']])

if __name__ == "__main__":
    inspect_data()
