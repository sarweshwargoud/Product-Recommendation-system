from recommender import RecommenderSystem
import inspect

try:
    rec = RecommenderSystem()
    print("Class loaded successfully.")
    
    if hasattr(rec, 'get_all_categories'):
        print("Method 'get_all_categories' EXISTS.")
        print(f"Categories: {rec.get_all_categories()}")
    else:
        print("Method 'get_all_categories' DOES NOT EXIST.")
        print("Dir:", dir(rec))
        
except Exception as e:
    print(f"Error: {e}")
