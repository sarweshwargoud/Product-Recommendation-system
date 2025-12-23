import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import re
import os

class RecommenderSystem:
    def __init__(self):
        self.df_products = None
        self.df_ratings = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.categories = []
        self.brands = []
        self.user_item_matrix = None
        self.item_similarity = None
        self.collab_indices = None
        
        # Known brands for cleaner extraction
        self.POPULAR_BRANDS = [
            'Apple', 'Samsung', 'HP', 'Dell', 'Lenovo', 'Asus', 'Acer', 'Microsoft', 
            'Sony', 'Canon', 'Nikon', 'Fujifilm', 'GoPro', 'Garmin', 'Fitbit', 
            'Bose', 'JBL', 'Sennheiser', 'Marshall', 'Beats', 'Xiaomi', 'Realme', 
            'OnePlus', 'Google', 'Oppo', 'Vivo', 'Huawei', 'Motorola', 'Nokia', 'Amazon',
            'Logitech', 'Razer', 'Corsair', 'SteelSeries', 'HyperX', 'Adidas', 'Nike', 'Puma',
            'Casio', 'Fossil', 'Tommy Hilfiger', 'Armani', 'Rolex', 'Seiko'
        ]

    def load_data(self, products_path='amz_uk_processed_data.csv', ratings_path='ratings.csv'):
        """Loads data with stratified sampling to ensure category diversity."""
        try:
            print("Loading data...")
            if 'amz_uk' in products_path:
                # 1. Define Categories and Quotas
                target_categories = [
                    'Laptops', 
                    'Mobile Phones & Smartphones', 
                    'Headphones & Earphones', 
                    'Smartwatches', 
                    'Cameras', 'Digital Cameras', 'Action Cameras',
                    'PC Gaming Accessories',
                    'Shoes', 'Watches'
                ]
                
                # We want ~2500 items per category group to get a good mix
                quota_per_cat = 2500
                collected_counts = {cat: 0 for cat in target_categories}
                chunks_to_concat = []
                
                # Clean Price Function (GBP -> INR)
                def clean_price(p):
                    try:
                        p_str = str(p).replace('£', '').replace(',', '').strip()
                        return int(float(p_str) * 105)
                    except:
                        return 0

                # Brand Extraction Function
                def extract_brand(title):
                    if not isinstance(title, str): return "Generic"
                    title_upper = title.upper()
                    for brand in self.POPULAR_BRANDS:
                        # Check distinct word or at start
                        pattern = r'\b' + re.escape(brand.upper()) + r'\b'
                        if re.search(pattern, title_upper):
                            return brand
                    # Fallback: First word
                    return title.split()[0] if title else "Generic"

                print(f"Scanning for {target_categories}...")
                
                if not os.path.exists(products_path):
                    raise FileNotFoundError

                # scan the whole file efficiently
                # We use a larger chunksize to speed up iteration
                for chunk in pd.read_csv(products_path, chunksize=100000):
                    # For each target category, grab rows if we haven't met quota
                    relevant_mask = pd.Series(False, index=chunk.index)
                    
                    for cat in target_categories:
                        if collected_counts[cat] < quota_per_cat:
                            # Match category or sub-category
                            # (Simple strict match for now based on known csv values)
                            cat_mask = chunk['categoryName'] == cat
                            count_in_chunk = cat_mask.sum()
                            
                            if count_in_chunk > 0:
                                # Add to relevant set
                                relevant_mask = relevant_mask | cat_mask
                                collected_counts[cat] += count_in_chunk
                    
                    # Store the relevant rows from this chunk
                    if relevant_mask.any():
                        chunks_to_concat.append(chunk[relevant_mask].copy())
                    
                    # Check if all quotas filed (limit total)
                    if sum(collected_counts.values()) > 25000:
                        print("Hit max dataset size limit (25k). Stopping scan.")
                        break

                if not chunks_to_concat:
                    print("Warning: No target categories found. Loading head.")
                    self.df_products = pd.read_csv(products_path, nrows=5000)
                else:
                    self.df_products = pd.concat(chunks_to_concat)
                    print(f"Loaded {len(self.df_products)} items.")
                    print("Counts per category:", collected_counts)

                # Rename Columns
                self.df_products.rename(columns={
                    'asin': 'product_id',
                    'title': 'name',
                    'imgUrl': 'image_url',
                    'categoryName': 'category',
                    'stars': 'rating',
                    'reviews': 'review_count'
                }, inplace=True)

                # Apply Transformations
                self.df_products['price'] = self.df_products['price'].apply(clean_price)
                self.df_products['brand'] = self.df_products['name'].apply(extract_brand)
                self.df_products['description'] = self.df_products['name']
                
            else:
                self.df_products = pd.read_csv(products_path)

            # Metadata
            self.categories = sorted(self.df_products['category'].astype(str).unique().tolist())
            self.brands = sorted(self.df_products['brand'].astype(str).unique().tolist())

            # 2. Ratings logic (Regenerate if needed)
            regenerate = True
            if os.path.exists(ratings_path):
                try:
                    self.df_ratings = pd.read_csv(ratings_path)
                    if not self.df_ratings.empty:
                        rated_ids = set(self.df_ratings['product_id'].unique())
                        product_ids = set(self.df_products['product_id'].unique())
                        overlap = rated_ids.intersection(product_ids)
                        if len(overlap) > 500: # Ensure decent overlap
                            regenerate = False
                            print(f"Ratings loaded with {len(overlap)} overlaps.")
                except:
                    pass
            
            if regenerate:
                print("Regenerating ratings...")
                self._generate_synthetic_ratings(ratings_path)

            self._build_collaborative_model()
            print("Data loaded successfully.")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _generate_synthetic_ratings(self, path):
        product_ids = self.df_products['product_id'].tolist()
        ratings_data = []
        user_ids = range(1, 101) 
        for user_id in user_ids:
            num_ratings = np.random.randint(5, 20)
            rated_items = np.random.choice(product_ids, size=min(num_ratings, len(product_ids)), replace=False)
            for pid in rated_items:
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.4, 0.25])
                ratings_data.append([user_id, pid, rating])
        self.df_ratings = pd.DataFrame(ratings_data, columns=['user_id', 'product_id', 'rating'])
        self.df_ratings.to_csv(path, index=False)

    def preprocess(self):
        if self.df_products is None: return
        
        # Align index for TF-IDF
        self.df_products.reset_index(drop=True, inplace=True)
        
        self.df_products['name'] = self.df_products['name'].fillna('')
        self.df_products['category'] = self.df_products['category'].fillna('')
        self.df_products['description'] = self.df_products['description'].fillna('')
        
        self.df_products['combined_features'] = (
            self.df_products['name'] + " " + 
            self.df_products['category'] + " " + 
            self.df_products['description']
        )
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df_products['combined_features'])
        print("TF-IDF Vectorization complete.")

    def _build_collaborative_model(self):
        if self.df_ratings is None: return
        self.user_item_matrix = self.df_ratings.pivot_table(index='product_id', columns='user_id', values='rating').fillna(0)
        if not self.user_item_matrix.empty:
            self.item_similarity = cosine_similarity(self.user_item_matrix)
            self.collab_indices = pd.Series(range(len(self.user_item_matrix)), index=self.user_item_matrix.index)
            print("Collaborative Model built.")

    def get_collaborative_recommendations(self, user_id=1, top_n=10):
        if self.user_item_matrix is None or user_id not in self.df_ratings['user_id'].values:
            return pd.DataFrame()
        user_ratings = self.df_ratings[self.df_ratings['user_id'] == user_id]
        liked_items = user_ratings[user_ratings['rating'] > 3]['product_id'].tolist()
        if not liked_items: return pd.DataFrame()
        
        scores = {}
        for item_id in liked_items:
            if item_id in self.collab_indices:
                idx = self.collab_indices[item_id]
                sim_scores = self.item_similarity[idx]
                for other_idx, score in enumerate(sim_scores):
                    if score > 0:
                        other_itm = self.user_item_matrix.index[other_idx]
                        if other_itm not in liked_items:
                            scores[other_itm] = scores.get(other_itm, 0) + score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        valid_ids = [k for k,v in sorted_items]
        return self.df_products[self.df_products['product_id'].isin(valid_ids)]

    def recommend(self, query, category=None, top_n=12, user_id=None):
        if self.df_products is None: return pd.DataFrame()
        
        df_filtered = self.df_products.copy()
        
        # 1. Price Parsing
        query_text = query.lower().strip() if query else ""
        price_limit = None
        match = re.search(r'(under|below|less than)\s+(\d+)', query_text)
        if match:
            try:
                price_limit = int(match.group(2))
                query_text = re.sub(r'(under|below|less than)\s+(\d+)', '', query_text).strip()
            except: pass
            
        if price_limit:
            df_filtered = df_filtered[df_filtered['price'] <= price_limit]

        # 2. Strict Filter Attempts (Relax able)
        initial_count = len(df_filtered)
        
        # Detect Brand
        found_brand = None
        if query_text:
            query_words = query_text.split()
            for brand in self.POPULAR_BRANDS:
                if any(brand.lower() == word for word in query_words):
                    found_brand = brand
                    break
        
        if found_brand:
            # Soft filtering: We prefer brand matches
            brand_subset = df_filtered[df_filtered['brand'] == found_brand]
            if not brand_subset.empty:
                df_filtered = brand_subset

        # Detect Category Keyword
        cat_keywords = {
            'laptop': ['Laptops', 'Computers'],
            'phone': ['Mobile', 'Phone', 'Smartphone'],
            'watch': ['Watch', 'Wearable'],
            'camera': ['Camera', 'Photo'],
            'headphone': ['Headphone', 'Earphone'],
            'shoe': ['Shoe', 'Sneaker']
        }
        
        found_cat_kw = False
        for kw, matches in cat_keywords.items():
            if kw in query_text:
                # Construct regex pattern
                pattern = '|'.join(matches)
                cat_subset = df_filtered[df_filtered['category'].str.contains(pattern, case=False, na=False)]
                if not cat_subset.empty:
                    df_filtered = cat_subset
                    found_cat_kw = True
                break
        
        # Widget Category Filter
        if category and category != "All":
            df_filtered = df_filtered[df_filtered['category'] == category]
            
        # 3. Text Search (TF-IDF)
        if query_text:
            query_vec = self.vectorizer.transform([query_text])
            
            # Use Index Slicing. 
            target_indices = df_filtered.index
            
            # Compute Sim
            subset_tfidf = self.tfidf_matrix[target_indices]
            sim_scores = linear_kernel(query_vec, subset_tfidf).flatten()
            
            df_filtered = df_filtered.copy() # Avoid SettingWithCopy
            df_filtered['similarity'] = sim_scores
            
            # Sort
            df_filtered = df_filtered.sort_values('similarity', ascending=False)
            
            # If we didn't find specific filters, assume general text search and filter weak matches
            if not (found_brand or found_cat_kw):
                df_filtered = df_filtered[df_filtered['similarity'] > 0]
                
        else:
            df_filtered['similarity'] = 0
            df_filtered = df_filtered.sort_values(['rating', 'review_count'], ascending=[False, False])

        # 4. Fallback if empty (Relaxation)
        if df_filtered.empty and initial_count > 0:
            return self.recommend_fallback(query, price_limit, top_n)

        return df_filtered.head(top_n)

    def recommend_fallback(self, query, price_limit, top_n):
        # Just pure text search on main dataset
        df = self.df_products.copy()
        if price_limit:
            df = df[df['price'] <= price_limit]
            
        try:
            query_vec = self.vectorizer.transform([query])
            sim_scores = linear_kernel(query_vec, self.tfidf_matrix).flatten()
            df['similarity'] = sim_scores
            return df[df['similarity'] > 0].sort_values('similarity', ascending=False).head(top_n)
        except:
            return pd.DataFrame()

    def get_all_categories(self):
        return self.categories

if __name__ == "__main__":
    rec = RecommenderSystem()
    if rec.load_data():
        rec.preprocess()
        # Non-ASCII chars can cause print issues in Windows terminals without chcp 65001
        try:
            print("Testing Search 'HP Laptop under 40000':")
            print(rec.recommend("HP Laptop under 40000", top_n=5)[['name', 'brand', 'price']])
        except:
            print("Error printing results due to encoding.")
