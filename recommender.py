import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re

class RecommenderSystem:
    def __init__(self):
        self.df_products = None
        self.df_ratings = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.categories = []
        self.brands = []

    def load_data(self, products_path='products.csv', ratings_path='ratings.csv'):
        """Loads products and ratings data."""
        try:
            self.df_products = pd.read_csv(products_path)
            self.df_ratings = pd.read_csv(ratings_path)
            
            # Extract unique categories and brands for filtering
            self.categories = self.df_products['category'].unique().tolist()
            self.brands = self.df_products['brand'].unique().tolist()
            
            print("Data loaded successfully.")
            return True
        except FileNotFoundError:
            print("Error: CSV files not found. Please generate data first.")
            return False

    def preprocess(self):
        """Preprocesses data and builds the content-based filtering model."""
        if self.df_products is None:
            return

        # Combine text features for TF-IDF
        self.df_products['combined_features'] = (
            self.df_products['name'] + " " + 
            self.df_products['category'] + " " + 
            self.df_products['brand'] + " " +
            self.df_products['description']
        )
        
        self.df_products['combined_features'] = self.df_products['combined_features'].fillna('')

        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.df_products['combined_features'])

        # Compute Cosine Similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Create a reverse mapping of indices and product names
        self.indices = pd.Series(self.df_products.index, index=self.df_products['name']).drop_duplicates()
        print("Preprocessing complete. Model built.")

    def recommend(self, query, category=None, top_n=8):
        """
        Smart recommendation logic with Price Parsing and Category Filtering.
        """
        if self.df_products is None:
            return pd.DataFrame(columns=['name', 'category', 'brand', 'price', 'description', 'rating', 'review_count', 'image_url'])

        query_lower = query.lower().strip() if query else ""
        
        # --- Price Parsing ---
        price_limit = None
        # Regex for "under 50000", "below 50000", "for 50000"
        price_match = re.search(r'(under|below|for|less than)\s+(\d+)', query_lower)
        if price_match:
            price_limit = int(price_match.group(2))
            # Remove the price part from query to avoid text matching issues
            query_lower = re.sub(r'(under|below|for|less than)\s+(\d+)', '', query_lower).strip()
            print(f"Price Limit Detected: {price_limit}")

        # Start with full dataset
        df_filtered = self.df_products.copy()

        # 1. Apply Price Filter
        if price_limit:
            df_filtered = df_filtered[df_filtered['price'] <= price_limit]

        # 2. Apply Category Filter (Explicit)
        if category and category != "All":
            df_filtered = df_filtered[df_filtered['category'] == category]
        
        if df_filtered.empty:
            return pd.DataFrame(columns=self.df_products.columns)

        # If no query, just return top rated in the filtered set
        if not query_lower:
            return df_filtered.sort_values(['rating', 'review_count'], ascending=[False, False]).head(top_n)

        # --- Logic on Filtered Data (Search) ---
        
        # 3. Category Match (Implicit from query, if not already filtered)
        # Only check if explicit category wasn't provided
        if not category or category == "All":
            matched_category = next((cat for cat in self.categories if cat.lower() in query_lower or query_lower in cat.lower()), None)
            if matched_category:
                print(f"Category Match: {matched_category}")
                # Filter by category
                cat_filtered = df_filtered[df_filtered['category'] == matched_category].copy()
                if not cat_filtered.empty:
                    return cat_filtered.sort_values(['rating', 'review_count'], ascending=[False, False]).head(top_n)

        # 4. Brand Match
        matched_brand = next((brand for brand in self.brands if brand.lower() in query_lower or query_lower in brand.lower()), None)
        
        if matched_brand:
            print(f"Brand Match: {matched_brand}")
            brand_filtered = df_filtered[df_filtered['brand'] == matched_brand].copy()
            if not brand_filtered.empty:
                return brand_filtered.sort_values(['rating', 'review_count'], ascending=[False, False]).head(top_n)

        # 5. Content-Based Similarity & Enhanced Search
        
        # Exact match check
        exact_match = df_filtered[df_filtered['name'].str.lower() == query_lower]
        if not exact_match.empty:
            # Return exact match + similar items
            # (Simplified: just return exact match for now to be precise)
            return exact_match.head(top_n)

        # Term-based scoring
        terms = query_lower.split()
        if not terms:
             return df_filtered.sort_values('rating', ascending=False).head(top_n)
        
        def calculate_match_score(row):
            score = 0
            text = row['combined_features'].lower()
            for term in terms:
                if term in text:
                    score += 1
            return score

        df_filtered['match_score'] = df_filtered.apply(calculate_match_score, axis=1)
        
        # Filter out rows with 0 match score
        df_filtered = df_filtered[df_filtered['match_score'] > 0]
        
        if not df_filtered.empty:
            # Sort by match score (desc), then rating (desc)
            df_filtered = df_filtered.sort_values(['match_score', 'rating', 'review_count'], ascending=[False, False, False])
            return df_filtered.head(top_n)

        return pd.DataFrame(columns=self.df_products.columns)

        # 3. Content-Based Similarity & Enhanced Search
        # If query is complex (e.g. "Samsung Phone 5G"), we need to check description/combined features
        
        # First, check for exact product name match
        if query in self.indices:
            idx = self.indices[query]
            if isinstance(idx, pd.Series):
                idx = idx.iloc[0]
            
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:top_n+1]
            product_indices = [i[0] for i in sim_scores]
            
            recommendations = self.df_products.iloc[product_indices].copy()
            recommendations['similarity_score'] = [round(i[1], 2) for i in sim_scores]
            return recommendations

        # If not exact match, try to filter by terms
        # Split query into terms
        terms = query_lower.split()
        
        # Start with all products
        filtered = self.df_products.copy()
        
        # Filter products that contain ANY of the terms in their combined features
        # But to be more specific, let's score them based on how many terms match
        
        def calculate_match_score(row):
            score = 0
            text = row['combined_features'].lower()
            for term in terms:
                if term in text:
                    score += 1
            return score

        filtered['match_score'] = filtered.apply(calculate_match_score, axis=1)
        
        # Filter out rows with 0 match score
        filtered = filtered[filtered['match_score'] > 0]
        
        if not filtered.empty:
            # Sort by match score (desc), then rating (desc)
            filtered = filtered.sort_values(['match_score', 'rating', 'review_count'], ascending=[False, False, False])
            return filtered.head(top_n)

        return pd.DataFrame(columns=self.df_products.columns)

    def get_all_product_names(self):
        if self.df_products is not None:
            return self.df_products['name'].tolist()
        return []
    
    def get_all_categories(self):
        return self.categories

if __name__ == "__main__":
    rec = RecommenderSystem()
    if rec.load_data():
        rec.preprocess()
        print("\nTest: Mobile")
        res = rec.recommend("Mobile")
        if not res.empty:
            print(res[['name', 'rating', 'price']].head(3))
        else:
            print("No results for Mobile")
            
        print("\nTest: Samsung")
        res = rec.recommend("Samsung")
        if not res.empty:
            print(res[['name', 'rating', 'price']].head(3))
        else:
            print("No results for Samsung")
