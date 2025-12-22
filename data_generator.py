import pandas as pd
import numpy as np
import random

def generate_data():
    print("Generating expanded synthetic data with INR pricing...")
    
    products = []
    
    # Define categories and their specific brands/attributes
    categories_data = {
        'Mobiles': {
            'brands': ['Apple', 'Samsung', 'Google', 'OnePlus', 'Xiaomi', 'Realme'],
            'adjectives': ['Pro', 'Ultra', 'Max', 'Lite', '5G', 'Fold', 'Flip'],
            'nouns': ['iPhone 15', 'Galaxy S24', 'Pixel 8', 'Nord', 'Redmi Note', 'GT'],
            'price_range': (10000, 150000), # INR
            'image_ids': ['Rv2kTIuya_I', 'uFyioIkaep8', 'ZHrVk_KPiJU']
        },
        'Laptops': {
            'brands': ['Dell', 'HP', 'Apple', 'Lenovo', 'Asus', 'Acer'],
            'adjectives': ['Gaming', 'Ultrabook', 'Pro', 'Air', 'Convertible', 'Business'],
            'nouns': ['XPS', 'Spectre', 'MacBook', 'ThinkPad', 'ROG', 'Swift'],
            'price_range': (30000, 250000), # INR
            'image_ids': ['7Zvf8045-Qo', 'nVqRBcNBsno', 'UIgiD10aTOk']
        },
        'Headphones': {
            'brands': ['Sony', 'Bose', 'JBL', 'Sennheiser', 'Apple', 'Beats'],
            'adjectives': ['Wireless', 'Noise-Cancelling', 'True Wireless', 'Studio', 'Bass'],
            'nouns': ['WH-1000XM5', 'QuietComfort', 'Tune', 'Momentum', 'AirPods', 'Studio3'],
            'price_range': (2000, 35000), # INR
            'image_ids': ['JphZSKMy2qA', 'eiGjG_n4BrU', 'wjEcEOI8rWg']
        },
        'Fashion': {
            'brands': ['Nike', 'Adidas', 'Zara', 'H&M', 'Levi\'s', 'Gucci'],
            'adjectives': ['Casual', 'Sport', 'Formal', 'Vintage', 'Slim-Fit', 'Summer'],
            'nouns': ['T-Shirt', 'Sneakers', 'Jeans', 'Jacket', 'Dress', 'Hoodie'],
            'price_range': (500, 15000), # INR
            'image_ids': ['h8-ISoZOC9M', 'acn5ERAeSb4', '4rUYuwJ2vGw', 'fiA7sq2PdnQ']
        },
        'Smart Watches': {
            'brands': ['Apple', 'Samsung', 'Garmin', 'Fitbit', 'Fossil'],
            'adjectives': ['Series 9', 'Pro', 'Sport', 'Classic', 'Versa'],
            'nouns': ['Watch', 'Galaxy Watch', 'Fenix', 'Sense', 'Gen 6'],
            'price_range': (5000, 80000), # INR
            'image_ids': ['n2ZrdKos-rU', '29pdRYIyo-4', 'PZIMsQAnEh8']
        },
        'Cameras': {
            'brands': ['Canon', 'Nikon', 'Sony', 'Fujifilm', 'GoPro'],
            'adjectives': ['DSLR', 'Mirrorless', 'Action', '4K', 'Compact'],
            'nouns': ['EOS', 'Z6', 'Alpha', 'X-T5', 'Hero'],
            'price_range': (25000, 300000), # INR
            'image_ids': ['xSTQu4g6lZ8', 'HjZ0iY5PkI0', 'pWhFcl-LvXI', 'JAJhlDObGI8', 'G6PElEPGr9k']
        }
    }

    product_id_counter = 1
    
    for category, data in categories_data.items():
        # Generate ~40 items per category
        for _ in range(40):
            brand = random.choice(data['brands'])
            adj = random.choice(data['adjectives'])
            noun = random.choice(data['nouns'])
            
            # Ensure name sounds somewhat realistic
            if category == 'Fashion':
                name = f"{brand} {adj} {noun}"
            else:
                name = f"{brand} {noun} {adj}"
                
            # Randomize name slightly to avoid duplicates
            if random.random() > 0.5:
                name += f" {random.randint(2024, 2025)}"
            
            price = random.randint(data['price_range'][0], data['price_range'][1])
            # Make price look nice (e.g. 999 instead of 997)
            price = (price // 100) * 100 + 99
            
            rating = round(random.uniform(3.0, 5.0), 1)
            reviews = random.randint(10, 2000)
            
            # Detailed description with "specifications" keywords
            specs = []
            if category == 'Mobiles':
                specs = [f"{random.choice(['128GB', '256GB', '512GB'])} Storage", f"{random.choice(['8GB', '12GB'])} RAM", "50MP Camera", "5000mAh Battery"]
            elif category == 'Laptops':
                specs = [f"{random.choice(['i5', 'i7', 'i9', 'M3'])} Processor", "16GB RAM", "1TB SSD", "14-inch Display"]
            elif category == 'Headphones':
                specs = ["30h Battery", "Active Noise Cancellation", "Bluetooth 5.3"]
            
            spec_str = " | ".join(specs)
            
            desc_templates = [
                f"The new {name} from {brand} is a game changer. Features: {spec_str}.",
                f"Experience premium quality with {brand}'s {name}. Top rated by users. Specs: {spec_str}.",
                f"Get the best value with {name}. Perfect for everyday use. Includes {spec_str}.",
                f"High performance {category} designed for enthusiasts. {adj} features included.",
                f"Stylish and durable, the {name} is a must-have. Comes with {spec_str}."
            ]
            description = random.choice(desc_templates)
            
            # Curated Unsplash Image
            img_id = random.choice(data['image_ids'])
            image_url = f"https://images.unsplash.com/photo-{img_id}?auto=format&fit=crop&w=300&q=80"
            
            products.append([product_id_counter, name, category, brand, price, description, rating, reviews, image_url])
            product_id_counter += 1
            
    df_products = pd.DataFrame(products, columns=['product_id', 'name', 'category', 'brand', 'price', 'description', 'rating', 'review_count', 'image_url'])
    df_products.to_csv('products.csv', index=False)
    print(f"Created products.csv with {len(df_products)} items across {len(categories_data)} categories.")

    # 2. Generate Ratings (User-Item interactions)
    # 50 Users, each rating 10-30 products
    ratings = []
    user_ids = range(1, 51)
    
    for user_id in user_ids:
        num_ratings = random.randint(10, 30)
        rated_products = random.sample(range(1, product_id_counter), num_ratings)
        
        for prod_id in rated_products:
            # Bias towards higher ratings for better products (simulated)
            prod_rating = df_products.loc[df_products['product_id'] == prod_id, 'rating'].values[0]
            
            if prod_rating >= 4.0:
                user_rating = random.choices([3, 4, 5], weights=[10, 40, 50])[0]
            else:
                user_rating = random.choices([1, 2, 3, 4, 5], weights=[20, 20, 30, 20, 10])[0]
                
            ratings.append([user_id, prod_id, user_rating])
            
    df_ratings = pd.DataFrame(ratings, columns=['user_id', 'product_id', 'rating'])
    df_ratings.to_csv('ratings.csv', index=False)
    print(f"Created ratings.csv with {len(df_ratings)} interactions.")

if __name__ == "__main__":
    generate_data()
