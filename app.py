import streamlit as st
import pandas as pd
from recommender import RecommenderSystem
import time

# Page Config
st.set_page_config(
    page_title="StreamFlix - Shop Smart",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Recommender
@st.cache_resource
def get_recommender():
    rec = RecommenderSystem()
    if rec.load_data():
        rec.preprocess()
    return rec

rec = get_recommender()

# Session State for Wishlist
if 'wishlist' not in st.session_state:
    st.session_state.wishlist = []

def add_to_wishlist(product_name):
    if product_name not in st.session_state.wishlist:
        st.session_state.wishlist.append(product_name)
        st.toast(f"Added {product_name} to Wishlist!", icon="❤️")
        time.sleep(0.5)
        st.rerun()
    else:
        st.toast(f"{product_name} is already in your Wishlist.", icon="ℹ️")

def remove_from_wishlist(product_name):
    if product_name in st.session_state.wishlist:
        st.session_state.wishlist.remove(product_name)
        st.toast(f"Removed {product_name} from Wishlist.", icon="🗑️")
        time.sleep(0.5)
        st.rerun()

# Custom CSS for Medium Dark Theme & Animations
st.markdown("""
    <style>
    /* Main Background - Medium Dark */
    .stApp {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #121212;
        border-right: 1px solid #333;
    }
    
    /* Typography */
    h1, h2, h3, h4, p, label, span, div {
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Top Search Bar Styling */
    .stTextInput > div > div > input {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border-radius: 8px;
        padding: 10px 20px;
        border: 1px solid #444;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4F46E5; /* Indigo Focus */
    }
    
    /* Product Card */
    .product-card {
        background-color: #2d2d2d;
        border-radius: 8px;
        padding: 15px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 20px;
        border: 1px solid #383838;
        position: relative;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        border-color: #4F46E5; /* Indigo Border on Hover */
    }
    
    .price-tag {
        font-size: 1.1em;
        font-weight: bold;
        color: #4ade80; /* Soft Green */
    }
    
    .rating-badge {
        background-color: #383838;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        color: #ffd700; /* Gold Star */
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #4F46E5; /* Indigo Primary */
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #4338ca; /* Darker Indigo */
    }
    
    /* Remove Button in Sidebar */
    .remove-btn {
        color: #ef4444; /* Red for remove */
        cursor: pointer;
    }
    
    </style>
""", unsafe_allow_html=True)

# --- Sidebar: Wishlist ---
st.sidebar.title("🛒 My Wishlist")
if st.session_state.wishlist:
    for item in st.session_state.wishlist:
        col_w1, col_w2 = st.sidebar.columns([4, 1])
        with col_w1:
            st.write(f"- {item}")
        with col_w2:
            if st.button("❌", key=f"remove_{item}", help="Remove from wishlist"):
                remove_from_wishlist(item)
    
    if st.sidebar.button("Clear All"):
        st.session_state.wishlist = []
        st.rerun()
else:
    st.sidebar.info("Your wishlist is empty.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 Categories")
categories = rec.get_all_categories()
selected_cat = st.sidebar.radio("Filter by:", ["All"] + categories)

# Handle Query Params for Image Clicks
if "view" in st.query_params:
    view_id = st.query_params["view"]
    try:
        view_id = int(view_id)
        st.session_state[f"details_{view_id}"] = True
        st.query_params.clear()
        st.rerun()
    except ValueError:
        pass

# --- Top Navigation & Search ---
st.markdown("<h1 style='text-align: center;'>Product Recommend System</h1>", unsafe_allow_html=True)
search_query = st.text_input("Search", placeholder="e.g. Laptop under 50000, Samsung Phone...", label_visibility="collapsed")

# --- Main Content ---

# Combined Filtering Logic
# Pass both search query and selected category to recommender
results = rec.recommend(search_query, category=selected_cat, top_n=12)

# Display Header
if search_query:
    if selected_cat != "All":
        st.subheader(f"Results for '{search_query}' in {selected_cat}")
    else:
        st.subheader(f"Results for '{search_query}'")
elif selected_cat != "All":
    st.subheader(f"Browsing {selected_cat}")
else:
    st.subheader("🔥 Trending Now")

# Display Results Grid
if not results.empty:
    # Create grid layout
    for i in range(0, len(results), 4):
        cols = st.columns(4)
        batch = results.iloc[i:i+4]
        
        for idx, (_, row) in enumerate(batch.iterrows()):
            with cols[idx]:
                # Card Container with Clickable Image
                # We use target="_self" to reload the page with the query param
                st.markdown(f"""
                <div class="product-card">
                    <a href="?view={row['product_id']}" target="_self" style="text-decoration: none; color: inherit;">
                        <img src="{row['image_url']}" style="width:100%; border-radius: 4px; height: 150px; object-fit: cover; margin-bottom: 10px; cursor: pointer;">
                    </a>
                    <h4 style="margin: 0; font-size: 1em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{row['name']}</h4>
                    <p style="font-size: 0.8em; color: #aaa; margin-bottom: 5px;">{row['brand']}</p>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span class="price-tag">₹{row['price']:,}</span>
                        <span class="rating-badge">★ {row['rating']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Action Buttons
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("👁️ View", key=f"view_{row['product_id']}"):
                        st.session_state[f"details_{row['product_id']}"] = True
                with c2:
                    if st.button("➕ Add", key=f"add_{row['product_id']}"):
                        add_to_wishlist(row['name'])
            
                # Detail View
                if st.session_state.get(f"details_{row['product_id']}", False):
                    with st.expander("Product Details", expanded=True):
                        st.write(f"**Brand:** {row['brand']}")
                        st.write(f"**Category:** {row['category']}")
                        st.write(f"**Description:** {row['description']}")
                        st.write(f"**Price:** ₹{row['price']:,}")
                        st.write(f"**Reviews:** {row['review_count']}")
                        if st.button("Close", key=f"close_{row['product_id']}"):
                            st.session_state[f"details_{row['product_id']}"] = False
                            st.rerun()

else:
    st.info("No products found. Try a different search or category.")
