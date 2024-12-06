import pandas as pd
import pickle
 
# Open and read file to cosine_sim_new
with open('products_cosine_sim.pkl', 'rb') as f:
    cosine_sim_new = pickle.load(f)

# Đọc dữ liệu sản phẩm
df_products = pd.read_csv('./data/San_pham.csv')
# Lấy 10 sản phẩm
random_products = df_products.tail(n=10)
# print(random_products)

def get_recommendations(df, ma_san_pham, cosine_sim, nums=5):
    # Get the index of the product that matches the ma_san_pham
    matching_indices = df.index[df['ma_san_pham'] == ma_san_pham].tolist()
    if not matching_indices:
        print(f"No product found with ID: {ma_san_pham}")
        return pd.DataFrame()  # Return an empty DataFrame if no match
    idx = matching_indices[0]
 
    # Get the pairwise similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim[idx]))
 
    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
 
    # Get the scores of the nums most similar products (Ignoring the product itself)
    sim_scores = sim_scores[1:nums+1]
 
    # Get the product indices
    product_indices = [i[0] for i in sim_scores]
 
    # Return the top n most similar products as a DataFrame
    return df.iloc[product_indices]

recommendations = get_recommendations(df_products, 318900012, cosine_sim=cosine_sim_new, nums=3)