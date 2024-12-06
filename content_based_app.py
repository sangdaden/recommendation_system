import pandas as pd
import streamlit as st
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Hàm tải dữ liệu
@st.cache
def load_data():
    san_pham = pd.read_csv("san_pham.csv")
    danh_gia = pd.read_csv("danh_gia.csv")
    khach_hang = pd.read_csv("khach_hang.csv")
    return san_pham, danh_gia, khach_hang

san_pham, danh_gia, khach_hang = load_data()

# Gợi ý dựa trên sản phẩm (Content-based)
def content_based_recommendation(product_name, san_pham):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(san_pham["mo_ta"])
    cosine_sim = cosine_similarity(tfidf_matrix)
    indices = pd.Series(san_pham.index, index=san_pham["ten_san_pham"]).drop_duplicates()

    if product_name not in indices:
        return "Không tìm thấy sản phẩm!", []

    idx = indices[product_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 sản phẩm tương tự
    product_indices = [i[0] for i in sim_scores]
    return san_pham.iloc[product_indices]

# Gợi ý dựa trên lịch sử người dùng (Collaborative Filtering)
def collaborative_filtering(user_id, danh_gia, san_pham):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(danh_gia[["ma_khach_hang", "ma_san_pham", "so_sao"]], reader)
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)

    user_rated_products = danh_gia[danh_gia["ma_khach_hang"] == user_id]["ma_san_pham"]
    recommendations = []
    for product_id in san_pham["ma_san_pham"]:
        if product_id not in user_rated_products.values:
            pred = algo.predict(user_id, product_id)
            recommendations.append((product_id, pred.est))

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]
    recommended_products = san_pham[san_pham["ma_san_pham"].isin([rec[0] for rec in recommendations])]
    return recommended_products

# Giao diện Streamlit với Tabs
st.title("💄 Beauty Product Recommendation System")
st.write("Chọn chế độ gợi ý sản phẩm phù hợp!")

# Tabs để lựa chọn chế độ gợi ý
tabs = st.tabs(["🔍 Gợi ý theo sản phẩm", "👤 Gợi ý theo lịch sử người dùng", "🔥 Gợi ý theo xu hướng", "📦 Gợi ý theo nhóm sản phẩm"])

# Tab 1: Gợi ý theo sản phẩm
with tabs[0]:
    st.subheader("🔍 Gợi ý theo sản phẩm")
    product_name = st.selectbox("Chọn sản phẩm bạn thích:", san_pham["ten_san_pham"].unique())
    
    # Select box để chọn lọc theo giá bán hoặc điểm trung bình
    filter_criteria = st.radio("Chọn tiêu chí lọc:", ("Giá bán", "Điểm trung bình"), key="filter_criteria_product")
    
    if filter_criteria == "Giá bán":
        # Slider để chọn giá bán với key duy nhất
        min_price, max_price = st.slider("Chọn khoảng giá bán", 
                                          min_value=int(san_pham["gia_ban"].min()), 
                                          max_value=int(san_pham["gia_ban"].max()), 
                                          value=(int(san_pham["gia_ban"].min()), int(san_pham["gia_ban"].max())),
                                          key="price_slider_product")
        min_rating, max_rating = 1, 5  # Không cần slider cho điểm trung bình

    elif filter_criteria == "Điểm trung bình":
        # Slider để chọn điểm trung bình
        min_rating, max_rating = st.slider("Chọn khoảng điểm trung bình", 
                                           min_value=1, 
                                           max_value=5, 
                                           value=(1, 5),
                                           key="rating_slider_product")
        min_price, max_price = int(san_pham["gia_ban"].min()), int(san_pham["gia_ban"].max())  # Không cần slider cho giá bán

    if st.button("Gợi ý sản phẩm (theo sản phẩm)", key="product_button"):
        if product_name:
            recommendations = content_based_recommendation(product_name, san_pham)
            if isinstance(recommendations, str):  # Nếu không tìm thấy sản phẩm
                st.warning(recommendations)
            else:
                # Lọc sản phẩm theo điều kiện đã chọn
                filtered_recommendations = recommendations[
                    (recommendations["gia_ban"] >= min_price) & 
                    (recommendations["gia_ban"] <= max_price) & 
                    (recommendations["diem_trung_binh"] >= min_rating) &
                    (recommendations["diem_trung_binh"] <= max_rating)
                ]
                if filtered_recommendations.empty:
                    st.warning("Không có sản phẩm nào phù hợp với điều kiện lọc.")
                else:
                    st.write("### Gợi ý các sản phẩm tương tự:")
                    # Hiển thị kết quả dưới dạng bảng
                    filtered_recommendations = filtered_recommendations[["ten_san_pham", "gia_ban", "gia_goc", "diem_trung_binh"]]
                    st.dataframe(filtered_recommendations)
        else:
            st.error("Vui lòng chọn sản phẩm!")

# Tab 2: Gợi ý theo lịch sử người dùng
with tabs[1]:
    st.subheader("👤 Gợi ý theo lịch sử người dùng")
    user_id = st.number_input("Nhập mã khách hàng:", min_value=1, step=1, value=1)
    
    # Select box để chọn lọc theo giá bán hoặc điểm trung bình
    filter_criteria = st.radio("Chọn tiêu chí lọc:", ("Giá bán", "Điểm trung bình"), key="filter_criteria_user")
    
    if filter_criteria == "Giá bán":
        # Slider để chọn giá bán với key duy nhất
        min_price, max_price = st.slider("Chọn khoảng giá bán", 
                                          min_value=int(san_pham["gia_ban"].min()), 
                                          max_value=int(san_pham["gia_ban"].max()), 
                                          value=(int(san_pham["gia_ban"].min()), int(san_pham["gia_ban"].max())),
                                          key="price_slider_user")
        min_rating, max_rating = 1, 5  # Không cần slider cho điểm trung bình

    elif filter_criteria == "Điểm trung bình":
        # Slider để chọn điểm trung bình
        min_rating, max_rating = st.slider("Chọn khoảng điểm trung bình", 
                                           min_value=1, 
                                           max_value=5, 
                                           value=(1, 5),
                                           key="rating_slider_user")
        min_price, max_price = int(san_pham["gia_ban"].min()), int(san_pham["gia_ban"].max())  # Không cần slider cho giá bán

    if st.button("Gợi ý sản phẩm (theo người dùng)", key="user_button"):
        recommendations = collaborative_filtering(user_id, danh_gia, san_pham)
        if recommendations.empty:
            st.warning("Không tìm thấy dữ liệu đánh giá của khách hàng này.")
        else:
            # Lọc sản phẩm theo điều kiện đã chọn
            filtered_recommendations = recommendations[
                (recommendations["gia_ban"] >= min_price) & 
                (recommendations["gia_ban"] <= max_price) & 
                (recommendations["diem_trung_binh"] >= min_rating) &
                (recommendations["diem_trung_binh"] <= max_rating)
            ]
            if filtered_recommendations.empty:
                st.warning("Không có sản phẩm nào phù hợp với điều kiện lọc.")
            else:
                st.write(f"### Gợi ý sản phẩm cho Khách Hàng ID: {user_id}")
                # Hiển thị kết quả dưới dạng bảng
                filtered_recommendations = filtered_recommendations[["ten_san_pham", "gia_ban", "gia_goc", "diem_trung_binh"]]
                st.dataframe(filtered_recommendations)

# Tab 3: Gợi ý theo xu hướng
with tabs[2]:
    st.subheader("🔥 Gợi ý theo xu hướng")

# Tab 4: Gợi ý theo nhóm sản phẩm
with tabs[3]:
    st.subheader("📦 Gợi ý theo nhóm sản phẩm")