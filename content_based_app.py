import pandas as pd
import streamlit as st
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Hàm tải dữ liệu
@st.cache_data 
def load_data():
    san_pham = pd.read_csv("San_pham.csv")
    danh_gia = pd.read_csv("Danh_gia.csv")
    khach_hang = pd.read_csv("Khach_hang.csv")
    return san_pham, danh_gia, khach_hang

san_pham, danh_gia, khach_hang = load_data()

# Lấy thêm sản phẩm khách hàng đã mua
def get_customer_purchased_products(user_id, danh_gia, san_pham):
    # Lọc danh sách các sản phẩm đã mua bởi khách hàng
    purchased_products = danh_gia[danh_gia["ma_khach_hang"] == user_id]
    
    if purchased_products.empty:
        return "Khách hàng chưa mua sản phẩm nào.", pd.DataFrame()
    
    # Kết hợp thông tin sản phẩm từ bảng san_pham
    purchased_products_details = pd.merge(
        purchased_products,
        san_pham,
        on="ma_san_pham",
        how="inner"
    )
    
    # Trả về thông tin chi tiết các sản phẩm đã mua
    return purchased_products_details


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
def collaborative_filtering(user_id, danh_gia, san_pham, khach_hang):
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

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:5]
    recommended_products = san_pham[san_pham["ma_san_pham"].isin([rec[0] for rec in recommendations])]

    # Lấy thông tin khách hàng từ bảng 'khach_hang'
    customer_info = khach_hang[khach_hang["ma_khach_hang"] == user_id]
    customer_name = customer_info["ho_ten"].iloc[0] if not customer_info.empty else "Khách hàng không xác định"

    return customer_name, recommended_products

# Giao diện Streamlit với Tabs
# st.image('hasaki_banner_2.jpg')
st.title("💄🧴 🧺 Hasaki's Product Recommendation System")
# st.write("Chọn chế độ gợi ý sản phẩm phù hợp!")

menu = ["Đặt Vấn Đề", "Hướng Giải Quyết", "Hệ Thống Đề Xuất", "Về Nhóm Thực Hiện"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 Phan Thanh Sang & 
                 Tạ Quang Hưng""")
st.sidebar.write("""#### Giảng viên hướng dẫn:
                 Cô Khuất Thùy Phương""")
st.sidebar.write("""#### Thời gian thực hiện:
                 12/2024""")

if choice == 'Đặt Vấn Đề':
    st.header("🧑‍💻 Tổng quan vấn đề cho Hasaki")
    # Phần giới thiệu
    st.subheader("Giới thiệu")
    st.write("""
        Cửa hàng mỹ phẩm cần giúp khách hàng dễ dàng tìm thấy sản phẩm phù hợp trong hàng ngàn lựa chọn. 
        Hệ thống gợi ý (Recommendation System) mang lại lợi ích lớn bằng cách cá nhân hóa trải nghiệm mua sắm, giảm thời gian tìm kiếm, và tăng doanh số.
    """)
    # Các vấn đề gặp phải
    st.subheader("Vấn đề cần giải quyết")
    st.write("""
        1. **Số lượng sản phẩm đa dạng**: Khách hàng khó chọn sản phẩm phù hợp.  
        2. **Thiếu cá nhân hóa**: Sản phẩm không đáp ứng đúng nhu cầu từng người.  
        3. **Khó khám phá sản phẩm mới**: Cơ hội quảng bá bị bỏ lỡ.  
    """)

    # Mục tiêu hệ thống
    st.subheader("Mục tiêu")
    st.write("""
        1. Cá nhân hóa gợi ý để tăng sự hài lòng của khách hàng.  
        2. Hỗ trợ khám phá sản phẩm mới, cải thiện doanh số.  
        3. Giữ chân khách hàng thông qua trải nghiệm mua sắm tốt hơn.  
    """)

    st.write("Với hệ thống gợi ý, cửa hàng mỹ phẩm có thể tối ưu hóa trải nghiệm khách hàng và đạt được mục tiêu kinh doanh bền vững.")

elif choice == 'Hướng Giải Quyết':
    st.header("💪 Hướng xử lý vấn đề")
    # Tabs để chia trang
    tab1, tab2 = st.tabs(["Collaborative Filtering (SVD)", "Content-Based Filtering"])
    # ----------- Collaborative Filtering ----------- #
    with tab1:
        st.subheader("Collaborative Filtering")
        st.write("""
        **Collaborative Filtering** gợi ý sản phẩm dựa trên hành vi của người dùng. 
        Phương pháp sử dụng dữ liệu lịch sử đánh giá để dự đoán điểm số sản phẩm chưa tương tác.
        
        **Quy trình:**
        1. Thu thập lịch sử tương tác (đánh giá, mua hàng).
        2. Áp dụng SVD để phân rã ma trận tương tác.
        3. Dự đoán điểm số và gợi ý sản phẩm có điểm cao nhất.

        **Ưu điểm:** Khám phá sản phẩm mới.  
        **Hạn chế:** Gặp vấn đề "lạnh" với người dùng hoặc sản phẩm mới.
        """)

    # ----------- Content-Based Filtering ----------- #
    with tab2:
        st.subheader("Content-Based Filtering")
        st.write("""
        **Content-Based Filtering** gợi ý dựa trên đặc điểm của sản phẩm và sở thích người dùng. 
        Sử dụng độ tương đồng (cosine similarity) giữa các sản phẩm để gợi ý.

        **Quy trình:**
        1. Thu thập dữ liệu sản phẩm (mô tả, đặc điểm).
        2. Tính độ tương đồng bằng cosine similarity.
        3. Gợi ý các sản phẩm có độ tương đồng cao nhất.

        **Ưu điểm:** Cá nhân hóa gợi ý tốt.  
        **Hạn chế:** Hạn chế trong việc khám phá sản phẩm mới.
    """)
    # Kết luận
    st.header("Kết luận")
    st.write("""
        Cả hai phương pháp đều hữu ích:
        - **Collaborative Filtering**: Tốt cho khám phá sản phẩm mới.
        - **Content-Based Filtering**: Tốt cho cá nhân hóa.  

        Kết hợp cả hai để tăng hiệu quả gợi ý.
    """)

elif choice == 'Hệ Thống Đề Xuất':   
    # Tabs để lựa chọn chế độ gợi ý
    tabs = st.tabs(["🔍 Gợi ý theo sản phẩm", "👤 Gợi ý theo lịch sử người dùng"])

    # Tab 1: Gợi ý theo sản phẩm
    with tabs[0]:
        st.subheader("🔍 Gợi ý theo sản phẩm (Content Based)")
        product_name = st.selectbox("Chọn sản phẩm bạn thích:", san_pham["ten_san_pham"].unique())
        
        # Lựa chọn tiêu chí lọc
        filter_criteria = st.radio("Chọn tiêu chí lọc:", ("Giá bán", "Điểm trung bình"), key="filter_criteria_product")
        
        if filter_criteria == "Giá bán":
            min_price, max_price = st.slider(
                "Chọn khoảng giá bán",
                min_value=int(san_pham["gia_ban"].min()),
                max_value=int(san_pham["gia_ban"].max()),
                value=(int(san_pham["gia_ban"].min()), int(san_pham["gia_ban"].max())),
                key="price_slider_product"
            )
            min_rating, max_rating = 1, 5  # Không cần lọc điểm trung bình

        elif filter_criteria == "Điểm trung bình":
            min_rating, max_rating = st.slider(
                "Chọn khoảng điểm trung bình",
                min_value=1,
                max_value=5,
                value=(1, 5),
                key="rating_slider_product"
            )
            min_price, max_price = int(san_pham["gia_ban"].min()), int(san_pham["gia_ban"].max())  # Không cần lọc giá bán

        if st.button("Gợi ý sản phẩm (theo sản phẩm)", key="product_button"):
            if product_name:
                recommendations = content_based_recommendation(product_name, san_pham)
                if isinstance(recommendations, str):  # Nếu không tìm thấy sản phẩm
                    st.warning(recommendations)
                else:
                    # Lọc sản phẩm theo tiêu chí
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
                        # Biến đếm để xác định 3 sản phẩm đầu tiên
                        count = 0

                        # Hiển thị danh sách sản phẩm
                        for index, row in filtered_recommendations.iterrows():
                            # Thêm icon ngọn lửa cho 3 sản phẩm đầu tiên
                            if count < 3:
                                product_title = f"🔥 {row['ten_san_pham']} - Giá: {row['gia_ban']} - ⭐: {row['diem_trung_binh']}"
                            else:
                                product_title = f"{row['ten_san_pham']} - Giá: {row['gia_ban']} - ⭐: {row['diem_trung_binh']}"

                            # Hiển thị sản phẩm dưới dạng expander
                            with st.expander(product_title):
                                st.write(f"**Mô tả:** {row['mo_ta']}")
                                st.write(f"**Điểm trung bình:** {row['diem_trung_binh']}")

                            # Tăng bộ đếm sau khi hiển thị một sản phẩm
                            count += 1

    # Tab 2: Gợi ý theo lịch sử người dùng
    with tabs[1]:
        st.subheader("👤 Gợi ý theo lịch sử người dùng (Collaborative Filtering)")

        # Tạo danh sách dropdown từ bảng khach_hang
        customer_options = khach_hang.apply(
            lambda row: f"ID:{row['ma_khach_hang']}, Tên khách hàng: {row['ho_ten']}", axis=1
        ).tolist()

        # Dropdown menu để chọn khách hàng
        selected_customer = st.selectbox("Chọn khách hàng:", options=customer_options)

        # Tách `user_id` từ lựa chọn
        user_id = int(selected_customer.split(",")[0].split(":")[1])  # Lấy phần ID từ chuỗi

        # Lựa chọn tiêu chí lọc
        filter_criteria = st.radio("Chọn tiêu chí lọc:", ("Giá bán", "Điểm trung bình"), key="filter_criteria_user")

        if filter_criteria == "Giá bán":
            min_price, max_price = st.slider(
                "Chọn khoảng giá bán",
                min_value=int(san_pham["gia_ban"].min()),
                max_value=int(san_pham["gia_ban"].max()),
                value=(int(san_pham["gia_ban"].min()), int(san_pham["gia_ban"].max())),
                key="price_slider_user"
            )
            min_rating, max_rating = 1, 5  # Không cần lọc điểm trung bình

        elif filter_criteria == "Điểm trung bình":
            min_rating, max_rating = st.slider(
                "Chọn khoảng điểm trung bình",
                min_value=1,
                max_value=5,
                value=(1, 5),
                key="rating_slider_user"
            )
            min_price, max_price = int(san_pham["gia_ban"].min()), int(san_pham["gia_ban"].max())  # Không cần lọc giá bán

        if st.button("Xem sản phẩm đã mua", key="view_purchased_products"):
            purchased_products = get_customer_purchased_products(user_id, danh_gia, san_pham)
            
            if isinstance(purchased_products, str):
                st.warning(purchased_products)
            else:
                st.write(f"### Các sản phẩm đã mua bởi Khách Hàng ID: {user_id}")
                for index, row in purchased_products.iterrows():
                    product_title = f"{row['ten_san_pham']} - Giá: {row['gia_ban']} - ⭐: {row['diem_trung_binh']}"
                    with st.expander(product_title):
                        st.write(f"**Mô tả:** {row['mo_ta']}")
                        st.write(f"**Ngày bình luận:** {row['ngay_binh_luan']}")
                        st.write(f"**Nội dung bình luận:** {row['noi_dung_binh_luan']}")


        if st.button("Gợi ý sản phẩm (theo người dùng)", key="user_button"):
            customer_name, recommendations = collaborative_filtering(user_id, danh_gia, san_pham, khach_hang)

            if recommendations.empty:
                st.warning("Không tìm thấy dữ liệu đánh giá của khách hàng này.")
            else:
                # Lọc sản phẩm theo tiêu chí
                filtered_recommendations = recommendations[
                    (recommendations["gia_ban"] >= min_price) &
                    (recommendations["gia_ban"] <= max_price) &
                    (recommendations["diem_trung_binh"] >= min_rating) &
                    (recommendations["diem_trung_binh"] <= max_rating)
                ]

                if filtered_recommendations.empty:
                    st.warning("Không có sản phẩm nào phù hợp với điều kiện lọc.")
                else:
                    # Hiển thị thông tin khách hàng
                    st.write(f"### Gợi ý sản phẩm cho Khách Hàng ID: {user_id} - Tên: {customer_name}")

                    # Biến đếm để xác định 3 sản phẩm đầu tiên
                    count = 0

                    # Hiển thị danh sách sản phẩm
                    for index, row in filtered_recommendations.iterrows():
                        # Thêm icon ngọn lửa cho 3 sản phẩm đầu tiên
                        if count < 3:
                            product_title = f"🔥 {row['ten_san_pham']} - Giá: {row['gia_ban']} - ⭐: {row['diem_trung_binh']}"
                        else:
                            product_title = f"{row['ten_san_pham']} - Giá: {row['gia_ban']} - ⭐: {row['diem_trung_binh']}"

                        # Hiển thị sản phẩm dưới dạng expander
                        with st.expander(product_title):
                            st.write(f"**Mô tả:** {row['mo_ta']}")
                            st.write(f"**Điểm trung bình:** {row['diem_trung_binh']}")

                        # Tăng bộ đếm sau khi hiển thị một sản phẩm
                        count += 1


elif choice == 'Về Nhóm Thực Hiện': 
    tabs = st.tabs(["📃 Thông tin khóa học", "👨‍💻 Thành viên trong nhóm"])

    # Tab 1: Gợi ý theo sản phẩm
    with tabs[0]:
        st.title("Đồ Án Tốt Nghiệp Data Science & Machine Learning")
        st.subheader("Trung Tâm Tin Học - Trường Đại Học Khoa Học Tự Nhiên")

        # Hiển thị thông tin chi tiết
        st.write("**Nhóm:** Nhóm 7 - Tối thứ 7")
        st.write("**Khóa:** DL07_K299 - ONLINE")
        st.write("**Học viên thực hiện:** Phan Thanh Sang & Tạ Quang Hưng")
        st.write("**Giảng viên hướng dẫn:** Cô Khuất Thùy Phương")
        st.write("**Thời gian thực hiện:** 18/11/2024 - 16/12/2024")

    with tabs[1]:
        # Thêm 2 cột để hiển thị hình ảnh
        col1, col2 = st.columns(2)

        # Hiển thị hình trong cột 1
        with col1:
            st.image('sang.png', caption="Phan Thanh Sang", use_container_width =True)

        # Hiển thị hình trong cột 2
        with col2:
            st.image('hung.png', caption="Tạ Quang Hưng", use_container_width =True)
    # Phần liên hệ
    st.header("Liên hệ")
    st.write("""
        Nếu bạn cần góp ý hãy liên hệ:
        - **Email**: sang.phan2409@gmail.com
    """)
    