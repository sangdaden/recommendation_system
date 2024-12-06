import streamlit as st
import pandas as pd
import pickle
 
# function cần thiết
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
 
# Hiển thị đề xuất ra bảng
def display_recommended_products(recommended_products, cols=5):
    for i in range(0, len(recommended_products), cols):
        cols = st.columns(cols)
        for j, col in enumerate(cols):
            if i + j < len(recommended_products):
                product = recommended_products.iloc[i + j]
                with col:  
                    st.write(product['ten_san_pham'])                    
                    expander = st.expander(f"Mô tả")
                    product_description = product['mo_ta']
                    truncated_description = ' '.join(product_description.split()[:100]) + '...'
                    expander.write(truncated_description)
                    expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")          
 
# Đọc dữ liệu sản phẩm
df_products = pd.read_csv('San_pham.csv')
# Lấy 10 sản phẩm
random_products = df_products.tail(n=10)
# print(random_products)
 
st.session_state.random_products = random_products
 
# Open and read file to cosine_sim_new
with open('products_cosine_sim.pkl', 'rb') as f:
    cosine_sim_new = pickle.load(f)
 
###### Giao diện Streamlit ######
st.image('hasaki_banner.jpg')
if "sidebar_visible" not in st.session_state:
    st.session_state.sidebar_visible = True  # Mặc định hiển thị sidebar
 
# # Hàm toggle để ẩn/hiện sidebar
# def toggle_sidebar():
#     st.session_state.sidebar_visible = not st.session_state.sidebar_visible
 
# # Nút để ẩn/hiện sidebar
# st.button("Toggle Menu", on_click=toggle_sidebar)
 
if st.session_state.sidebar_visible:
    menu = ["Business Objective", "Build Project", "New Prediction"]
    choice = st.sidebar.selectbox('Menu', menu)
    # st.write(f"You selected: {choice}")
else:
    st.write("Sidebar is hidden. Click the button to show the menu.")
st.sidebar.write("""#### Thành viên thực hiện:
                 Phan Thanh Sang & Tạ Quang Hưng""")
st.sidebar.write("""#### Giảng viên hướng dẫn:
                 Cô Khuất Thùy Phương""")
st.sidebar.write("""#### Thời gian thực hiện:
                 12/2024""")
if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### Classifying spam and ham messages is one of the most common natural language processing tasks for emails and chat engines. With the advancements in machine learning and natural language processing techniques, it is now possible to separate spam messages from ham messages with a high degree of accuracy.
    """)  
    st.write("""###### => Problem/ Requirement: Use Machine Learning algorithms in Python for ham and spam message classification.""")
    # st.image("ham_spam.jpg")
    # Kiểm tra xem 'selected_ma_san_pham' đã có trong session_state hay chưa
    if 'selected_ma_san_pham' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
        st.session_state.selected_ma_san_pham = None
 
    # Theo cách cho người dùng chọn sản phẩm từ dropdown
    # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    product_options = [(row['ten_san_pham'], row['ma_san_pham']) for index, row in st.session_state.random_products.iterrows()]
    st.session_state.random_products
    # Tạo một dropdown với options là các tuple này
    selected_product = st.selectbox(
        "Chọn sản phẩm",
        options=product_options,
        format_func=lambda x: x[0]  # Hiển thị tên sản phẩm
    )
    # Display the selected product
    st.write("Bạn đã chọn:", selected_product)
 
    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_ma_san_pham = selected_product[1]
 
    if st.session_state.selected_ma_san_pham:
        st.write("ma_san_pham: ", st.session_state.selected_ma_san_pham)
        # Hiển thị thông tin sản phẩm được chọn
        selected_product = df_products[df_products['ma_san_pham'] == st.session_state.selected_ma_san_pham]
 
        if not selected_product.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_product['ten_san_pham'].values[0])
 
            product_description = selected_product['mo_ta'].values[0]
            truncated_description = ' '.join(product_description.split()[:100])
            st.write('##### Thông tin:')
            st.write(truncated_description, '...')
 
            st.write('##### Các sản phẩm liên quan:')
            recommendations = get_recommendations(df_products, st.session_state.selected_ma_san_pham, cosine_sim=cosine_sim_new, nums=3)
            display_recommended_products(recommendations, cols=3)
        else:
            st.write(f"Không tìm thấy sản phẩm với ID: {st.session_state.selected_ma_san_pham}")
elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Some data")
    # st.dataframe(data[['v2', 'v1']].head(3))
    # st.dataframe(data[['v2', 'v1']].tail(3))  
    st.write("##### 2. Visualize Ham and Spam")
    # fig1 = sns.countplot(data=data[['v1']], x='v1')    
    # st.pyplot(fig1.figure)
 
    st.write("##### 3. Build model...")
    st.write("##### 4. Evaluation")
    # st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
    # st.code("Accuracy:"+str(round(acc,2)))
    st.write("###### Confusion matrix:")
    # st.code(cm)
    st.write("###### Classification report:")
    # st.code(cr)
    # st.code("Roc AUC score:" + str(round(roc,2)))
 
    # calculate roc curve
    st.write("###### ROC curve")
    # fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    # fig, ax = plt.subplots()      
    # ax.plot([0, 1], [0, 1], linestyle='--')
    # ax.plot(fpr, tpr, marker='.')
    # st.pyplot(fig)
 
    st.write("##### 5. Summary: This model is good enough for Ham vs Spam classification.")
 
elif choice == 'New Prediction':
    st.subheader("Select data")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)            
            lines = lines[0]    
            flag = True                          
    if type=="Input":        
        content = st.text_area(label="Input your content:")
        if content!="":
            # lines = np.array([content])
            flag = True
   
    if flag:
        st.write("Content:")
        if len(lines)>0:
            st.code(lines)        
            # x_new = count_model.transform(lines)        
            # y_pred_new = ham_spam_model.predict(x_new)      
            # st.code("New predictions (0: Ham, 1: Spam): " + str(y_pred_new))