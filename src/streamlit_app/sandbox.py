import streamlit as st

products = [
    {"name": "Product 1", "price": "$10", "image_url": "https://via.placeholder.com/150"},
    {"name": "Product 2", "price": "$20", "image_url": "https://via.placeholder.com/150"},
    {"name": "Product 3", "price": "$30", "image_url": "https://via.placeholder.com/150"},
]

def sort_products(products, criterion):
    print(criterion, type(products))
    print(products[0])
    return products

def create_grid(products):
    with st.container():
        grid_container = st.container()
        for i in range(4):
            cols = st.columns(5)
            with grid_container:
                for j, product in enumerate(products[i*5:(i+1)*5]):
                    with cols[j]:
                        st.image(product["image_url"], width=100)
                        st.write(f"**{product['name']}**")
                        st.write(product["price"])

    with st.container():
        _, inline_col0, inline_col1, inline_col2, _ = st.columns([0.01, 0.05, 0.05, 0.05, 0.09])
        with inline_col0:
            st.button("inline_col0", key=f"inline_col0_key")
        with inline_col1:
            st.button("inline_col1", key=f"inline_col1_key")
        with inline_col2:
            st.button("inline_col2", key=f"inline_col2_key")
        if prompt := st.chat_input("Enter your question here"):
            prompt = prompt.strip()


# main part
st.title("Internet Shop")
with st.expander(label="Instruction"):
    st.write("Here comes the instruction")

# сортировка
with st.container():
    sort_criterion = st.radio("Сортировка:", ["Сначала дороже", "Сначала дешевле", "По отзывам", "По популярности"], horizontal=True)

# inline filters


sorted_products = sort_products(products, sort_criterion)

create_grid(sorted_products)
