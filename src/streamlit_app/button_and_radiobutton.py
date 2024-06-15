import streamlit as st

item = {'a':'a', 'b':'b'}

# Tailwind CSS for styling
tailwind_css = """
<html> 
<head> 
  <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet"> 
</head> 
</html>
"""
st.markdown(tailwind_css, unsafe_allow_html=True)

# Radio button options
options = {
    "12 месяцев": "12",
    "24 месяца": "24",
    "36 месяцев": "36"
}

# Display the radio buttons using Streamlit
selected_option = st.radio(
    "Choose an option",
    list(options.keys()),
    format_func=lambda x: x
)

# Get the value corresponding to the selected option
selected_value = options[selected_option]

# Display the selected option value
st.write(f"Selected Option: {selected_value**2}")
st.markdown(item.get('a'))