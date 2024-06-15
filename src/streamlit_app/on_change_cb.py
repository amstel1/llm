import streamlit as st

# Create a placeholder for the result
result_placeholder = st.empty()

# Define a callback function that will be called when the radio button selection changes
def on_change_callback():
    # Get the selected value from the radio button
    selected_value = st.session_state.selected_value
    # Calculate the value to the power of 2
    result = selected_value ** 2
    # Update the result placeholder with the new result
    result_placeholder.write(f"The value {selected_value} to the power of 2 is: {result}")

# Create a radio button with options 12 and 24
st.radio(
    "Choose a number:",
    options=[12, 24],
    key='selected_value',
    on_change=on_change_callback
)

# Initial call to display the result for the default selected value (if needed)
on_change_callback()