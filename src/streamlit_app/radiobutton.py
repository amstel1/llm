import streamlit as st
import streamlit.components.v1 as components

# HTML and Tailwind CSS for beautiful radio buttons
html_code = """
<html>
    <head>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    </head>
    <body class="bg-gray-100">
        <div class="container mx-auto mt-10">
            <div class="max-w-md mx-auto bg-white rounded-xl shadow-md overflow-hidden md:max-w-2xl">
                <div class="md:flex">
                    <div class="p-8">
                        <div class="uppercase tracking-wide text-sm text-indigo-500 font-semibold">Choose an option</div>
                        <form id="radioForm">
                            <div class="mt-4">
                                <label class="inline-flex items-center">
                                    <input type="radio" class="form-radio h-5 w-5 text-indigo-600" name="option" value="12">
                                    <span class="ml-2 text-gray-700">12 месяцев</span>
                                </label>
                            </div>
                            <div class="mt-4">
                                <label class="inline-flex items-center">
                                    <input type="radio" class="form-radio h-5 w-5 text-indigo-600" name="option" value="24">
                                    <span class="ml-2 text-gray-700">24 месяца</span>
                                </label>
                            </div>
                            <div class="mt-4">
                                <label class="inline-flex items-center">
                                    <input type="radio" class="form-radio h-5 w-5 text-indigo-600" name="option" value="36">
                                    <span class="ml-2 text-gray-700">36 месяцев</span>
                                </label>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
        <script>
            const form = document.getElementById('radioForm');
            form.addEventListener('change', (event) => {
                const selectedOption = event.target.value;
                fetch(`${window.location.origin}/_stcore/report?name=selected_option&value=${selectedOption}`, { method: 'GET' });
            });
        </script>
    </body>
</html>
"""

components.html(html_code, height=400)

# Display the selected option in Streamlit
selected_option = st.query_params.get('selected_option', [''])[0]
st.write(f"Selected Option: {selected_option}")