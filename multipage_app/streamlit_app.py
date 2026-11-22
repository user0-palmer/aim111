import streamlit as st

page = st.navigation(
    [
        st.Page("app_pages/hello.py", title="Hello", icon="👋"),
        st.Page("app_pages/ml_playground.py", title="ML Playground", icon="🤖"),
        st.Page("app_pages/plotting_demo.py", title="Plotting demo", icon="📈"),
        st.Page("app_pages/mapping_demo.py", title="Mapping demo", icon="🌍"),
        st.Page("app_pages/dataframe_demo.py", title="DataFrame demo", icon="📊"),
    ]
)
page.run()
