"""
Entrypoint for Streamlit Community Cloud (streamlit.io).
Allows running either `streamlit run app.py` or `streamlit run streamlit_app.py`.
"""
from pathlib import Path
import runpy

app_file = Path(__file__).parent / "app.py"
runpy.run_path(str(app_file), run_name="__main__")
