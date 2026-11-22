# Streamlit Multipage App Example

This project implements the official Streamlit tutorial: [Create a multipage app](https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app), enhanced with an **Interactive Machine Learning Playground**.

---

## 1. Directory Structure

```text
multipage_app/
├── app_pages/
│   ├── hello.py            # Welcome & documentation overview
│   ├── ml_playground.py    # 🤖 Interactive Live Machine Learning Playground
│   ├── plotting_demo.py    # Animated real-time line chart demo
│   ├── mapping_demo.py     # 3D Pydeck geospatial mapping demo
│   └── dataframe_demo.py   # UN agricultural production dataframe demo
├── requirements.txt
├── README.md
└── streamlit_app.py        # Entrypoint defining st.Page and st.navigation
```

---

## 2. Features in `ML Playground` (Option 1)

- **Datasets:** Moons (non-linear), Concentric Circles, Linearly Separable, Multi-Cluster Blobs, Iris Petals, and Wine Recognition (2D PCA).
- **Classifiers:** Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Logistic Regression, and Decision Trees.
- **Real-Time Interactive Decision Boundary:** Dynamic Plotly contour mesh showing decision regions or continuous probability gradients, overlaying training and test instances.
- **Diagnostic Analytics:** Confusion Matrix heatmap, ROC Curve with AUC score, feature importance / coefficient weights, and test prediction confidence table.

---

## 3. Running the App

Activate your Python virtual environment and run the entrypoint file:

```bash
/run/media/josh/75D7-DC5F/School/AIM111/git/aim111/.venv/bin/streamlit run /run/media/josh/75D7-DC5F/School/AIM111/git/aim111/multipage_app/streamlit_app.py
```
