import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from sklearn.datasets import (
    make_moons,
    make_circles,
    make_classification,
    make_blobs,
    load_iris,
    load_wine,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)

# ---------------------------------------------------------
# Page Title & Introduction
# ---------------------------------------------------------
st.markdown("# 🤖 Interactive Machine Learning Playground")
st.markdown(
    """
    **Welcome to the AIM111 Interactive Classifier Lab!**
    This application allows you to explore how **Supervised Machine Learning Algorithms** learn from data, 
    map decision boundaries, and make predictions on unseen test cases in real-time.
    """
)

with st.expander("📖 **What is this playground and what is it doing? (Click to expand)**", expanded=False):
    st.markdown(
        """
        ### The Supervised Learning Workflow
        In supervised classification, a model is given historical examples consisting of **features** (inputs, $X_1, X_2$) 
        and corresponding **labels** (targets, $y$). The goal of the algorithm is to learn a mathematical function 
        $f(X_1, X_2) \to y$ that correctly predicts the label for new, previously unseen points.

        **Here is what happens behind the scenes every time you interact with this page:**
        1. **Data Ingestion & Sampling:** The app generates or loads a 2D dataset based on your sidebar settings.
        2. **Train/Test Splitting:** The data is divided into two sets:
           - **Training Set (circles):** The data the model is allowed to study and learn from.
           - **Test Set (diamonds):** A held-out evaluation set the model has never seen, used to grade its real-world generalization ability.
        3. **Model Training (Fitting):** The chosen algorithm fits its mathematical parameters (weights, decision nodes, or support vectors) to the training data.
        4. **Decision Boundary Surface Generation:** The app scans tens of thousands of coordinate points across a fine 2D grid, asks the model to predict the class or probability at every coordinate, and paints the colored decision contour.
        5. **Evaluation:** The model is tested against the held-out test points to calculate accuracy, precision, recall, confusion matrix, and ROC curves.
        """
    )

st.markdown("---")

# ---------------------------------------------------------
# Sidebar Controls & Hyperparameters
# ---------------------------------------------------------
st.sidebar.markdown("## 🎛️ Control Panel")
st.sidebar.markdown("### 1. Dataset Configuration")

dataset_name = st.sidebar.selectbox(
    "Choose Dataset:",
    [
        "Moons (Non-Linear) 🌙",
        "Circles (Concentric) 🔘",
        "Linearly Separable 📦",
        "Multi-Cluster Blobs 🌀",
        "Iris Flowers (Petals) 🌸",
        "Wine Recognition (2D PCA) 🍷",
    ],
    index=0,
)

# Dataset Generation / Loading logic with metadata tracking
dataset_meta = {}

if "Moons" in dataset_name:
    n_samples = st.sidebar.slider("Number of Samples (N):", 100, 1000, 350, step=50)
    noise = st.sidebar.slider("Noise Level (Dispersion):", 0.05, 0.50, 0.20, step=0.05)
    random_state = st.sidebar.number_input("Random Seed (random_state):", value=42, step=1)
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    target_names = ["Class 0 (Upper Crescent)", "Class 1 (Lower Crescent)"]
    feat_names = ["Coordinate X₁ (Horizontal)", "Coordinate X₂ (Vertical)"]
    dataset_meta = {
        "source": "Scikit-Learn Synthetic Generator (`make_moons`)",
        "description": "Two interlocking crescent/half-moon shapes positioned in 2D Euclidean space.",
        "geometry": "Highly non-linear. A straight line cannot separate the two arcs without significant error.",
        "challenge": "Tests whether an algorithm can follow smooth geometric curvature without getting confused by the interlocking gap.",
        "is_synthetic": True,
        "n_samples": n_samples,
        "noise": noise,
        "seed": random_state,
    }

elif "Circles" in dataset_name:
    n_samples = st.sidebar.slider("Number of Samples (N):", 100, 1000, 350, step=50)
    noise = st.sidebar.slider("Noise Level (Dispersion):", 0.02, 0.30, 0.12, step=0.02)
    factor = st.sidebar.slider("Inner/Outer Radius Ratio:", 0.2, 0.8, 0.5, step=0.05)
    random_state = st.sidebar.number_input("Random Seed (random_state):", value=42, step=1)
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    target_names = ["Outer Ring (Class 0)", "Inner Core (Class 1)"]
    feat_names = ["Radial Dimension X₁", "Radial Dimension X₂"]
    dataset_meta = {
        "source": "Scikit-Learn Synthetic Generator (`make_circles`)",
        "description": "A smaller circular cluster completely enclosed inside a larger concentric outer circular ring.",
        "geometry": "Radially symmetric and non-linearly separable in 2D Euclidean space.",
        "challenge": "Linear models (Logistic Regression, Linear SVM) will fail completely (~50% accuracy). Tests kernel transformations (e.g. RBF) or radial distance metrics (KNN).",
        "is_synthetic": True,
        "n_samples": n_samples,
        "noise": noise,
        "seed": random_state,
        "extra_param": f"Inner/Outer Ratio: {factor}",
    }

elif "Linearly Separable" in dataset_name:
    n_samples = st.sidebar.slider("Number of Samples (N):", 100, 1000, 350, step=50)
    random_state = st.sidebar.number_input("Random Seed (random_state):", value=42, step=1)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=random_state,
        n_clusters_per_class=1,
    )
    target_names = ["Class 0 (Lower Cluster)", "Class 1 (Upper Cluster)"]
    feat_names = ["Feature X₁", "Feature X₂"]
    dataset_meta = {
        "source": "Scikit-Learn Synthetic Generator (`make_classification`)",
        "description": "Two distinct Gaussian clusters separated by an open linear hyperplane margin.",
        "geometry": "Linearly separable.",
        "challenge": "Baseline benchmark. Tests whether complex non-linear models overfit and create unnecessarily convoluted boundaries on a simple linear problem.",
        "is_synthetic": True,
        "n_samples": n_samples,
        "noise": "N/A (Standard Gaussian)",
        "seed": random_state,
    }

elif "Multi-Cluster" in dataset_name:
    n_samples = st.sidebar.slider("Number of Samples (N):", 100, 1000, 400, step=50)
    n_centers = st.sidebar.slider("Number of Clusters (Classes):", 2, 4, 3, step=1)
    cluster_std = st.sidebar.slider("Cluster Dispersion (Spread):", 0.5, 2.5, 1.2, step=0.1)
    random_state = st.sidebar.number_input("Random Seed (random_state):", value=42, step=1)
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    target_names = [f"Cluster Group {i}" for i in range(n_centers)]
    feat_names = ["Spatial Dimension X₁", "Spatial Dimension X₂"]
    dataset_meta = {
        "source": "Scikit-Learn Synthetic Generator (`make_blobs`)",
        "description": f"{n_centers} distinct multi-modal Gaussian clusters distributed in 2D space.",
        "geometry": "Multi-class spatial distribution with variable inter-cluster overlap.",
        "challenge": "Tests multi-class classification handling (One-vs-Rest vs Multinomial) and boundary behavior where 3 or more classes meet at a junction.",
        "is_synthetic": True,
        "n_samples": n_samples,
        "noise": f"Dispersion σ = {cluster_std}",
        "seed": random_state,
        "extra_param": f"Clusters: {n_centers}",
    }

elif "Iris" in dataset_name:
    iris = load_iris()
    X = iris.data[:, 2:4]  # Petal Length & Petal Width
    y = iris.target
    target_names = list(iris.target_names)
    feat_names = ["Petal Length (cm)", "Petal Width (cm)"]
    dataset_meta = {
        "source": "Fisher's Famous 1936 Botanical Benchmark (`load_iris`)",
        "description": "150 physical floral specimens collected by botanist Edgar Anderson in the Gaspé Peninsula, Quebec.",
        "geometry": "Three biological species: Iris Setosa, Iris Versicolor, and Iris Virginica.",
        "challenge": "Setosa is cleanly separable, but Versicolor and Virginica overlap at their botanical boundary, testing realistic biological classification.",
        "is_synthetic": False,
        "n_samples": 150,
        "noise": "Real-world biological variation",
        "seed": "Fixed historical dataset",
    }

else:  # Wine
    wine = load_wine()
    pca = PCA(n_components=2, random_state=42)
    X = pca.fit_transform(wine.data)
    y = wine.target
    target_names = list(wine.target_names)
    feat_names = ["PCA Component 1 (Chemical Variance)", "PCA Component 2 (Chemical Variance)"]
    dataset_meta = {
        "source": "UCI Machine Learning Repository / University of Genoa (1991)",
        "description": "178 wine samples from three Italian cultivars, analyzed across 13 chemical constituents (alcohol, malic acid, flavonoids, phenols, proline).",
        "geometry": "The 13-dimensional chemical space has been projected to 2D using Principal Component Analysis (PCA) while preserving 55%+ of total variance.",
        "challenge": "Multi-class real-world chemical classification with high-dimensional variance compressed into orthogonal eigenvectors.",
        "is_synthetic": False,
        "n_samples": 178,
        "noise": "Real-world chemical spectroscopy variation",
        "seed": "Fixed historical dataset",
    }

test_size = st.sidebar.slider("Test Set Split Ratio:", 0.15, 0.50, 0.25, step=0.05)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# ---------------------------------------------------------
# Section 1: Selected Dataset Profile & Context
# ---------------------------------------------------------
st.subheader(f"📊 Dataset Profile: {dataset_name}")

d_col1, d_col2 = st.columns([3, 2])
with d_col1:
    st.markdown(f"**Origin & Provenance:** {dataset_meta['source']}")
    st.markdown(f"**What are we looking at?** {dataset_meta['description']}")
    st.markdown(f"**Features (Axes):** X-Axis represents `{feat_names[0]}`, Y-Axis represents `{feat_names[1]}`.")
    st.markdown(f"**Target Classes:** {', '.join([f'`{t}`' for t in target_names])}.")
    st.info(f"💡 **The Machine Learning Challenge:** {dataset_meta['challenge']}")

with d_col2:
    st.markdown("#### Active Data Parameters")
    st.markdown(f"- **Total Instances ($N$):** `{len(X)}` points")
    st.markdown(f"- **Training Set Size:** `{len(X_train)}` points ({(1 - test_size) * 100:.0f}%)")
    st.markdown(f"- **Test Evaluation Size:** `{len(X_test)}` points ({test_size * 100:.0f}%)")
    st.markdown(f"- **Noise / Dispersion:** `{dataset_meta['noise']}`")
    st.markdown(f"- **Random Seed (`random_state`):** `{dataset_meta['seed']}`")

# Deep-dive callout on what the Random Seed and Sliders do
with st.expander("🎲 **What does changing the Random Seed, Sample Size, and Noise actually do?**", expanded=False):
    st.markdown(
        r"""
        - **What is a Random Seed (`random_state`)?**
          Computers cannot generate truly random numbers; they use **Pseudo-Random Number Generators (PRNGs)** driven by mathematical formulas.
          The *seed* is the initial numerical starting state.
          - When the seed is constant (`42`), the points land in the exact same coordinates and the train/test split is identical every time. This guarantees **scientific reproducibility**.
          - When you change the seed (e.g. from `42` to `100`), the PRNG follows a different sequence: points land in different positions, noise values randomize differently, and different points are held out for testing.
          - **Why this matters in AIM111:** Changing the seed stress-tests the model. If a classifier achieves 96% accuracy on seed 42 but drops to 78% on seed 43, it was merely lucky on one particular layout!
        - **Sample Size ($N$):** Controls data density. Sparse data (100 points) forces models to extrapolate with little evidence, increasing variance. Dense data (1000 points) makes decision boundaries sharper and more reliable.
        - **Noise Level:** Adds Gaussian perturbation $\mathcal{N}(0, \sigma^2)$ to coordinates, simulating real-world measurement error or overlapping classes. High noise forces algorithms to balance fitting true trends vs fitting random noise.
        - **Test Split Ratio:** Defines what fraction of data is quarantined. A 25% test split means 75% is used for training and 25% is used to rigorously grade the model on data it has never seen.
        """
    )

st.markdown("---")

# ---------------------------------------------------------
# Sidebar: Classifier Selection & Hyperparameters
# ---------------------------------------------------------
st.sidebar.markdown("### 2. Model Selection & Tuning")

model_choice = st.sidebar.selectbox(
    "Classifier Algorithm:",
    [
        "Random Forest 🌲",
        "Support Vector Machine (SVM) ⚡",
        "K-Nearest Neighbors (KNN) 📍",
        "Logistic Regression 📈",
        "Decision Tree 🌳",
    ],
    index=0,
)

model_params_summary = {}

if "Random Forest" in model_choice:
    n_estimators = st.sidebar.slider("Number of Trees (n_estimators):", 10, 200, 60, step=10)
    max_depth_val = st.sidebar.slider("Max Tree Depth (0 = Unlimited):", 0, 20, 5, step=1)
    max_depth = None if max_depth_val == 0 else max_depth_val
    criterion = st.sidebar.radio("Split Impurity Criterion:", ["gini", "entropy"], horizontal=True)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=criterion,
        random_state=42,
    )
    model_params_summary = {
        "n_estimators": f"{n_estimators} trees",
        "max_depth": "Unlimited" if max_depth is None else str(max_depth),
        "criterion": criterion.capitalize(),
    }

elif "Support Vector Machine" in model_choice:
    kernel = st.sidebar.selectbox("Kernel Type:", ["rbf", "linear", "poly", "sigmoid"], index=0)
    C_val = st.sidebar.slider("Regularization Parameter (C):", 0.05, 10.0, 1.0, step=0.05)
    gamma = st.sidebar.selectbox("Gamma (Kernel Coefficient):", ["scale", "auto", 0.1, 0.5, 1.0, 2.0], index=0)
    clf = SVC(kernel=kernel, C=C_val, gamma=gamma, probability=True, random_state=42)
    model_params_summary = {
        "kernel": kernel.upper(),
        "C": f"{C_val} (Higher = strict boundary, Lower = softer margin)",
        "gamma": str(gamma),
    }

elif "K-Nearest Neighbors" in model_choice:
    n_neighbors = st.sidebar.slider("Number of Neighbors (K):", 1, 25, 5, step=1)
    weights = st.sidebar.radio("Neighbor Weighting:", ["uniform", "distance"], horizontal=True)
    metric = st.sidebar.selectbox("Distance Metric:", ["euclidean", "manhattan", "chebyshev"], index=0)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
    model_params_summary = {
        "K Neighbors": f"{n_neighbors} points",
        "Weighting": weights.capitalize(),
        "Distance Metric": metric.capitalize(),
    }

elif "Logistic Regression" in model_choice:
    C_val = st.sidebar.slider("Inverse Regularization (C):", 0.01, 10.0, 1.0, step=0.05)
    clf = LogisticRegression(C=C_val, solver="lbfgs", max_iter=500, random_state=42)
    model_params_summary = {
        "C": f"{C_val}",
        "Solver": "L-BFGS (Quasi-Newton optimization)",
        "Decision Surface": "Strict 1D Linear Hyperplane",
    }

else:  # Decision Tree
    max_depth_val = st.sidebar.slider("Max Tree Depth (0 = Unlimited):", 0, 15, 4, step=1)
    max_depth = None if max_depth_val == 0 else max_depth_val
    min_samples_split = st.sidebar.slider("Min Samples to Split Node:", 2, 20, 2, step=1)
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
    )
    model_params_summary = {
        "max_depth": "Unlimited" if max_depth is None else str(max_depth),
        "min_samples_split": f"{min_samples_split} samples",
        "Split Logic": "Axis-aligned orthogonal thresholds",
    }

# Train the model live
clf.fit(X_train, y_train)

# Predictions & Metrics
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

num_classes = len(np.unique(y))
avg_method = "binary" if num_classes == 2 else "macro"

acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, average=avg_method, zero_division=0)
rec = recall_score(y_test, y_pred_test, average=avg_method, zero_division=0)
f1 = f1_score(y_test, y_pred_test, average=avg_method, zero_division=0)

# ---------------------------------------------------------
# Section 2: Model Architecture & Philosophy Explainer
# ---------------------------------------------------------
st.subheader(f"🧠 Model Architecture: {model_choice}")

st.markdown(
    """
    > 💡 **Important Conceptual Clarification:**  
    > Changing the model selection **does not alter the dataset in any way**. The points, coordinates, and labels remain identical.  
    > What changes is the **mathematical philosophy and decision logic** the model uses to partition the feature space!
    """
)

m_col1, m_col2 = st.columns([3, 2])
with m_col1:
    if "Random Forest" in model_choice:
        st.markdown(
            """
            - **Mathematical Philosophy:** **Ensemble Bagging (Bootstrap Aggregation)**.
            - **What makes it unique:** Instead of relying on one tree, it trains dozens of independent decision trees on random subsets of the data and random subsets of features. When predicting, all trees vote, and the majority wins.
            - **Boundary Behavior:** By averaging hundreds of right-angle tree cuts, it creates a finely stepped, robust boundary that resists overfitting much better than an individual decision tree.
            """
        )
    elif "Support Vector Machine" in model_choice:
        st.markdown(
            """
            - **Mathematical Philosophy:** **Maximum-Margin Hyperplane & Kernel Trick**.
            - **What makes it unique:** Rather than just finding *any* separating boundary, SVM finds the unique boundary that maximizes the geometric distance (margin) to the closest data points (**support vectors**).
            - **Boundary Behavior:** When using the **RBF (Radial Basis Function)** kernel, it projects the 2D points into an infinite-dimensional Hilbert space where complex interlocking curves become linearly separable, producing smooth, flowing decision contours.
            """
        )
    elif "K-Nearest Neighbors" in model_choice:
        st.markdown(
            """
            - **Mathematical Philosophy:** **Instance-Based Lazy Learning**.
            - **What makes it unique:** KNN does not learn a mathematical formula or adjust weights. It memorizes the training data. At prediction time, it calculates the distance to every stored point, finds the $K$ closest neighbors, and takes a majority vote.
            - **Boundary Behavior:** Creates local geometric Voronoi-like "islands" and puzzle-piece boundaries. Highly flexible, but computationally expensive on huge datasets.
            """
        )
    elif "Logistic Regression" in model_choice:
        st.markdown(
            r"""
            - **Mathematical Philosophy:** **Linear Parametric Probabilistic Model**.
            - **What makes it unique:** Computes a weighted sum $z = w_1 X_1 + w_2 X_2 + b$ and passes it through the logistic sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ to estimate class probabilities.
            - **Boundary Behavior:** The decision threshold is strictly a **straight line (1D hyperplane)**. It cannot bend around curves, but provides transparent, mathematically interpretable feature odds ratios.
            """
        )
    else:  # Decision Tree
        st.markdown(
            """
            - **Mathematical Philosophy:** **Recursive Binary Partitioning**.
            - **What makes it unique:** It constructs a top-down flowchart of orthogonal if/else rules (e.g. *if $X_1 > 1.2$ then go left, else go right*).
            - **Boundary Behavior:** Boundaries are strictly axis-aligned vertical and horizontal lines. Fast and intuitive, but prone to high variance and overfitting if `max_depth` is unrestricted.
            """
        )

with m_col2:
    st.markdown("#### Active Model Hyperparameters")
    for k, v in model_params_summary.items():
        st.markdown(f"- **{k}:** `{v}`")

st.markdown("---")

# ---------------------------------------------------------
# Performance KPI Cards
# ---------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Test Accuracy", f"{acc * 100:.1f}%", help="Percentage of held-out test points correctly classified.")
k2.metric("Precision", f"{prec * 100:.1f}%", help="When model predicted positive, how often was it right? (TP / (TP + FP))")
k3.metric("Recall (Sensitivity)", f"{rec * 100:.1f}%", help="Of all actual positive cases, how many did the model catch? (TP / (TP + FN))")
k4.metric("F1-Score", f"{f1 * 100:.1f}%", help="Harmonic mean of Precision and Recall. Balances false alarms vs missed detections.")

st.markdown("---")

# ---------------------------------------------------------
# Interactive 2D Decision Boundary Visualization
# ---------------------------------------------------------
st.subheader("🎯 Live 2D Decision Boundary & Classification Surface")

c_opt1, c_opt2 = st.columns([2, 1])
with c_opt1:
    view_mode = st.radio(
        "Boundary Display Mode:",
        ["Color Decision Regions (Categorical)", "Confidence Probability Gradient (Continuous)"],
        horizontal=True,
    )
with c_opt2:
    show_test_only = st.checkbox("Highlight Test Samples Only", value=False)

# Compute 2D mesh grid
padding = 0.6
x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding

h = 0.04  # grid resolution
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]

fig = go.Figure()

# Background Decision Surface
if view_mode.startswith("Color Decision") or num_classes > 2:
    Z = clf.predict(grid_points)
    Z = Z.reshape(xx.shape)
    fig.add_trace(
        go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z,
            colorscale="Viridis",
            opacity=0.35,
            showscale=False,
            contours=dict(showlines=True, coloring="heatmap"),
            hoverinfo="none",
        )
    )
else:
    if hasattr(clf, "predict_proba"):
        Z_prob = clf.predict_proba(grid_points)[:, 1]
    else:
        Z_prob = clf.decision_function(grid_points)
    Z_prob = Z_prob.reshape(xx.shape)
    fig.add_trace(
        go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z_prob,
            colorscale="RdBu",
            opacity=0.5,
            showscale=True,
            colorbar=dict(title="P(Class 1)", len=0.8),
            contours=dict(showlines=True),
            hoverinfo="none",
        )
    )

# Training Points Scatter
if not show_test_only:
    for class_idx, class_label in enumerate(target_names):
        mask_train = y_train == class_idx
        fig.add_trace(
            go.Scatter(
                x=X_train[mask_train, 0],
                y=X_train[mask_train, 1],
                mode="markers",
                name=f"Train: {class_label}",
                marker=dict(size=8, opacity=0.8, line=dict(color="#222", width=1)),
                hovertemplate=f"<b>Train - {class_label}</b><br>{feat_names[0]}: %{{x:.2f}}<br>{feat_names[1]}: %{{y:.2f}}<extra></extra>",
            )
        )

# Test Points Scatter
for class_idx, class_label in enumerate(target_names):
    mask_test = y_test == class_idx
    fig.add_trace(
        go.Scatter(
            x=X_test[mask_test, 0],
            y=X_test[mask_test, 1],
            mode="markers",
            name=f"Test: {class_label}",
            marker=dict(
                size=12,
                symbol="diamond",
                opacity=0.95,
                line=dict(color="#ffffff", width=2),
            ),
            hovertemplate=f"<b>Test - {class_label}</b><br>{feat_names[0]}: %{{x:.2f}}<br>{feat_names[1]}: %{{y:.2f}}<extra></extra>",
        )
    )

fig.update_layout(
    xaxis_title=feat_names[0],
    yaxis_title=feat_names[1],
    height=540,
    margin=dict(l=30, r=30, t=30, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "🔍 **Visual Reading Guide:** Circles represent Training data. Diamonds with white borders represent held-out Test evaluation data. "
    "The background colored zones represent the model's predicted classification regions across coordinate space."
)

st.markdown("---")

# ---------------------------------------------------------
# Detailed Diagnostic Tabs with Context
# ---------------------------------------------------------
st.subheader("🔬 Model Diagnostic Analysis & Interpretability")

tab_cm, tab_roc, tab_importance, tab_data = st.tabs(
    [
        "📊 Confusion Matrix",
        "📈 ROC & Precision-Recall",
        "🔍 Model Insights & Weights",
        "📋 Test Predictions Table",
    ]
)

# Tab 1: Confusion Matrix
with tab_cm:
    st.markdown("#### Test Set Confusion Matrix")
    st.markdown(
        """
        The **Confusion Matrix** provides an exact count breakdown comparing the ground-truth actual labels 
        (vertical Y-axis) against the model's predictions (horizontal X-axis) on the **held-out test set**.
        - **Main Diagonal (Top-Left to Bottom-Right):** Correct predictions (True Positives and True Negatives). Higher numbers are better.
        - **Off-Diagonal Cells:** Errors.
          - **False Positive (Type I Error):** The model predicted a class, but it was actually another class.
          - **False Negative (Type II Error):** The model missed an actual instance of a class.
        """
    )
    cm = confusion_matrix(y_test, y_pred_test)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted Class (Model Output)", y="Actual True Class (Ground Truth)", color="Count"),
        x=target_names,
        y=target_names,
        color_continuous_scale="Blues",
    )
    fig_cm.update_layout(height=380, margin=dict(l=40, r=40, t=30, b=30))
    st.plotly_chart(fig_cm, use_container_width=True)

# Tab 2: ROC Curve
with tab_roc:
    st.markdown("#### Receiver Operating Characteristic (ROC) & AUC")
    st.markdown(
        """
        The **ROC Curve** evaluates binary classifier discrimination across all possible probability thresholds (from 0.0 to 1.0):
        - **Y-Axis: True Positive Rate (Sensitivity / Recall):** $\\frac{\\text{TP}}{\\text{TP} + \\text{FN}}$. The fraction of actual positives correctly flagged.
        - **X-Axis: False Positive Rate (1 - Specificity):** $\\frac{\\text{FP}}{\\text{FP} + \\text{TN}}$. The fraction of actual negatives falsely flagged as positive.
        - **Area Under the Curve (AUC):**
          - **$\text{AUC} = 1.0$:** Flawless classifier with zero overlap between positive and negative distributions.
          - **$\text{AUC} = 0.8 - 0.9$:** Strong, robust classification performance.
          - **$\text{AUC} = 0.5$ (Dashed Gray Diagonal):** No better than flipping a fair coin.
        """
    )
    if num_classes == 2 and hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC Curve (AUC = {roc_auc:.3f})", line=dict(color="#00aaff", width=3))
        )
        fig_roc.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess Baseline (AUC = 0.50)", line=dict(dash="dash", color="gray"))
        )
        fig_roc.update_layout(
            xaxis_title="False Positive Rate (FPR)",
            yaxis_title="True Positive Rate (TPR)",
            height=380,
            margin=dict(l=40, r=40, t=30, b=30),
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info("ℹ️ The ROC Curve is displayed for binary classification tasks (Moons, Circles, Linearly Separable). Multi-class problems are evaluated using the Confusion Matrix.")

# Tab 3: Model Insights
with tab_importance:
    st.markdown(f"#### Internal Model Representations & Feature Weights ({model_choice.split()[0]})")
    st.markdown(
        """
        This tab reveals the internal mathematical parameters the algorithm learned to separate the data:
        """
    )
    if hasattr(clf, "feature_importances_"):
        st.markdown(
            "- **Feature Importance (MDI / Mean Decrease in Impurity):** Indicates the relative contribution of each feature "
            "toward reducing Gini impurity across all decision splits in the ensemble."
        )
        fi_df = pd.DataFrame({"Feature": feat_names, "Importance": clf.feature_importances_})
        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Teal",
        )
        fig_fi.update_layout(height=260)
        st.plotly_chart(fig_fi, use_container_width=True)

    elif hasattr(clf, "coef_"):
        st.markdown(
            "- **Hyperplane Coefficients ($w_1, w_2$):** Represents the slopes of the linear decision boundary. "
            "A positive weight increases the log-odds of Class 1 as the feature increases, while a negative weight decreases it."
        )
        coef_df = pd.DataFrame({"Feature": feat_names, "Coefficient Weight": clf.coef_[0][:2]})
        fig_coef = px.bar(
            coef_df,
            x="Coefficient Weight",
            y="Feature",
            orientation="h",
            color="Coefficient Weight",
            color_continuous_scale="Plasma",
        )
        fig_coef.update_layout(height=260)
        st.plotly_chart(fig_coef, use_container_width=True)

    elif hasattr(clf, "support_vectors_"):
        st.markdown(
            "- **Support Vectors:** The critical borderline instances from the training set that sit directly on the margin boundaries. "
            "All other training points could be deleted from memory without changing the boundary position!"
        )
        st.write(f"**Total Support Vectors identified:** `{len(clf.support_vectors_)}` out of `{len(X_train)}` training points ({len(clf.support_vectors_) / len(X_train):.1%})")
        st.write(f"**Support Vectors per Class:** `{clf.n_support_}`")

    else:
        st.write(f"Algorithm: **{model_choice}** using $K={n_neighbors}$ nearest neighbors with `{metric}` distance metric.")

# Tab 4: Predictions Inspector
with tab_data:
    st.markdown("#### Test Instances and Prediction Auditing")
    st.markdown(
        """
        This table allows you to audit individual test cases. You can filter or sort by **Correct?** to inspect 
        the specific borderline instances where the classifier made misclassifications, and view the model's 
        **Confidence** probability score.
        """
    )
    pred_df = pd.DataFrame(X_test, columns=feat_names)
    pred_df["True Label"] = [target_names[idx] for idx in y_test]
    pred_df["Predicted Label"] = [target_names[idx] for idx in y_pred_test]
    pred_df["Correct?"] = pred_df["True Label"] == pred_df["Predicted Label"]

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_test)
        pred_df["Confidence"] = np.max(probs, axis=1)
        st.dataframe(
            pred_df.style.format({"Confidence": "{:.1%}", feat_names[0]: "{:.2f}", feat_names[1]: "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
