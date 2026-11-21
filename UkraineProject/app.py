import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Ukraine Humanitarian Fund Explorer",
    page_icon="🇺🇦",
    layout="centered",
)

# Custom CSS for Ukrainian flag background and grey middle column
st.markdown(
    """
    <style>
    /* Ukrainian Flag Background (Top Half Blue #0057B7, Bottom Half Yellow #FFD700) */
    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #0057B7 0%, #0057B7 50%, #FFD700 50%, #FFD700 100%) !important;
        background-attachment: fixed !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Middle column of data keeping its grey background */
    .block-container, [data-testid="stMainBlockContainer"] {
        background-color: #262730 !important;
        color: #FAFAFA !important;
        padding: 2.2rem 2.8rem !important;
        border-radius: 14px !important;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.5) !important;
        margin-top: 1.8rem !important;
        margin-bottom: 2.5rem !important;
    }

    .block-container h1, .block-container h2, .block-container h3, 
    .block-container h4, .block-container p, .block-container label,
    .block-container span {
        color: #FAFAFA !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load data
data_dir = Path(__file__).parent
projects_df = pd.read_csv(data_dir / "uhf_projects_2019_2026.csv")
subgrants_df = pd.read_csv(data_dir / "uhf_subgrants_2019_2026.csv")

# Ensure year is formatted as a discrete string so multi-year charts do not format as numbers (e.g. 2,019)
projects_df["year_str"] = projects_df["allocation_year"].astype(str)
all_years_str = [str(y) for y in sorted(projects_df["allocation_year"].unique())]

# Calculations for overall context
total_funding_since_2019 = projects_df["budget_usd"].sum()
total_donators = projects_df["prime_partner"].nunique()
total_categories = projects_df["prime_partner_type"].nunique()
total_subcategories = projects_df["prime_partner"].nunique()

st.title("Ukraine Humanitarian Fund Explorer")

st.write(
    f"Since the full scale invasion of Ukraine there have been "
    f"${total_funding_since_2019:,.2f} USD donated from {total_donators} organizations "
    f"across {total_categories} categories and {total_subcategories} sub categories. "
    f"On this page you can select a specific year, category and/or sub category to view "
    f"how these groups donated and the various projects involved in helping the people of Ukraine."
)

st.divider()

# Helper for Select All / Clear All
def set_all_keys(keys, value):
    for k in keys:
        st.session_state[k] = value

# 1. Year Selection
years_available = ["All Years (2019–2026)"] + sorted(projects_df["allocation_year"].unique(), reverse=True)
selected_year_val = st.selectbox("Select Year", years_available, index=0)

if selected_year_val == "All Years (2019–2026)":
    year_df = projects_df.copy()
else:
    year_df = projects_df[projects_df["allocation_year"] == selected_year_val].copy()

# ---------------------------------------------------------
# Expandable Filter Checkbox Lists (Category, Subcategory, Project)
# ---------------------------------------------------------
st.subheader("Filter Donations")

# 1. Categories Filter
cat_totals = (
    year_df.groupby("prime_partner_type")["budget_usd"]
    .sum()
    .sort_values(ascending=False)
)
cat_keys = [f"cat_{c}" for c in cat_totals.index]
for k in cat_keys:
    if k not in st.session_state:
        st.session_state[k] = True

selected_categories = [c for c in cat_totals.index if st.session_state.get(f"cat_{c}", True)]

with st.expander(f"📁 Categories ({len(selected_categories)}/{len(cat_totals)} selected)", expanded=False):
    btn_col1, btn_col2 = st.columns(2)
    btn_col1.button("Select All", key="btn_sel_all_cat", on_click=set_all_keys, args=(cat_keys, True))
    btn_col2.button("Clear All", key="btn_clr_all_cat", on_click=set_all_keys, args=(cat_keys, False))

    for cat_name, amt in cat_totals.items():
        st.checkbox(f"{cat_name} (${amt:,.2f})", key=f"cat_{cat_name}")

selected_categories = [c for c in cat_totals.index if st.session_state.get(f"cat_{c}", True)]
cat_filtered_df = year_df[year_df["prime_partner_type"].isin(selected_categories)]

# 2. Subcategories (Partners) Filter
subcat_totals = (
    cat_filtered_df.groupby("prime_partner")["budget_usd"]
    .sum()
    .sort_values(ascending=False)
)
subcat_keys = [f"subcat_{p}" for p in subcat_totals.index]
for k in subcat_keys:
    if k not in st.session_state:
        st.session_state[k] = True

selected_subcats = [p for p in subcat_totals.index if st.session_state.get(f"subcat_{p}", True)]

with st.expander(f"📁 Subcategories / Partners ({len(selected_subcats)}/{len(subcat_totals)} selected)", expanded=False):
    btn_col1, btn_col2 = st.columns(2)
    btn_col1.button("Select All", key="btn_sel_all_subcat", on_click=set_all_keys, args=(subcat_keys, True))
    btn_col2.button("Clear All", key="btn_clr_all_subcat", on_click=set_all_keys, args=(subcat_keys, False))

    with st.container(height=200):
        for partner_name, amt in subcat_totals.items():
            st.checkbox(f"{partner_name} (${amt:,.2f})", key=f"subcat_{partner_name}")

selected_subcats = [p for p in subcat_totals.index if st.session_state.get(f"subcat_{p}", True)]
subcat_filtered_df = cat_filtered_df[cat_filtered_df["prime_partner"].isin(selected_subcats)]

# 3. Projects Filter
proj_sorted_df = subcat_filtered_df.sort_values("budget_usd", ascending=False)
proj_keys = [f"proj_{row['project_code']}" for _, row in proj_sorted_df.iterrows()]
for k in proj_keys:
    if k not in st.session_state:
        st.session_state[k] = True

selected_projects = [
    row["project_code"]
    for _, row in proj_sorted_df.iterrows()
    if st.session_state.get(f"proj_{row['project_code']}", True)
]

with st.expander(f"📁 Projects ({len(selected_projects)}/{len(proj_sorted_df)} selected)", expanded=False):
    btn_col1, btn_col2 = st.columns(2)
    btn_col1.button("Select All", key="btn_sel_all_proj", on_click=set_all_keys, args=(proj_keys, True))
    btn_col2.button("Clear All", key="btn_clr_all_proj", on_click=set_all_keys, args=(proj_keys, False))

    with st.container(height=220):
        for _, row in proj_sorted_df.iterrows():
            st.checkbox(
                f"${row['budget_usd']:,.2f} — {row['project_title'][:65]}... [{row['project_code']}]",
                key=f"proj_{row['project_code']}",
            )

selected_projects = [
    row["project_code"]
    for _, row in proj_sorted_df.iterrows()
    if st.session_state.get(f"proj_{row['project_code']}", True)
]
final_filtered_df = proj_sorted_df[proj_sorted_df["project_code"].isin(selected_projects)]

st.divider()

# ---------------------------------------------------------
# Line Graph: Multi-Year Timeline or Swapped Axes for Specific Year
# ---------------------------------------------------------
if selected_year_val == "All Years (2019–2026)":
    st.subheader("Funding Trend Over Time (2019–2026)")
    st.caption("X-axis: Years | Y-axis: Funding (USD)")
else:
    st.subheader(f"Funding Distribution ({selected_year_val})")
    st.caption("Axes Swapped: Money (USD) on the X-axis")

filtered_sum = final_filtered_df["budget_usd"].sum()
st.write(f"**Filtered Total:** ${filtered_sum:,.2f} across **{len(final_filtered_df)}** selected project(s)")

# Radio buttons to change view from categories to subcategories
chart_view = st.radio(
    "Change View:",
    ["Categories", "Subcategories (Partners)"],
    horizontal=True,
)

if not final_filtered_df.empty:
    if selected_year_val == "All Years (2019–2026)":
        # Multi-year timeline: X = Years, Y = Money
        if chart_view == "Categories":
            trend = (
                final_filtered_df.pivot_table(
                    index="year_str",
                    columns="prime_partner_type",
                    values="budget_usd",
                    aggfunc="sum",
                    fill_value=0,
                )
                .reindex(all_years_str, fill_value=0)
            )
            st.line_chart(trend)
        else:
            partner_sums = (
                final_filtered_df.groupby("prime_partner")["budget_usd"]
                .sum()
                .sort_values(ascending=False)
            )
            if len(partner_sums) > 10:
                st.caption(f"Showing top 10 partners by funding among the {len(partner_sums)} active partners.")
                top_partners = partner_sums.head(10).index
                chart_data_df = final_filtered_df[final_filtered_df["prime_partner"].isin(top_partners)]
            else:
                chart_data_df = final_filtered_df

            trend = (
                chart_data_df.pivot_table(
                    index="year_str",
                    columns="prime_partner",
                    values="budget_usd",
                    aggfunc="sum",
                    fill_value=0,
                )
                .reindex(all_years_str, fill_value=0)
            )
            st.line_chart(trend)
    else:
        # Specific Year: Swap axes so Money (USD) is on the X-axis
        if chart_view == "Categories":
            cat_chart_df = (
                final_filtered_df.groupby("prime_partner_type")["budget_usd"]
                .sum()
                .reset_index()
                .rename(columns={"prime_partner_type": "Category", "budget_usd": "Budget (USD)"})
                .sort_values("Budget (USD)", ascending=True)
            )
            chart = (
                alt.Chart(cat_chart_df)
                .mark_line(point=dict(size=70, filled=True, color="#FFD700"), color="#0057B7")
                .encode(
                    x=alt.X("Budget (USD):Q", title="Funding (USD)", axis=alt.Axis(format="$,.0f")),
                    y=alt.Y("Category:N", sort="-x", title="Category"),
                    tooltip=["Category:N", alt.Tooltip("Budget (USD):Q", format="$,.2f")],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            partner_sums = (
                final_filtered_df.groupby("prime_partner")["budget_usd"]
                .sum()
                .sort_values(ascending=False)
            )
            if len(partner_sums) > 10:
                st.caption(f"Showing top 10 partners by funding for {selected_year_val} among active selection.")
                top_partners = partner_sums.head(10).index
                sub_df = final_filtered_df[final_filtered_df["prime_partner"].isin(top_partners)]
            else:
                sub_df = final_filtered_df

            part_chart_df = (
                sub_df.groupby("prime_partner")["budget_usd"]
                .sum()
                .reset_index()
                .rename(columns={"prime_partner": "Partner", "budget_usd": "Budget (USD)"})
                .sort_values("Budget (USD)", ascending=True)
            )
            chart = (
                alt.Chart(part_chart_df)
                .mark_line(point=dict(size=70, filled=True, color="#FFD700"), color="#0057B7")
                .encode(
                    x=alt.X("Budget (USD):Q", title="Funding (USD)", axis=alt.Axis(format="$,.0f")),
                    y=alt.Y("Partner:N", sort="-x", title="Partner Organization"),
                    tooltip=["Partner:N", alt.Tooltip("Budget (USD):Q", format="$,.2f")],
                )
                .properties(height=360)
            )
            st.altair_chart(chart, use_container_width=True)
else:
    st.info("No projects selected. Please check at least one project, subcategory, or category above.")

st.divider()

# ---------------------------------------------------------
# Detailed Project & Line Item View
# ---------------------------------------------------------
st.subheader("Project & Line Item Detail")

if not final_filtered_df.empty:
    project_display_options = [
        f"{row['project_title'][:70]}... (${row['budget_usd']:,.2f}) [{row['project_code']}]"
        for _, row in final_filtered_df.iterrows()
    ]
    chosen_idx = st.selectbox(
        "Inspect specific project line items:",
        range(len(project_display_options)),
        format_func=lambda i: project_display_options[i],
    )
    chosen_project = final_filtered_df.iloc[chosen_idx]

    st.write(f"**Title:** {chosen_project['project_title']}")
    st.write(f"**Code:** {chosen_project['project_code']} &nbsp;|&nbsp; **Status:** {chosen_project['project_status']}")
    st.write(f"**Prime Partner:** {chosen_project['prime_partner']} ({chosen_project['prime_partner_type']})")
    st.write(f"**Total Budget:** ${chosen_project['budget_usd']:,.2f}")
    st.write(f"**Direct Retained:** ${chosen_project['budget_usd'] - chosen_project['subgrants_total_usd']:,.2f}")
    st.write(f"**Total Subgrants:** ${chosen_project['subgrants_total_usd']:,.2f}")

    # Subgrants line items table
    line_items = subgrants_df[subgrants_df["project_code"] == chosen_project["project_code"]][
        ["subpartner", "subpartner_type", "subgrant_usd"]
    ].sort_values("subgrant_usd", ascending=False)

    if not line_items.empty:
        line_items.columns = ["Subpartner", "Type", "Amount (USD)"]
        st.dataframe(line_items.style.format({"Amount (USD)": "${:,.2f}"}), use_container_width=True)
    else:
        st.write("No subgrants for this project (100% direct implementation by prime partner).")

    # Download button for filtered projects
    csv_data = final_filtered_df.to_csv(index=False)
    st.download_button("Download Filtered Projects CSV", csv_data, "filtered_projects.csv")
else:
    st.write("Select at least one project to see details.")
