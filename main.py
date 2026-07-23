import streamlit as st

st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛍️"
)

import pandas as pd

from my_module.model import (
    load_and_prepare_data,
    train_model,
    get_cluster_profiles,
    predict_new_customer,
)

# ---------------------------------------------------------
# Load data & train model (cached — runs once)
# ---------------------------------------------------------
df, gender_encoder = load_and_prepare_data()
result = train_model(df, n_clusters=5)

df_clustered = result["df"]
cluster_profiles = get_cluster_profiles(df_clustered)

st.title("🛍️ Mall Customer Segmentation")
st.write(
    "This app groups mall customers into segments based on their age, gender, "
    "income, and spending behavior using K-Means clustering — helping design "
    "targeted marketing strategies for each group."
)

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧩 Clusters", "🔮 Predict a Customer"])

with tab1:
    st.subheader("Dataset")
    if st.checkbox("Show raw data"):
        st.dataframe(df, use_container_width=True)

    st.header("Distributions")
    dist_tab1, dist_tab2, dist_tab3 = st.tabs(["Age", "Annual Income", "Spending Score"])

    with dist_tab1:
        age_bins = pd.cut(df["Age"], bins=10)
        age_counts = age_bins.value_counts().sort_index()
        age_counts.index = age_counts.index.astype(str)
        st.bar_chart(age_counts, use_container_width=True)
        st.markdown(
            "> **What this shows:** the age distribution of mall customers. "
            "Most customers are aged between **30 and 40 years**, with fewer "
            "customers at the very young or older ends."
        )

    with dist_tab2:
        income_bins = pd.cut(df["Annual Income (k$)"], bins=10)
        income_counts = income_bins.value_counts().sort_index()
        income_counts.index = income_counts.index.astype(str)
        st.bar_chart(income_counts, use_container_width=True)
        st.markdown(
            "> **What this shows:** annual income is close to a **normal distribution**, "
            "with most customers earning between **$50k and $80k** per year."
        )

    with dist_tab3:
        spending_bins = pd.cut(df["Spending Score (1-100)"], bins=10)
        spending_counts = spending_bins.value_counts().sort_index()
        spending_counts.index = spending_counts.index.astype(str)
        st.bar_chart(spending_counts, use_container_width=True)
        st.markdown(
            "> **What this shows:** most customers have a spending score "
            "**centered around 40-60**, meaning moderate spending behavior is the norm."
        )

    st.header("Annual Income vs Spending Score")
    scatter_df = df.copy()
    scatter_df["Gender"] = scatter_df["Gender"].map({0: "Female", 1: "Male"})
    st.scatter_chart(
        scatter_df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Gender",
        use_container_width=True,
    )
    st.markdown(
        "> **What this shows:** plotting income against spending score reveals "
        "**clear, visually separable groups** — roughly 5 distinct customer segments, "
        "which is the main signal used for clustering."
    )

    st.header("Elbow Method")
    elbow_df = pd.DataFrame({"Inertia": result["inertia"]}, index=range(1, 11))
    elbow_df.index.name = "Number of Clusters (K)"
    st.line_chart(elbow_df, use_container_width=True)
    st.markdown(
        "> **What this shows:** inertia (how tightly packed the clusters are) drops "
        "sharply at first, then flattens out. The bend ('elbow') happens around "
        "**K = 5**, which is why 5 clusters were chosen for this model."
    )

    st.metric("Silhouette Score (K=5)", f"{result['silhouette']:.3f}")
    st.markdown(
        "> **What this means:** the Silhouette Score measures how well-separated "
        "the clusters are, from **-1** (wrong grouping) to **1** (perfectly separated). "
        "This score reflects how distinct the 5 customer segments are from each other."
    )

with tab2:
    st.subheader("Customer Segments (PCA Projection)")

    pca_df = pd.DataFrame(result["components"], columns=["PCA Component 1", "PCA Component 2"])
    pca_df["Group"] = "Cluster " + df_clustered["Cluster"].astype(str)

    centroids_pca = result["pca"].transform(result["kmeans"].cluster_centers_)
    centroids_df = pd.DataFrame(centroids_pca, columns=["PCA Component 1", "PCA Component 2"])
    centroids_df["Group"] = "Centroid"

    combined_df = pd.concat([pca_df, centroids_df], ignore_index=True)

    st.scatter_chart(
        combined_df,
        x="PCA Component 1",
        y="PCA Component 2",
        color="Group",
        use_container_width=True,
    )
    st.markdown(
        "> **What this shows:** each color represents a different customer segment, "
        "and **Centroid** marks the center of each group. The X and Y axes are "
        "**PCA-reduced dimensions** — a compressed combination of all features used for "
        "clustering (Gender, Age, Income, Spending, Income×Spending), not a single raw feature."
    )

    st.subheader("Cluster Profiles")
    st.dataframe(cluster_profiles, use_container_width=True)

    st.subheader("Age Group & Income Level breakdown per Cluster")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Age Group**")
        st.dataframe(df_clustered.groupby("Cluster")["Age_Group"].value_counts().unstack(fill_value=0))
    with col2:
        st.write("**Income Level**")
        st.dataframe(df_clustered.groupby("Cluster")["Income_Level"].value_counts().unstack(fill_value=0))

with tab3:
    st.subheader("Enter customer details")

    c1, c2 = st.columns(2)
    with c1:
        gender_input = st.selectbox("Gender", ["Female", "Male"])
        age_input = st.slider("Age", 18, 70, 30)
    with c2:
        income_input = st.slider("Annual Income (k$)", 15, 140, 60)
        spending_input = st.slider("Spending Score (1-100)", 1, 99, 50)

    if st.button("Predict Segment", use_container_width=True):
        cluster_pred, new_point_pca = predict_new_customer(
            gender_input, age_input, income_input, spending_input, result
        )
        profile_text = cluster_profiles.loc[cluster_pred, "Profile"]

        st.success(f"This customer belongs to **Cluster {cluster_pred}**")
        st.info(profile_text)

        pred_pca_df = pd.DataFrame(result["components"], columns=["PCA Component 1", "PCA Component 2"])
        pred_pca_df["Group"] = "Cluster " + df_clustered["Cluster"].astype(str)

        new_point_df = pd.DataFrame(new_point_pca, columns=["PCA Component 1", "PCA Component 2"])
        new_point_df["Group"] = "New Customer"

        pred_combined_df = pd.concat([pred_pca_df, new_point_df], ignore_index=True)

        st.scatter_chart(
            pred_combined_df,
            x="PCA Component 1",
            y="PCA Component 2",
            color="Group",
            use_container_width=True,
        )
        st.markdown(
            "> **What this shows:** the **New Customer** point marks where this "
            "customer falls relative to the existing segments — the closer it sits "
            "to a colored group, the more that group's behavior applies to them."
        )