import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
 
FEATURE_COLS = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)", "Income_Spending"]
NUMERICAL_COLS = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Income_Spending"]
 
 
@st.cache_data
def load_and_prepare_data(csv_path="Mall_Customers.csv"):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["CustomerID"])
 
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[18, 25, 35, 50, 70],
        labels=["Young", "Adult", "Middle", "Senior"]
    )
    df["Income_Level"] = pd.cut(
        df["Annual Income (k$)"],
        bins=[0, 40, 70, 140],
        labels=["Low", "Medium", "High"]
    )
    df["Income_Spending"] = df["Annual Income (k$)"] * df["Spending Score (1-100)"]
 
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])  
 
    return df, le
 
 
@st.cache_resource
def train_model(df, n_clusters=5):
    X = df[FEATURE_COLS].copy()
 
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[NUMERICAL_COLS] = scaler.fit_transform(X_scaled[NUMERICAL_COLS])
 
    # Elbow data
    inertia = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
 
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, clusters)
 
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
 
    df = df.copy()
    df["Cluster"] = clusters
 
    return {
        "df": df,
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
        "components": components,
        "inertia": inertia,
        "silhouette": sil_score,
    }
 
 
def describe_cluster(row):
    """Simple human-readable profile for a cluster based on its mean values."""
    income = row["Annual Income (k$)"]
    spending = row["Spending Score (1-100)"]
 
    if income >= 70 and spending >= 60:
        return "High Income, High Spending — Premium / VIP customers"
    elif income >= 70 and spending < 40:
        return "High Income, Low Spending — Careful / budget-conscious customers"
    elif income < 40 and spending >= 60:
        return "Low Income, High Spending — Impulse buyers"
    elif income < 40 and spending < 40:
        return "Low Income, Low Spending — Low engagement customers"
    else:
        return "Mid Income, Mid Spending — Average / balanced customers"
 
 
def get_cluster_profiles(df_clustered):
    profiles = (
        df_clustered.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
        .mean()
        .round(1)
    )
    profiles["Profile"] = profiles.apply(describe_cluster, axis=1)
    return profiles
 
 
def predict_new_customer(gender, age, income, spending, result):
    """
    Takes raw user inputs and the dict returned by train_model(),
    returns (cluster_id, pca_point) for the new customer.
    """
    gender_val = 1 if gender == "Male" else 0
    income_spending_val = income * spending
 
    new_customer = pd.DataFrame([{
        "Gender": gender_val,
        "Age": age,
        "Annual Income (k$)": income,
        "Spending Score (1-100)": spending,
        "Income_Spending": income_spending_val,
    }])
 
    new_scaled = new_customer.copy()
    new_scaled[NUMERICAL_COLS] = result["scaler"].transform(new_customer[NUMERICAL_COLS])
 
    cluster_pred = result["kmeans"].predict(new_scaled)[0]
    new_point_pca = result["pca"].transform(new_scaled)
 
    return cluster_pred, new_point_pca
 
