#%% 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GSA_Interpolator import SolarEnergyInterpolator


#%% Load data

df = pd.read_pickle('main.pkl')


#%%
# --------------------------------------------
# Helper functions
# --------------------------------------------
def apply_degradation(predictions, ym_list):
    """
    Apply 0.5% yearly degradation from Jan of each following year.
    For each YYYY-MM in ym_list, we determine how many years since the first year.
    """
    # Convert to datetime
    dates = pd.to_datetime(ym_list)
    base_year = dates.min().year

    degraded = []
    for date, pred in zip(dates, predictions):
        years_passed = date.year - base_year
        factor = (1 - 0.005) ** years_passed
        degraded.append(pred * factor)

    return np.array(degraded)


def find_optimal_pf(model_values, actual_values):
    """
    Solve for optimal scaling factor pf that minimizes squared error:
    minimize ||pf*m - y||^2
    Closed form: pf = (m·y) / (m·m)
    """
    m = np.array(model_values)
    y = np.array(actual_values)
    if np.sum(m * m) == 0:
        return 1.0
    return np.sum(m * y) / np.sum(m * m)


def plot_comparison(ym_list, actual, predicted):

    # Use dark background style
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(10, 4))

    # Convert to readable month labels
    dates = pd.to_datetime(ym_list)
    month_labels = dates.strftime("%b\n%Y")  # e.g. Jan\n2024

    # Plot actual vs predicted
    ax.plot(month_labels, actual, label="Actual Energy", linewidth=2, marker='*', markersize=10)
    ax.plot(month_labels, predicted, label="Model (scaled)", linestyle="--", linewidth=2, marker='o', markersize=10)

    # Axis labels & title
    ax.set_xlabel("Month", fontsize=14)
    ax.set_ylabel("Energy (kWh)", fontsize=14)

    # Tick labels
    ax.tick_params(axis='both', labelsize=12)

    # Grid & legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    st.pyplot(fig)

# ----------------------------------------------------
# Streamlit App
# ----------------------------------------------------

mode = st.radio(
    "Select input mode:",
    ["A: Select site directly", "B: Cascading filters"],
    horizontal=True
)

# ===================================================
# 1️⃣ MODE A — DIRECT SEARCH (NO FILTERING)
# ===================================================
if mode == "A: Select site directly":

    st.subheader("Mode A: Direct site selection")

    site_names = sorted(df["SiteName"].dropna().unique())
    selected_site = st.selectbox(
        "Site Name",
        site_names,
        help="Type to search"
    )

    st.success(f"Selected site: {selected_site}")


# ===================================================
# 2️⃣ MODE B — FILTERED SELECTION
# ===================================================
if mode == "B: Cascading filters":

    st.subheader("Mode B: Filter to narrow down sites")

    # Two-column layout for nicer UI
    col1, col2 = st.columns(2)

    with col1:
        # Step 1: Country filter
        countries = sorted(df["Country"].dropna().unique())
        selected_country = st.selectbox("Country", [""] + countries)

        # Step 2: Choose cascade path
        cascade_choice = st.radio(
            "Filter sites by:",
            ["POId", "Developer"],
            horizontal=True
        )

    with col2:
        # Filter dataframe after country selection
        df_country = df if selected_country == "" else df[df["Country"] == selected_country]

        # Step 3a: If cascaded via POId
        if cascade_choice == "POId":
            po_ids = sorted(df_country["POId"].dropna().unique())
            selected_po = st.selectbox("POId", [""] + list(map(str, po_ids)))
            
            # Filter again
            df_po = df_country if selected_po == "" else df_country[df_country["POId"].astype(str) == selected_po]
            filtered_df = df_po

        # Step 3b: If cascaded via Developer
        else:
            developers = sorted(df_country["Developer"].dropna().unique())
            selected_dev = st.selectbox("Developer", [""] + developers)

            # Filter again
            df_dev = df_country if selected_dev == "" else df_country[df_country["Developer"] == selected_dev]
            filtered_df = df_dev

    # Final Step: SiteName (always filtered based on above)
    site_names = sorted(filtered_df["SiteName"].dropna().unique())
    selected_site = st.selectbox("Site Name", [""] + site_names)

    if selected_site:
        st.success(f"Selected site: {selected_site}")   
    
site_row = df[df["SiteName"] == selected_site].iloc[0]

st.write("### Selected Site Summary")
st.write(site_row[['Developer','COD','Latitude','Longitude','Country','Capacity']])

average = st.number_input("PT Static Average (kWh/yr)", value=1520.0, step=0.1)

# Extract values
ym_list = site_row["ym_list"]
energy_list = np.array(site_row["energy_list"])/1e3

# %% Initialize model
model = SolarEnergyInterpolator() 

# ----------------------------------------------------
# Run model + degradation + PF estimation
# ----------------------------------------------------


if st.button("Run Model & Fit PF"):

    latitude = site_row['Latitude']
    longitude = site_row['Longitude']
    capacity = site_row['Capacity']
    COD = site_row['COD']

    
    # 1. Model prediction (12 months)
    case1_vec, case1_total, case2_total, case3_total, case4_total = model.get_solar_energy(latitude, longitude, capacity, COD, average)
    model_12 = case1_vec
    
    # 2. Align to actual months (mapping by month number)
    months = [int(m.split("-")[1]) for m in ym_list]
    aligned = np.array([model_12[m - 1] for m in months])

    # 3. Apply degradation
    degraded = apply_degradation(aligned, ym_list)

    # 4. Find optimal PF
    pf_opt = find_optimal_pf(degraded, energy_list)

    st.success(f"Optimal PF found: **{pf_opt:.4f}**")

    # 5. Plot
    scaled_pred = pf_opt * degraded
    plot_comparison(ym_list, energy_list, scaled_pred)

    # Save state for PF override
    st.session_state["degraded"] = degraded
    st.session_state["energy_list"] = energy_list
    st.session_state["ym_list"] = ym_list
    st.session_state["pf_opt"] = pf_opt


