#%% 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GSA_Interpolator import SolarEnergyInterpolator


#%% Load data

# df = pd.read_pickle('main.pkl')
df = pd.read_pickle('main_wCAResults.pkl')


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

    fig, (ax1, ax2) = plt.subplots(2, 1, layout="constrained")

    # Convert to readable month labels
    dates = pd.to_datetime(ym_list)
    month_labels = dates.strftime("%b\n%Y")  # e.g. Jan\n2024

    # Plot 1: Timeseries of actual vs predicted
    # Plot actual vs predicted
    ax1.plot(month_labels, actual, label="Actual Energy", linewidth=2, marker='*', markersize=10)
    ax1.plot(month_labels, predicted, label="Model (scaled)", linestyle="--", linewidth=2, marker='o', markersize=10)

    # Axis labels & title
    ax1.set_xlabel("Month", fontsize=12)
    ax1.set_ylabel("Energy (kWh)", fontsize=12)

    # Tick labels
    ax1.tick_params(axis='both', labelsize=10)

    # Grid & legend
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Cummulative Energy of actual vs predicted
    actual_cum = np.cumsum(actual)
    predicted_cum = np.cumsum(predicted)
    
    ax2.plot(month_labels, actual_cum, label="Actual Energy", linewidth=2, marker='*', markersize=10)
    ax2.plot(month_labels, predicted_cum, label="Model (scaled)", linestyle="--", linewidth=2, marker='o', markersize=10)

    # Axis labels & title
    ax2.set_xlabel("Month", fontsize=12)
    ax2.set_ylabel("Cummulative Energy (kWh)", fontsize=12)

    # Tick labels
    ax2.tick_params(axis='both', labelsize=10)

    # Grid & legend
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    st.pyplot(fig)
    
def plot_portfolio(pf_opt, tee, df):

    # Use dark background style
    plt.style.use("dark_background")

    fig, (ax1, ax2) = plt.subplots(2, 1, layout="constrained")
    
    # Constrain DF
    df = df[df['optimal_pf'] < 2]

    # Plot 1: Optimal PF
    n, bins, patches = ax1.hist(df['optimal_pf'], bins=50, edgecolor='black', alpha=0.7, label="Portfolio Distribution")
    ax1.vlines(x = pf_opt, ymin=0, ymax=max(n),linestyles='--', linewidth = 2.5, label="Current Value")

    # Axis labels & title
    ax1.set_xlabel("PF = Actual/Model's Ideal", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot 2: TEE

    n, bins, patches = ax2.hist(100*df['tee'], bins=50, edgecolor='black', alpha=0.7, label="Portfolio Distribution")
    ax2.vlines(x = tee*100, ymin=0, ymax=max(n),linestyles='--', linewidth = 2.5, label="Current Value")

    # Axis labels & title
    ax2.set_xlabel("Cummulative Energy Error [%]", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    st.pyplot(fig)

# ----------------------------------------------------
# Streamlit App
# ----------------------------------------------------

st.title("Model vs. Data: TS Analyzer")

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

    st.success(f"Performance Factor: **{pf_opt:.4f}**")

    # 5. Plot
    scaled_pred = pf_opt * degraded
    
    st.write("Model vs Data Comparison")
    plot_comparison(ym_list, energy_list, scaled_pred)
        
    # Calculate total error
    tee = np.abs(np.sum(scaled_pred) - np.sum(energy_list))/np.sum(energy_list)

    # Save state for PF override
    st.session_state["degraded"] = degraded
    st.session_state["energy_list"] = energy_list
    st.session_state["ym_list"] = ym_list
    st.session_state["pf_opt"] = pf_opt
    
    # Write output values
    st.success(f"Aggregate Energy Error: {tee*100:.1f}%")
    if tee < 0.2:
    # Displays a green box with an "OK" icon
        st.success('Status: OK')
    else:
    # Displays a red box with an "X" icon
        st.error('Status: CHECK')
    
    st.write("Compare values with Portfolio")    
    # Compare current values with portfolio    
    plot_portfolio(pf_opt, tee, df)

