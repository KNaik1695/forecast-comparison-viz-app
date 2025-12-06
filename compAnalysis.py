#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GSA_Interpolator import SolarEnergyInterpolator
import matplotlib.colors as mcolors

# Load datasets

df = pd.read_pickle('main.pkl')

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

#%% Initialize model
model = SolarEnergyInterpolator() 

#%% Run batch analysis for all sites

def compute_optimal_pf_for_dataset(df):

    optimal_pfs = []
    total_energy_error = []
    average = 1520.0
    i = 0

    df['prediction'] = pd.Series([None] * len(df), dtype=object)
    df['nrmse'] = pd.Series([None] * len(df), dtype=object)
    
    for _, row in df.iterrows():
        
        latitude = row['Latitude']
        longitude = row['Longitude']
        capacity = row['Capacity']
        COD = row['COD']
        
        ym_list = row["ym_list"]
        energy_list = np.array(row["energy_list"])/1e3
        
        # 1. Model prediction (12 months)
        case1_vec, _, _, _, _ = model.get_solar_energy(latitude, longitude, capacity, COD, average)
        model_12 = case1_vec
    
        # 2. Align to actual months (mapping by month number)
        months = [int(m.split("-")[1]) for m in ym_list]
        aligned = np.array([model_12[m - 1] for m in months])
        
        # 3. Apply degradation
        degraded = apply_degradation(aligned, ym_list)

        # 4. Find optimal PF
        pf_opt = find_optimal_pf(degraded, energy_list)
        
        optimal_pfs.append(pf_opt)
        
        #  Save prediction and error values
        curr_index = row.name
        df.at[curr_index, 'prediction'] = aligned
        nrmse = np.sqrt(np.mean(((pf_opt*aligned)-energy_list)**2))/np.mean(energy_list)
        df.at[curr_index, 'nrmse'] = nrmse
        
        tee_row = np.abs(np.sum(pf_opt*aligned) - np.sum(energy_list))/ np.sum(energy_list)
        total_energy_error.append(tee_row)
                
        if i % 100 == 0:
            print(i)
        i += 1

    df["optimal_pf"] = optimal_pfs
    df['tee'] = total_energy_error
    return df
# %% Run batch optimization

df = compute_optimal_pf_for_dataset(df)

# %% Histogram plot

# Plot the histogram
fig, ax = plt.subplots(figsize=(8, 5))
# Pandas hist returns axes, but we need the actual patch objects for coloring
n, bins, patches = ax.hist(df['optimal_pf'], bins=50, ec='black', alpha=0.75, 
                           label='Frequency distribution around mean')

ax.set_xlabel('PF Value')
ax.set_ylabel('Count')

# Apply the custom red-to-green colormap based on frequency proximity to max

# Get the maximum frequency achieved in any bin
max_freq = max(n)
# Define a custom colormap: Red at 0 (low freq), Green at 1 (max freq)
cmap = mcolors.LinearSegmentedColormap.from_list("RedGreenFreq", ["red", "yellow", "green"], N=256)

# Normalize the frequencies to a range between 0 and 1 (0 is min freq, 1 is max freq)
norm = mcolors.Normalize(vmin=min(n), vmax=max_freq)

# Iterate over each bar (patch) and set its color based on its height
for patch, height in zip(patches, n):
    # Map the bar's height to a color using the normalized scale and colormap
    color_value = norm(height)
    patch.set_facecolor(cmap(color_value))


# Display the plot
plt.grid()
plt.show()


# %%

def compute_multi_optimal_pf_for_dataset(df, month_end):

    optimal_pfs = []
    total_energy_error = []
    average = 1520.0
    i = 0

    df['prediction'] = pd.Series([None] * len(df), dtype=object)
    df['nrmse'] = pd.Series([None] * len(df), dtype=object)
    
    for _, row in df.iterrows():
        
        latitude = row['Latitude']
        longitude = row['Longitude']
        capacity = row['Capacity']
        COD = row['COD']
        
        ym_list = row["ym_list"]
        energy_list = np.array(row["energy_list"])/1e3
        
        # 1. Model prediction (12 months)
        case1_vec, _, _, _, _ = model.get_solar_energy(latitude, longitude, capacity, COD, average)
        model_12 = case1_vec
    
        # 2. Align to actual months (mapping by month number)
        months = [int(m.split("-")[1]) for m in ym_list]
        aligned = np.array([model_12[m - 1] for m in months])
        
        # 3. Apply degradation
        degraded = apply_degradation(aligned, ym_list)

        # 4. Find optimal PF
        ind = month_end
        
        pf_opt = find_optimal_pf(degraded[:ind], energy_list[:ind])

        optimal_pfs.append(pf_opt)
        
        #  Save prediction and error values
        curr_index = row.name
        df.at[curr_index, 'prediction'] = aligned
        nrmse = np.sqrt(np.mean(((pf_opt*aligned)-energy_list)**2))/np.mean(energy_list)
        df.at[curr_index, 'nrmse'] = nrmse
        
        tee_row = np.abs(np.sum(pf_opt*aligned) - np.sum(energy_list))/ np.sum(energy_list)
        total_energy_error.append(tee_row)
                
        if i % 100 == 0:
            print(i)
            
        i += 1

    df["optimal_pf"] = optimal_pfs
    df['tee'] = total_energy_error
    return df


#%% 
month_end = 12
df = df[df['num_readings'] > month_end]
df = compute_multi_optimal_pf_for_dataset(df, month_end)

print("NRSME:",round(df['nrmse'].min(),3), round(df['nrmse'].max(),3), round(df['nrmse'].mean(),3))
print("TEE:",round(df['tee'].min(),3), round(df['tee'].max(),3), round(df['tee'].mean(),3))
# %%

num_data_vec = [3,4,5,6,9,12]

dyn_min_vec = []
dyn_max_vec = []
dyn_mean_vec = [] 

tee_min_vec = []
tee_max_vec = []
tee_mean_vec = []

for n in num_data_vec:
    df = df[df['num_readings'] > n]
    df = compute_multi_optimal_pf_for_dataset(df, n)
    
    dyn_min_vec.append(100*round(df['nrmse'].min(),3))
    dyn_max_vec.append(100*round(df['nrmse'].max(),3))
    dyn_mean_vec.append(100*round(df['nrmse'].mean(),3))
    
    tee_min_vec.append(100*round(df['tee'].min(),3))
    tee_max_vec.append(100*round(df['tee'].max(),3))
    tee_mean_vec.append(100*round(df['tee'].mean(),3))

# Optimal PF values    
dyn_min_vec.append(0)
dyn_max_vec.append(100*1.546)
dyn_mean_vec.append(100*0.278)

tee_min_vec.append(0)
tee_max_vec.append(100*0.119)
tee_mean_vec.append(100*0.012)

    
# %% Visualizing the spread of the NRMSE and TEE values over the PF values 

# Re-run code to find data for optimal PF values
df = pd.read_pickle('main.pkl')
df = compute_optimal_pf_for_dataset(df)
# Sort by a thrshold
df = df[df['optimal_pf'] < 1.2]

# For Dynamic Error (nrmse)
import matplotlib.pyplot as plt
# Using the hexbin function to create hexagonal bins (common for 2D histograms)
plt.figure(figsize=(8, 6))
plt.hexbin(df['optimal_pf'], 100*df['tee'], 
           gridsize=30,       # Number of hexagonal bins across the plot
           cmap='inferno',    # Colormap
           mincnt=1)          # Only show bins with at least one point
plt.colorbar(label='Count in Bin')
plt.xlabel('PF', fontsize = 14)
plt.ylabel('Dynamic Error [%]', fontsize = 14)
plt.tick_params(axis='both', labelsize = 12)
plt.show()

# For TEE (cumulative error)
import matplotlib.pyplot as plt
# Using the hexbin function to create hexagonal bins (common for 2D histograms)
plt.figure(figsize=(8, 6))
plt.hexbin(df['optimal_pf'], 100*df['tee'], 
           gridsize=30,       # Number of hexagonal bins across the plot
           cmap='inferno',    # Colormap
           mincnt=1)          # Only show bins with at least one point
plt.colorbar(label='Count in Bin')
plt.xlabel('PF', fontsize = 14)
plt.ylabel('Cummulative Error [%]', fontsize = 14)
plt.tick_params(axis='both', labelsize = 12)
plt.show()

