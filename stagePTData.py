#%% 
import pandas as pd

# load data
site = pd.read_csv('PT_site.csv')
ts = pd.read_csv('PT_CMR.csv')
po = pd.read_csv('PT_sitePO.csv')

#%% pre-process site
# 1. Drop duplicate sites (possibly with multiple inverters)
site.drop_duplicates(subset='Id', inplace= True, keep='first')

# 2. Cast (possibly) values to numeric
site['Capacity'] = pd.to_numeric(site['Capacity'], errors='coerce')
site['Latitude'] = pd.to_numeric(site['Latitude'], errors='coerce')
site['Longitude'] = pd.to_numeric(site['Longitude'], errors='coerce')

# 3. Drop unnecessary columns
columns_to_keep = ['Id','Developer','SiteName','COD','Latitude','Longitude','Country','Capacity']
site = site[columns_to_keep]

site = pd.merge(site, po[['SiteId','POId']], left_on= 'Id', right_on='SiteId', how='inner')

#%% pre-process meter readings


ts.dropna(subset=['CertStartDt','CertEndDt'], inplace=True)

if not pd.api.types.is_datetime64_any_dtype(ts['CertStartDt']):
  ts['CertStartDt'] = ts['CertStartDt'].str.replace('Sept', 'Sep')
  ts['CertEndDt'] = ts['CertEndDt'].str.replace('Sept', 'Sep')

# Convert 'start' and 'end' columns to datetime format
ts['CertStartDt'] = pd.to_datetime(ts['CertStartDt'], format="%b %d, %Y, %I:%M:%S %p")
ts['CertEndDt'] = pd.to_datetime(ts['CertEndDt'], format="%b %d, %Y, %I:%M:%S %p")

# Calculate the time duration in months for each row
ts['duration_days'] = (ts['CertEndDt'] - ts['CertStartDt']).dt.days
ts['duration_months'] = (ts['CertEndDt'] - ts['CertStartDt']).dt.days / 30.44
ts["year"] = ts["CertStartDt"].dt.year
ts["month"] = ts["CertStartDt"].dt.month

# Filter readings which are longer than a month
ts = ts[ts['duration_days'] <= 31].copy()
ts["year"] = ts["CertStartDt"].dt.year
ts["month"] = ts["CertStartDt"].dt.month

# Group by site-year-month
monthly = (
    ts.groupby(["SiteId", "year", "month"])
      .agg(
          total_energy=("Volume(Wh)", "sum"),
          total_days=("duration_days", "sum")
      )
      .reset_index()
)

# Keep only months with >= 5 recorded days
monthly = monthly[monthly["total_days"] >= 5]

#%% Define function built lists for monthly lists
def build_lists(group):
    group = group.sort_values(["year", "month"])

    # Year-month string for easy alignment & plotting
    ym = group["year"].astype(str) + "-" + group["month"].astype(str).str.zfill(2)

    return pd.Series({
        "ym_list": ym.tolist(),
        "energy_list": group["total_energy"].tolist()
    })

result = monthly.groupby("SiteId").apply(build_lists).reset_index()

#%% Merge ts and site datasets

main = pd.merge(site, result, right_on='SiteId',left_on='Id')
main['num_readings'] = main['ym_list'].apply(len)
main.drop_duplicates(subset='SiteName', inplace=True)
# Save pre-processed datase
main.to_pickle('main.pkl')

# %%
