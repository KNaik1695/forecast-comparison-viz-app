#%%
from scipy.interpolate import RegularGridInterpolator
import numpy as np  
from dateutil.parser import parse
# import os 

# os.chdir("path for pv_potential_3d")

#%% Interpolator definition

class SolarEnergyInterpolator:
    def __init__(self):
        """
        1. Loads 3D grid file
        2. Initializes interpolator
        """
        data = np.load("pv_potential_3d.npz")
        # print("File loaded")
        
        # Initialize lats, lons
        pv_data = data["pv_data"]
        lons = data["lons"]
        lats = data["lats"]
        months_axis = np.arange(1, 13)

        # Create the interpolator (ignoring NaNs)
        self.interpolator = RegularGridInterpolator(
            (lons, lats, months_axis),
            pv_data,
            method='linear',  # Can be 'nearest' or 'cubic' as well
            bounds_error=False, # Does NOT raise error when input is out of bounds
            fill_value=0  # Will return 0 if out of bounds
        )
        # print("Grid interpolator initialized")

    def get_solar_energy(self, latitude, longitude, capacity, COD, staticAvg):
        """
        Interpolates the solar energy for the given lat, long, and month of year.

        Parameters:
            latitude (float): Latitude of the query point. (-60 to 65)
            longitude (float): Longitude of the query point. (-180 to 180)
            capacity: Solar capacity of the farm in kWp
            COD (date): Commercial Operation Date
            month (float): Month of the year of the query point. (1 to 12)

        Returns:
            float: Interpolated solar energy value or 0 if out of range.
        """

        
        days_vec = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
        # Case 1: Solar GSA (1-year yield)
        month = list(range(1,13))                                                               # Create month vector
        specEnergy = self.interpolator(( longitude, latitude, month))                           # Compute specific energy
        case1_vec = [capacity*specEnergy[i] * days_vec[i] for i in range(len(specEnergy))]      # Year 1 energy vector
        case1_total = round(sum(case1_vec), 2) 
        

        # Case 2: Solar GSA (COD to end-of-year yield)
        parsed_date = parse(COD)                                                                # Can be unknown format string
        # calculated from two parts: partial month (month 1) + full months (month 2 to end)
        # Partial month (calculate energy for the first month)
        month_number = parsed_date.month 
        partial_month_days = days_vec[month_number - 1] - parsed_date.day + 1                   # days in the partial month (min = 1 day)
        specEnergy_partial = self.interpolator(( longitude, latitude, month_number))
        partial_month_Egen = capacity*specEnergy_partial*partial_month_days
        
        # Full months
        if month_number < 12:
            month_list = list(range(month_number+1,13))                                         # computed only second month onwards
            specEnergy = self.interpolator(( longitude, latitude, month_list))                  # Compute specific energy
            relevant_days_vec = days_vec[month_number:12]                                       # Days vector for month 2 to EOY
            month2toEOY_vec = [capacity*specEnergy[i] * days_vec[i] for i in range(len(specEnergy))]  # COD to end-of-year vec
            full_months_Egen = round(sum(month2toEOY_vec), 2)  
        else:
            relevant_days_vec = [0]
            full_months_Egen = 0
        case2_total = round(partial_month_Egen + full_months_Egen, 2)
        
        # Case 3: Linear Regression (1-year yield)
        case3_total = capacity*staticAvg
        
        
        # Case 4: Linear Regression (COD to end-of-year yield)
        full_month_days = sum(relevant_days_vec)
        days_elapsed = partial_month_days + full_month_days
        case4_total = capacity*staticAvg*(days_elapsed/365)
        
        return case1_vec, case1_total, case2_total, case3_total, case4_total


