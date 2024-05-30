# see https://github.com/SolarDynamicsLLC/HelioSoil-FS/blob/main/demo_soiling_model.ipynb

import soiling_model.base_models as smb
import soiling_model.utilities as smu
import matplotlib.pyplot as plt
import numpy as np
import time

d = "woomera_demo/"
file_params = d+"parameters.xlsx"
file_weather = d+'woomera_data.xlsx'

file_SF = d+'SF_woomera_SolarPILOT.csv'             # solar field of 48 sectors located in Woomera
climate_file = d+'woomera_location_modified.epw'    # only used for optical efficiency computation

n_az,n_rad = (8,6)   # number of azimuth and radial sectorization of the solar field
n_trucks = 4         # number of trucks
n_cleans = 10        # number of cleanings in time interval


imodel = smb.field_model(file_params,file_SF,num_sectors=(n_az,n_rad))
sim_data = smb.simulation_inputs(file_weather,dust_type="PM10")
plant = smb.plant()
plant.import_plant(file_params)
imodel.helios.sector_plot()


imodel.sun_angles(sim_data)
imodel.helios_angles(plant)
imodel.helios.compute_extinction_weights(sim_data,imodel.loss_model,verbose=True)


imodel.deposition_flux(sim_data)
imodel.adhesion_removal(sim_data)
imodel.calculate_delta_soiled_area(sim_data)


airT = 20
windS = 5
experiment = 0
imodel.plot_area_flux(sim_data,experiment,airT,windS)


cleans = smu.simple_annual_cleaning_schedule(imodel.helios.tilt[0].shape[0],n_trucks,n_cleans,dt=sim_data.dt[0]/3600.00)
cleans = imodel.reflectance_loss(sim_data,{0:cleans})


soiling_factor = imodel.helios.soiling_factor[0] # zero for the first "run"
field_average_soiling_factor = np.mean(soiling_factor,axis=0)


# imodel.optical_efficiency(plant,sim_data,climate_file,verbose=True,n_az=10,n_el=10)
# field_average_clean_optical_efficiency = np.mean(imodel.helios.optical_efficiency[0]*imodel.helios.nominal_reflectance,axis=0)


# soiled_optical_efficiency = imodel.helios.optical_efficiency[0]*soiling_factor*imodel.helios.nominal_reflectance
# field_average_soiled_optical_efficiency = np.mean(soiled_optical_efficiency,axis=0)


t = sim_data.time[0]
sec_plot = 0 # choose sector for plot

t0 = 96
t1 = 96+96 # hours of the year for the zoomed-in plot

# examine field soiling - field average
fig, ax = plt.subplots()
ax.plot_date(t,field_average_soiling_factor,"-")
ax.set_xlim(t[t0],t[t1])
ax.set_xlabel("Time")
ax.set_ylabel("Field-Averaged Soiling Factor [-]")
plt.xticks(rotation=45)

# examine sector soiling
fig, ax = plt.subplots()
ax.plot_date(t,soiling_factor[sec_plot],"-")
ax.set_xlim(t.iloc[0],t.iloc[-1])
ax.set_xlabel("Time")
ax.set_ylabel("Soiling Factor [-]")
ax.set_title("Sector {0:d}".format(sec_plot))
plt.xticks(rotation=45)