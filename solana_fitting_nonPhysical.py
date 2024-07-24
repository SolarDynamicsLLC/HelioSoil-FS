# # Notebook demonstrating fitting of $hrz0$ to site data
# The below code shows how to use data from fixed-tilt experiments to compute the free parameter $hrz0$. The dust distribution is currently assumed and its parameters are defined in the file `parameters_qut_experiments.xlsx`.

# In[ ]:


import numpy as np
import soiling_model.base_models as smb
import soiling_model.fitting as smf
import soiling_model.utilities as smu
from matplotlib import rcParams
import matplotlib.pyplot as plt
from scipy.stats import norm

rcParams['figure.figsize'] = (5, 5)
from analysis_scripts.paper_specific_utilities import plot_for_paper, daily_soiling_rate, fit_quality_plots, \
    summarize_fit_quality
import scipy.stats as sps

# %matplotlib qt

save_file_root = "C:\\Users\\limpo\\SolarDynamics-code\\HelioSoil-FS\\results\\solana-fixed\\"
ls_save_file = save_file_root + "ls_fitting_results_solana"
sp_save_file = save_file_root + "sp_fitting_results_solana"
cm_save_file = save_file_root + "cm_fitting_results_solana"
reflectometer_incidence_angle = 15  # angle of incidence of reflectometer
reflectometer_acceptance_angle = 12.5e-3  # half acceptance angle of reflectance measurements
k_factor = 2.404  # calibration factor for TSP measurements in experiments
second_surf = True  # True if using the second-surface model. Otherwise, use first-surface
d = "C:\\Users\\limpo\\SolarDynamics-code\\HelioSoil-FS\\solana-fixed\\"  # directory of parameter files (be sure to follow naming convention)
test_time_at_end = [0, 0, 0, 0]  # amount of test time to leave at the end of each file
parameter_file = d + "parameters_solana_exp.xlsx"

# ## Load in data and divide into training and testing
# 
# Specify mirrors for training, the k-factors of the dust measurements (if any), and the type of dust measurement (PMX or TSP).

# In[ ]:

train_experiments = [0, 1]  # indices for training experiments from 0 to len(files)-1
train_mirrors = ["Mirror_1"]  # which mirrors within the experiments are used for training
files_experiment, training_intervals, mirror_name_list, all_mirrors = smu.get_training_data(d, "experiment_")
dust_type = "TSP"

extract = lambda x, ind: [x[ii] for ii in ind]
files_experiment_train = extract(files_experiment, train_experiments)
training_intervals = extract(training_intervals, train_experiments)
t = [t for t in train_experiments]
training_string = "Training: " + str(train_mirrors) + ", Exp: " + str(t)

# Instantiate model and load in training data

# In[ ]:

sim_data_train = smb.simulation_inputs(files_experiment_train,
                                       k_factors=k_factor,
                                       dust_type=dust_type
                                       )
reflect_data_train = smb.reflectance_measurements(files_experiment_train,
                                                  sim_data_train.time,
                                                  number_of_measurements=9.0,
                                                  reflectometer_incidence_angle=reflectometer_incidence_angle,
                                                  reflectometer_acceptance_angle=reflectometer_acceptance_angle,
                                                  import_tilts=True,
                                                  column_names_to_import=train_mirrors
                                                  )

# Trim training data to specified ranges. The second trim ensures that the weather variables stop at the limits of the reflectance data

# In[ ]:

sim_data_train, reflect_data_train = smu.trim_experiment_data(sim_data_train,
                                                              reflect_data_train,
                                                              training_intervals
                                                              )

sim_data_train, reflect_data_train = smu.trim_experiment_data(sim_data_train,
                                                              reflect_data_train,
                                                              "reflectance_data"
                                                              )

# Load in the total data

# In[ ]:


sim_data_total = smb.simulation_inputs( files_experiment,
                                        k_factors=k_factor,
                                        dust_type=dust_type
                                        )

reflect_data_total = smb.reflectance_measurements(  files_experiment,
                                                    sim_data_total.time,
                                                    number_of_measurements=9.0,
                                                    reflectometer_incidence_angle=reflectometer_incidence_angle,
                                                    reflectometer_acceptance_angle=reflectometer_acceptance_angle,
                                                    import_tilts=True,
                                                    column_names_to_import=None
                                                    )

# ## Constant Mean Desposition Velocity
# ### Compute deposition velocity, angles. Mie weights not needed for constant mean model. 

# In[ ]:


constant_imodel = smf.constant_mean_deposition_velocity(parameter_file)
constant_imodel.helios_angles(sim_data_train,
                              reflect_data_train,
                              second_surface=second_surf)

# extinction weights not needed for constant mean model


# ### Fitting
# MLE or MAP

# In[ ]:


log_param_hat_con, log_param_cov_con = constant_imodel.fit_mle(sim_data_train,
                                                               reflect_data_train,
                                                               transform_to_original_scale=False)
constant_imodel.save(cm_save_file,
                     log_p_hat=log_param_hat_con,
                     log_p_hat_cov=log_param_cov_con,
                     training_simulation_data=sim_data_train,
                     training_reflectance_data=reflect_data_train)

s_con = np.sqrt(np.diag(log_param_cov_con))
param_ci_con = log_param_hat_con + 1.96 * s_con * np.array([[-1], [1]])
lower_ci_con = constant_imodel.transform_scale(param_ci_con[0, :])
upper_ci_con = constant_imodel.transform_scale(param_ci_con[1, :])
param_hat_con = constant_imodel.transform_scale(log_param_hat_con)
mu_tilde, sigma_dep_con = param_hat_con
print(f'mu_tilde: {mu_tilde:.2e} [{lower_ci_con[0]:.2e}, {upper_ci_con[0]:.2e}] [p.p./day]')
print(f'\sigma_dep (constant mean model): {sigma_dep_con:.2e} [{lower_ci_con[1]:.2e},{upper_ci_con[1]:.2e}] [p.p./day]')

# # MAP Fitting
# sigma_m = 1.5
# mu_m = -10.0
# sigma_sigma_dep = 5.0
# mu_sigma_dep = -2.0
# priors =    {   'log_mu_tilde': norm(scale=sigma_m,loc=mu_m),\
#                 'log_sigma_dep': norm(scale=sigma_sigma_dep,loc=mu_sigma_dep)\
#               }

# param_hat,param_cov = constant_imodel.fit_map(  sim_data_train,
#                                                 reflect_data_train,
#                                                 priors,verbose=True,
#                                                 transform_to_original_scale=True)


constant_imodel.update_model_parameters(param_hat_con)
_, _, _ = constant_imodel.plot_soiling_factor(sim_data_train,
                                              reflectance_data=reflect_data_train,
                                              figsize=(10, 10),
                                              reflectance_std='mean',
                                              save_path=save_file_root,
                                              fig_title="On Training Data")

# ### Predict with test data and plot

# In[ ]:


constant_imodel.helios_angles(sim_data_total,
                              reflect_data_total,
                              second_surface=second_surf)

# Extinction weights not needed for constant mean model


# In[ ]:


fig_total, ax_total, _, _, _ = constant_imodel.plot_soiling_factor(sim_data_total,
                                                                   reflectance_data=reflect_data_total,
                                                                   figsize=(12, 15),
                                                                   reflectance_std='mean',
                                                                   save_path=save_file_root + "constant_mean_fitting.png",
                                                                   fig_title=training_string + " (Constant Mean)",
                                                                   return_handles=True,
                                                                   repeat_y_labels=False)

# add lines indicating training times for mirrors 
# and experiments use for training.
for ii, e in enumerate(train_experiments):
    for jj, m in enumerate(all_mirrors):
        if m in train_mirrors:
            a = ax_total[jj, e]
            a.axvline(x=sim_data_train.time[ii][0], ls=':', color='red')
            a.axvline(x=sim_data_train.time[ii][sim_data_train.time[ii].index.max()], ls=':', color='red')

fig_total.subplots_adjust(wspace=0.1, hspace=0.3)

# ## Normalized reflectance plots for the paper

# In[ ]:


fig, ax = plot_for_paper(constant_imodel,
                         reflect_data_total,
                         sim_data_total,
                         train_experiments,
                         train_mirrors,
                         [["N/A", "N/A", "N/A", "N/A", "N/A"] for m in range(4)],
                         # note: these are not the actual orientations (the experimental values are actually the average of two orientations)
                         legend_shift=(0, 0),
                         rows_with_legend=[2],
                         num_legend_cols=4,
                         plot_rh=False)
fig.savefig(cm_save_file + ".pdf", bbox_inches='tight')

# ## High, Medium, Low daily loss distributions from total data

# In[ ]:
"""
pers = [5, 50, 95.0, 100]
labels = ['Low', 'Medium', 'High', 'Maximum']
colors = ['blue', 'green', 'purple', 'black']
fsz = 16

sims, a, a2 = daily_soiling_rate(sim_data_total,
                                 cm_save_file,
                                 M=100000,
                                 percents=pers,
                                 dust_type=dust_type)
# xL,xU = np.percentile(sims,[0.1,99.9])
xL, xU = -0.25, 3.0
lg = np.linspace(xL, xU, 1000)
inc_factor = imodel.helios.inc_ref_factor[0]

fig, ax = plt.subplots()
for ii in range(sims.shape[1]):
    ax.hist(sims[:, ii], 250, density=True,
            alpha=0.5, color=colors[ii],
            label=labels[ii])

    loc = inc_factor * mu_tilde * a[ii]
    s2 = (inc_factor * sigma_dep_con) ** 2 * a2[ii]
    dist = sps.norm(loc=loc * 100, scale=np.sqrt(s2) * 100)
    ax.plot(lg, dist.pdf(lg), color=colors[ii])
    print(f"Loss for {labels[ii]} scenario: {loc * 100:.2f} +/- {1.96 * 100 * np.sqrt(s2):.2f}")

ax.set_xlim((xL, xU))
ax.set_ylabel('Probability Density', fontsize=fsz + 2)
ax.set_xlabel('Loss (percentage points)', fontsize=fsz + 2)
ax.legend(fontsize=fsz)

fig.set_size_inches(5, 4)
fig.savefig(save_file_root + "losses_qut.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
"""
# # Fit Quality Assessments

# In[ ]:


# %% Fit Quality plots (constant-mean)
mirror_idxs = list(range(len(all_mirrors)))
test_experiments = [f for f in list(range(len(files_experiment))) if f not in train_experiments]
train_mirror_idx = [m for m in mirror_idxs if all_mirrors[m] in train_mirrors]
test_mirror_idx = [m for m in mirror_idxs if all_mirrors[m] not in train_mirrors]

fig, ax = summarize_fit_quality(constant_imodel,
                                reflect_data_total,
                                train_experiments,
                                train_mirror_idx,
                                test_mirror_idx, test_experiments,
                                min_loss=-2,
                                max_loss=6.0,
                                save_file=sp_save_file,
                                figsize=(10, 10),
                                include_fits=False)
for a in ax:
    a.set_xticks([0, 2, 4, 6])
    a.set_yticks([0, 2, 4, 6])

# In[ ]:


fig, ax = plt.subplots(figsize=(6, 6))

fit_quality_plots(constant_imodel,
                  reflect_data_total,
                  test_experiments,
                  test_mirror_idx + train_mirror_idx,
                  ax=ax,
                  min_loss=-2,
                  max_loss=15.0,
                  include_fits=False,
                  data_ls='k*',
                  data_label="Testing data",
                  replot=True,
                  vertical_adjust=-0.1,
                  cumulative=True)

fit_quality_plots(constant_imodel,
                  reflect_data_total,
                  train_experiments,
                  test_mirror_idx,
                  ax=ax,
                  min_loss=-2,
                  max_loss=15.0,
                  include_fits=False,
                  data_ls='g.',
                  data_label="Training (different tilts)",
                  replot=False,
                  vertical_adjust=-0.05,
                  cumulative=True)

fit_quality_plots(constant_imodel,
                  reflect_data_total,
                  train_experiments,
                  train_mirror_idx,
                  ax=ax,
                  min_loss=-2,
                  max_loss=15.0,
                  include_fits=False,
                  data_ls='m.',
                  replot=False,
                  data_label="Training",
                  cumulative=True)

ax.set_xlabel("Measured cumulative loss", fontsize=16)
ax.set_ylabel("Predicted cumulative loss", fontsize=16)
fig.savefig(cm_save_file + "_cumulative_fit_quality.pdf", bbox_inches='tight')

fig, ax = plt.subplots(figsize=(6, 6))
fit_quality_plots(constant_imodel,
                  reflect_data_total,
                  test_experiments,
                  test_mirror_idx,
                  ax=ax,
                  min_loss=-2,
                  max_loss=6.0,
                  include_fits=False,
                  data_ls='k*',
                  data_label="Testing data",
                  replot=True,
                  vertical_adjust=-0.1,
                  cumulative=False)

fit_quality_plots(constant_imodel,
                  reflect_data_total,
                  train_experiments,
                  test_mirror_idx,
                  ax=ax,
                  min_loss=-2,
                  max_loss=6.0,
                  include_fits=False,
                  data_ls='g.',
                  data_label="Training (different tilts)",
                  replot=False,
                  vertical_adjust=-0.05,
                  cumulative=False)

fit_quality_plots(constant_imodel,
                  reflect_data_total,
                  train_experiments,
                  train_mirror_idx,
                  ax=ax,
                  min_loss=-2,
                  max_loss=6.0,
                  include_fits=False,
                  data_ls='m.',
                  replot=False,
                  data_label="Training",
                  cumulative=False)

# ax.set_title("Loss change prediction quality assessment (constant mean)")
ax.set_xlabel(r"Measured $\Delta$loss")
ax.set_ylabel(r"Predicted $\Delta$loss")
