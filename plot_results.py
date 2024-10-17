from functions_MSF import *

limit_threads_to_N(14)

# Insert here the name of the name of the configuration file
configuration_file = 'TEST.yml'

# Load the configuration file
test = LoadConfig(configuration_file)

# Load the dataframe usign the MSFDataHandler class. The latter can also operate on the saved files removing them according
# a criterion etc...
handler = MSFDataHandler(LoadConfig(configuration_file))
df = handler.return_dataframe()

# Filter the dataframe to plot the results presented on the paper
df_alpha0_0 = df[df['alpha0'] == 0]
df_mu0im_range = df_alpha0_0[(df_alpha0_0['mu0im'] >= -2) & (df_alpha0_0['mu0im'] <= 2)]

# This function plots the results of the paper, the flag start_with_0_01 is used to plot only the intervals that are
# accompained by an analythical analysis around 0
plot_b_star_intervals(df_mu0im_range, start_with_0_01=True)
