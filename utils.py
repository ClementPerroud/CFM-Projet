import matplotlib.pyplot as plt

def composed_bps_returns(_bps_returns):
    final_bps_return = 1
    for bps_return in _bps_returns:
        final_bps_return *= (bps_return/10000 + 1)
    return (final_bps_return - 1)*10000

def plot_matrix(plt_df, x_column, y_column, z_column, agg_function, x_lim = 100, y_lim = 100, c_lim = 100):
    plt_matrix = plt_df.groupby([y_column, x_column]).agg({
        z_column: agg_function
    }).unstack(fill_value=0)
    plt.figure(1, figsize = (8,8))
    
    plt.matshow(plt_matrix.iloc[:x_lim, :y_lim], fignum=1)
    plt.colorbar()
    plt.clim(-c_lim, c_lim)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
