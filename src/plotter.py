# Plotting code here for testing / debugging purposes
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_500mb_field(ds, var_name, title=""):
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-95)}
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3)

    data = ds[var_name]
    lats = ds['latitude'].values
    lons = ds['longitude'].values

    cf = ax.contourf(lons, lats, data, levels=20,
                     transform=ccrs.PlateCarree(), cmap='RdYlBu_r')
    cs = ax.contour(lons, lats, data, levels=20,
                    transform=ccrs.PlateCarree(), colors='black', linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%d')

    plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, label=var_name)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()