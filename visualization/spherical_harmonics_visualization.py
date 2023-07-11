import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.special import sph_harm
from matplotlib import animation
from matplotlib.colors import LightSource
from IPython import display
import matplotlib

def get_SphHarm(l, m, component, thetas, phis):
    ## Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
    if component=='abs':
        rx = getattr(sph_harm(m, l, phis, thetas), 'real')
        ix = getattr(sph_harm(m, l, phis, thetas), 'imag')
        fcolors = np.sqrt(rx*rx+ix*ix)
    else:
        fcolors = getattr(sph_harm(m, l, phis, thetas), component)
    wavefun = fcolors
    fmax, fmin = fcolors.max(), fcolors.min()
    fcolors = (fcolors - fmin)/(fmax - fmin)
    ls = LightSource(180, 45)
    rgb = ls.shade(fcolors, cmap=cm.seismic, vert_exag=0.1, blend_mode='soft')
    if np.unique(wavefun).size==1:
        rgb = np.tile(np.array([1,0,0,1]), (rgb.shape[0],rgb.shape[1],1))
    return wavefun,rgb
    
def generate_animation(rgb, r, fig, ax, mode, cartesian_coordinates):
    ## Generate animation
    
    x = cartesian_coordinates[0]
    y = cartesian_coordinates[1]
    z = cartesian_coordinates[2]
    
    def init():
        for rgbi,axi,ri in zip(rgb,ax,r):
            ri = np.abs(np.squeeze(np.array(ri)))
            if mode=='sphere':
                axi.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=rgbi, shade=False)
            if mode=='radial':
                axi.plot_surface(np.multiply(ri,x), np.multiply(ri,y), np.multiply(ri,z),  rstride=1, cstride=1, facecolors=rgbi, shade=False)
        return fig,
    def animate(i):
        for axi in ax:
            axi.view_init(elev=30, azim=0+10*i)
        return fig,
    ani = animation.FuncAnimation(fig, animate, init_func=init,
                           frames=36, interval=50, blit=True)
    return ani


def visualizeAllHarmonicsOfDegree(degree, mode='sphere', resolution=50):
    """
    Visualizes spherical harmonics of degree l and all possible orders m.
    
    Parameters:
    degree (int): specifies degree of spherical harmonics to be visualized
    mode (string): specifies mode of the visualization values: sphere, radial
    resolution (int): specifies resolution of the visualization
    
    Returns:
    (None)
    """

    thetas = np.linspace(0, np.pi, resolution)
    phis = np.linspace(0, 2*np.pi, resolution)
    thetas, phis = np.meshgrid(thetas, phis)

    # The Cartesian coordinates of the unit sphere
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    cartesian_coordinates = (x,y,z)

    for order in range(-degree,degree+1):
        print(degree,order)
        fig = plt.figure(figsize=plt.figaspect(0.5))

        realR,realPart = get_SphHarm(degree, order, 'real', thetas, phis)
        imagR,imagPart = get_SphHarm(degree, order, 'imag', thetas, phis)

        realAx = fig.add_subplot(121, projection='3d')
        imagAx = fig.add_subplot(122, projection='3d')
        allaxs = [realAx, imagAx]

        for eachAx in allaxs:
            eachAx.set_axis_off()

        ani = generate_animation([realPart,imagPart],[realR,imagR],fig,allaxs,mode,cartesian_coordinates)

        ## Display animation
        print('Drawing ...')

        video = ani.to_html5_video()
        html = display.HTML(video)
        display.display(html)
        plt.close(fig)
        
def plotAllHarmonicsUpToDegree(max_degree, resolution=50):
    """
    Plots spherical harmonics up to degree l and all possible orders m.
    
    Parameters:
    max_degree (int): specifies maximum degree of spherical harmonics to be visualized
    resolution (int): specifies resolution of the visualization
    
    Returns:
    (None)
    """
    phis = np.linspace(0, np.pi, resolution)
    thetas = np.linspace(0, 2*np.pi, resolution)
    phis, thetas = np.meshgrid(phis, thetas)

    # The Cartesian coordinates of the unit sphere
    x = np.sin(phis) * np.cos(thetas)
    y = np.sin(phis) * np.sin(thetas)
    z = np.cos(phis)
    
    fig = plt.figure(figsize=(10, 5))
    
    number_of_plot_rows = max_degree + 1
    number_of_plot_columns = max_degree*2 + 1
    subplot_counter = 1
    
    for degree in range(max_degree+1):
        for order in range(-degree,degree+1):
            # Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
            fcolors = sph_harm(order, degree, thetas, phis).real
            fmax, fmin = fcolors.max(), fcolors.min()
            fcolors = (fcolors - fmin)/(fmax - fmin)
            
            fcolors = np.nan_to_num(fcolors,nan=0.0)

            # Add subplot and plot the sphere
            ax = fig.add_subplot(number_of_plot_rows, number_of_plot_columns, subplot_counter, projection='3d')
            ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.gist_rainbow(fcolors))
            
            # Turn off the axis planes
            ax.set_axis_off()
            
            subplot_counter += 1
        
        # Switch to the next row of plots
        subplot_counter += number_of_plot_columns-(degree*2)-1
            
    # Create colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    sm = cm.ScalarMappable(cmap='gist_rainbow')
    sm.set_clim(vmin=fmin, vmax=fmax)
    cbar = fig.colorbar(sm, cax=cbar_ax)

    plt.show()
