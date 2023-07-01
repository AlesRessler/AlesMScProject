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