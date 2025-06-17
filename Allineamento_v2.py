
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import scipy.optimize as opt
import copy
import os
import h5py

def gaus(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2)) 
    
def align(x1, y1, x2, y2, d, plotter):
    # Allineamento utilizzando il primo telescopio come riferimento

    nBins = 250
    # 'c' contiene le coordinate dei due telescopi
    c = [x1, x2, y1, y2]  
    div = []  # Lista per memorizzare la divergenza
    offset_lst = []  # Lista per memorizzare gli offset
    lst_ax = ["x", "y"]
    
    if plotter:
        fig, ax = plt.subplots(1, 2)
        fig.subplots_adjust(hspace=.4)
        fig.set_size_inches(6, 3)
    
    # Loop su x e y
    for i in range(2):
        # Calcolo dell'angolo tra i due telescopi in μrad
        theta = np.arctan((c[2*i+1] - c[2*i]) / d) * 1e6
        h_theta, b_thetas = np.histogram(theta, bins=nBins)
        b_theta = b_thetas[:-1] + (b_thetas[1] - b_thetas[0]) / 2
        
        # Fit gaussiano per il picco angolare
        sigma = np.std(theta)
        mu = np.mean(theta)
        a = np.max(h_theta)
        p0 = [a, mu, sigma]
        
        # Selezione dati per il fit
        logi = (b_theta > (mu - 2 * sigma)) & (b_theta < (mu + 2 * sigma))
        x_fit = b_theta[logi]
        y_fit = h_theta[logi]

        try:
            oP, pC = opt.curve_fit(gaus, xdata=x_fit, ydata=y_fit, sigma=np.sqrt(y_fit), p0=p0, absolute_sigma=True)
        except RuntimeError:
            print(f"Fit failed for axis {lst_ax[i]}. Proceeding without offset correction.")
            oP = [0, 0, 0]  # Valori di fallback in caso di errore

        # Calcolo dell'offset da applicare per l'allineamento
        mu = oP[1]
        offset = d * np.tan(mu / 1e6)  # Convert back to radians for the offset calculation
        offset_lst.append(offset)
        div.append(oP[2])
        
        if plotter:
            ax[i].plot(b_theta, h_theta, ds='steps-mid', color='hotpink', label='Not aligned')
            ax[i].plot(x_fit, gaus(x_fit, *oP), c='k', ls='--')
            ax[i].plot(mu, gaus(oP[1], *oP), '*', color='k', ms=11, label=f'Offset = {offset:.2e} cm')

            # Riplotto dopo l'applicazione dell'offset
            theta_aligned = np.arctan(((c[2*i+1] - offset) - c[2*i]) / d) * 1e6
            h_theta_aligned, b_theta_aligned = np.histogram(theta_aligned, bins=nBins)
            b_theta_aligned = b_theta_aligned[:-1] + (b_theta_aligned[1] - b_theta_aligned[0]) / 2
            ax[i].plot(b_theta_aligned, h_theta_aligned, ds='steps-mid', color='steelblue', label='Aligned')
            
            ax[i].set_xlabel(rf"$\theta_{{{lst_ax[i]}}}$ [$\mu$rad]")
            ax[i].set_ylabel('Entries')
            ax[i].set_xlim(np.mean(theta_aligned) - 10 * sigma, np.mean(theta_aligned) + 10 * sigma)  # Adjust limits based on the fit
            ax[i].grid()
            ax[i].legend()
    
    if plotter:
        plt.tight_layout()
        plt.show()
    
    return offset_lst, div
