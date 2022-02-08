# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:04:52 2020

@author: boillat
"""

import numpy as np

#------------------------------------------------------------------------------

def voigt(xbase, amp, sigma, gamma, mu, return_mtf=False):
    """
    Returns a Voigt function profile for the given x range values. If
    the return_mtf option is true, returns the MTF instead.

    Parameters
    ----------
    xrange : numpy array
        x values for the Voigt function. Must be equally spaced.
    amp : float
        Amplitude (maximum value)
    sigma : float
        Sigma value of the Gaussian part
    gamma : float
        Gamma value of the Lorentzian part
    mu : float
        Position of the peak
    return_mtf : boolean
        If True, returns the MTF instead of the spatial domain function

    Returns
    -------
    Tuple of 2 numpy arrays
        The first array contains the x values (of k values if the return_mtf
        option is True). The second array contains the function values
        (real number in the default case, complex number if the MTF is returned)

    """
    
        # Get the min, max and spacing value of the x range
        # (assumes the data is evenly spaced)
    xrange = [np.min(xbase), np.max(xbase),
              (np.max(xbase)-np.min(xbase))/(xbase.size-1)]
    
        # compute the effective values (if the spacings were 1)
    sig_eff = sigma/xrange[2]
    gam_eff = gamma/xrange[2]
    mu_eff = (mu - xrange[0])/xrange[2]-int(xbase.size/2)
    
        # use a number of points double the size of the space domain to avoid
        # "roll around" issues
    n = 2*(xbase.size)
    fbase = np.arange(-n/2, n/2)/n
    
    ffunc = np.exp(-2*np.pi**2*sig_eff**2*fbase**2 - np.pi*gam_eff*np.abs(fbase)
                      -2*np.pi*1j*mu_eff*fbase)
    
    if return_mtf:
        return fbase, ffunc
    
    else:
    
        ffunc = amp*ffunc/np.average(np.abs(ffunc))
        
        yfunc = np.abs(np.fft.fftshift(np.fft.ifft(ffunc)))
        
        i0 = int(n/2-int(xbase.size/2))
        i1 = i0 + xbase.size
        
        return yfunc[i0:i1]

#------------------------------------------------------------------------------

def get_lsf(img, xmax=15):
    """
    Extracts the line spread function (LSF) from a linear edge in an image
    This function uses the Canny edge detector followed by a Hough transform
    to extract the most prominent linear edge from the image. After this,
    the edge spread function is measured by averaging all pixel values within
    bands parallele to the detected edges, and the LSF is finally obtained as
    the first derivative of the ESF.

    Parameters
    ----------
    img : 2D numpy array
        Input image
    xmax : int, optional
        Maximum x coordinate of the LSF (minimum will be -xmax). The default is 15.

    Returns
    -------
    xbase_lsf : 1D numpy array
        x coordinates of the returned LSF.
    lsf : 1D numpy array
        values of the LSF.
    detected_line : array of arrays
        coordinates of the detected line in the form [[x0,x1],[y0,y1]].

    """

    from skimage.transform import hough_line, hough_line_peaks
    from skimage.feature import canny
    
        # detection of the edges (Canny)
    edges = canny(img)
    
        # Hough transform (angles from -90° to +90°, 0.1° steps)
    theta_vals = np.linspace(-np.pi/2, np.pi/2, 1800)
    h, theta, d = hough_line(edges, theta=theta_vals)
    
        # extract the peaks (keep only the first one)
    _, a, d = hough_line_peaks(h, theta, d)
    angle = float(a[0])
    dist = float(d[0])
    
        # compute the coordinates of the corresponding line
    xv = np.array((0, img.shape[1]))
    yv = (dist - xv * np.cos(angle)) / np.sin(angle)
    
        # return the detected line
    detected_line = [xv, yv]
    
        # coordinates of a unity length vector parallel to this line
    xb, yb = [xv[1]-xv[0], yv[1]-yv[0]]/((xv[1]-xv[0])**2 + (yv[1]-yv[0])**2)**0.5
    
        # for each pixel, compute the coordinate in the direction perpendicular
        # to the detected line
    dmap = np.array([[(x-xv[0])*yb - (y-yv[0])*xb for x in range(img.shape[1])]
        for y in range(img.shape[0])])
    
        # x coordinates for the ESF calculation (calculation points set on half
        # pixels so the LSF point will fall on integral values)
    xbase_esf = np.linspace(-xmax-0.5, xmax+0.5, 2*xmax+2)
    xbase_lsf = xbase_esf[:-1]+0.5
    
        # compute each ESF point as the average of all pixel values within a
        # 1 pixel wide band parallel to the detected line
    esf = np.array([np.average(img[np.where(np.abs(dmap-x) <= 0.5)])
                for x in xbase_esf])
    
        # if the ESF is in decreasing direction, invert it
    if esf[-1] < esf[0]:
        esf = -esf
        
        # compute the LSF as the derivate of the ESF
    lsf = (esf[1:] - esf[:-1])
    
        # return the values
    return xbase_lsf, lsf, detected_line

#------------------------------------------------------------------------------

def fit_lsf(xbase, lsf):
    """
    
    Fits the line spread function (LSF) with a Voigt function.

    Parameters
    ----------
    xbase : numpy array
        x values for the LSF
    lsf : numpy array
        LSF values

    Returns
    -------
    c : 4 elements array
        Fit coefficients in the form [amplitude, sigma, gamma, mu]

    """
    
    from scipy.optimize import curve_fit
    
        # perform the fit
    c, cov = curve_fit(voigt, xbase, lsf, [np.max(lsf), 1.0, 1.0, 0.0])
    
        # if the sigma value is negative, invert it
    if c[1] < 0:
        c[1] = -c[1]
        
        # return the coefficients
    return c

#------------------------------------------------------------------------------

def meas_resol(img, roi_def=None, xmax=15, silent=False, pix_size=None):
    
    import matplotlib.pyplot as plt
    from matplotlib import cm

        # if 'roi_def' is None or a single ROI, make a 1-element list out of it 
    if roi_def is None:
        roi_list = [None]
    elif type(roi_def[0]) == list or (isinstance(roi_def[0], tuple)):
        roi_list = roi_def
    else:
        roi_list = [roi_def]
        
        # prepare empty results list
    sigma_vals = []
    gamma_vals = []
    mtf_vals = []
        
        # loop for all ROIs
    for roi in roi_list:
        
            # get the sub-image corresponding to the ROI, if defined
        if roi is None:
            sub_img = img
        else:
            sub_img = img[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
        
            # get the corresponding line spread function
        xbase, lsf, edge = get_lsf(sub_img, xmax)
        
            # fit the line spread function
        coefs = fit_lsf(xbase, lsf)
        
            # get the sigma and gamma values
        sigma = coefs[1]
        gamma = coefs[2]
        
            # compute the 10% MTF cutoff
        mtf = ((gamma**2 + 8*np.log(10)*sigma**2)**0.5 - gamma)/(4*np.pi*sigma**2)
        
            # if the pixel size is defined, convert to line pairs / mm
        if pix_size is not None:
            mtf = mtf*1000/pix_size
            sigma = sigma*pix_size
            gamma = gamma*pix_size
            
            # append to the results list
        sigma_vals.append(sigma)
        gamma_vals.append(gamma)
        mtf_vals.append(mtf)
        
        if not silent:
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
            
            ax1.imshow(sub_img, cmap=cm.gray)
            ax1.plot(edge[0], edge[1], 'y')
            ax1.set_axis_off()
            ax1.set_xlim(edge[0])
            ax1.set_ylim((sub_img.shape[0], 0))
                
            xbase2 = np.linspace(-xmax, xmax, 20*xmax+1)
            lsf_fit = voigt(xbase2, *coefs)
            
            if pix_size is not None:
                ax2.plot(xbase*pix_size, lsf, 'r+')
                ax2.plot(xbase2*pix_size, lsf_fit, *coefs)
                ax2.set_xlabel('Position [um]')
            else:
                ax2.plot(xbase, lsf, 'r+')
                ax2.plot(xbase2, lsf_fit, *coefs)
                ax2.set_xlabel('Position [pix]')
            ax2.set_ylabel('Line spread function (LSF) [a.u.]')
            ax2.legend(['Measured LSF', 'Voigt fit'])
            ax2.set_ylim(-0.1*coefs[0], 1.1*coefs[0])
            
            kbase, mtf_func = voigt(xbase, *coefs, return_mtf=True)
            
            if pix_size is not None:
                ax3.semilogy(kbase*1000/pix_size, np.abs(mtf_func))
                ax3.set_xlabel('Frequency [lp/mm]')
            else:
                ax3.semilogy(kbase, np.abs(mtf_func))
                ax3.set_xlabel('Frequency [lp/pix]')
            ax3.set_ylabel('Modulation transfer function (MTF) [a.u.]')
            
            ax3.plot([0,1.5*mtf], [0.1,0.1], 'k--')
            ax3.plot([mtf,mtf], [0.05,1.1], 'k--')
            
            ax3.set_xlim(0, 1.5*mtf)
            ax3.set_ylim(0.05,1.1)
            
            dunit = 'pix' if pix_size is None else 'um'
            funit = 'lp/pix' if pix_size is None else 'lp/mm'
            
            if pix_size is None:
                rval = 0.5/mtf
            else:
                rval = 500/mtf
            
            values_text = 'Sigma = ' + '{:.3g}'.format(sigma) + ' ' + dunit + '\n' + \
                'Gamma = ' + '{:.3g}'.format(gamma) + ' ' + dunit + '\n' + \
                '10% MTF: ' + '{:.3g}'.format(mtf) + ' ' + funit + '\n' + \
                'Resolution: ' + '{:.3g}'.format(rval) + ' ' + dunit
            
            ax3.text(0.05*mtf, 0.055, values_text)
            
            plt.tight_layout()
            plt.plot()
            
    return sigma_vals, gamma_vals, mtf_vals
        
        
        
    
    