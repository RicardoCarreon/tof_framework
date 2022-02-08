# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 15:50:02 2021

@author: carreon_r
"""

from stack_proc_func import *
from img_utils_4_transmission import *
from plot_cross_sections import *
import pandas as pd


# =============================================================================
# 
#                          dataframe_from_file
#
# takes a file (.txt or csv) and converts it into a dataframe
# =============================================================================
        
def dataframe_from_file (file, columns_list, header= None, sep = ' ', names = ['No name'], skiprows=''):
    import pandas as pd
    if '.txt' in file:
        sep = '\t'
    table = pd.read_csv(file,header=header, usecols=columns_list, sep=sep, names=names, skiprows=skiprows)
    return table

# =============================================================================
# 
#                           get_pulse_windows
#
# reads a ToF and a table of shutter times to separate them into MCP windows 
# =============================================================================
def get_pulse_windows (table_tof, table_shutters):
        # this step keeps the non-zero values in the shutters table
    table_shutters = table_shutters.loc[~(table_shutters==0).all(axis=1)]
        # sum the shutter values horizontally
    table_shutters['sum'] = table_shutters.sum(axis=1)
        # sum cummulatively the values and extract just the 'sum' values that are the boundaries. for more info, check the MCP shutter windows in the calculations folder
    boundaries = list(table_shutters.cumsum(axis=0)['sum'])
        # get the boundaries indexes by searching in the ToF information
    idx_boundaries = []
    for value in boundaries:
            # search for the closest value and give the info in form of an integer to append it in a list
        idx_boundaries.append(list(table_tof.sub(value).abs().idxmin())[0])
            # since python starts from zero and does not take the las value in a sliced list, we add 1
    idx_boundaries = list(np.array(idx_boundaries)+1)
    return idx_boundaries, boundaries


# =============================================================================
#                           get_poly_func
# =============================================================================
    # gives you the polynomial function depending on the degree you want and a list of coefficients
'''
if the list of coefficients is an empty  list, it will give an expression with variables 
'''
def get_poly_func(degree, list_coeff=[]):
        # import sympy
    from sympy import Symbol, sympify, Mul
    import warnings
    warnings.filterwarnings("ignore")
    
        # initialize in 0 the expression
    func = 0

    if list_coeff == []:
            # initialize variables required for the required degree (max 23)
        list_coeff = []
            # dictionary where the variables for the coefficients are
        var_dict = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k','l','m','n','o','p','q','r','s','t','u','v','w']
        
            # slice and take the variables needed for the degree requested
        for i in var_dict[0:degree+1]:
                # create symbols out of each variable and save them in a list
            i = Symbol(i)
            list_coeff.append(i)
            
    if list_coeff != [] and len(list_coeff) != degree+1:
        print('The number of coefficients does not match with the given degree, please check it and try again')
        return

        # initialize x as the main variable
    x = Symbol('x')
    for co in list_coeff:
            # create the expression by addiong the values to it
        func = func + Mul(co*x**degree)
            # reduce the polynomial expression by one degree
        degree-=1
            # five the expression in symbols
    return sympify(func)


# =============================================================================
# 
#                           give_curve_center_mass
#
# function that calculates the x and y coordintes of a curve's center of mass
# =============================================================================
'''
OLD version without Sympify

# def give_curve_center_mass (x_array, y_array, plot_curve = True):
    
#     from scipy.integrate import quad
#         # fit the curve to a trendline of a polynomial equation grade 4
#     coeff = np.polyfit(x_array,y_array,4)
#     xtrend = np.linspace(x_array[0],x_array[-1],100)
#     ytrend = np.polyval(coeff, xtrend)
    
#         # functions to create the integrals to be solved
#     def function(x, a,b,c,d,e):
#         return a*x**4 + b*x**3 + c*x**2 + d*x + e 
    
#     def myfunction(x, a,b,c,d,e):
#         return x*(a*x**4 + b*x**3 + c*x**2 + d*x + e)
    
#     def mxfunction(x, a,b,c,d,e):
#         return 0.5*(a*x**4 + b*x**3 + c*x**2 + d*x + e)**2
    
#         # transform the coefficients into a list and assign to the values to the variables
#     coeff = list(coeff)
#     a,b,c,d,e = coeff
    
#         #solve the integral for the first and last value of the x-axis
#     res, _ = quad(function, x_array[0],x_array[-1], args=(a,b,c,d,e))
#         #solve the integral for the Mx weight
#     resmx, _ = quad(mxfunction, x_array[0],x_array[-1], args=(a,b,c,d,e))
#         #solve the integral for the My weight
#     resmy, _ = quad(myfunction, x_array[0],x_array[-1], args=(a,b,c,d,e))
    
#         # calculate x and y coordinates for the center of mass
#     CMx = resmy/res
#     CMy = resmx/res
    
#         # plot the curve if necessary
#     if plot_curve:
        
#         plt.title('Curve Fit and Center of Mass')
#         plt.plot(x_array,y_array,'b*')
#         #plt.ylim([0,round(np.max(y_array)+1)])
#         plt.xlabel('ToF [s]')
#         plt.plot(xtrend,ytrend,'r-')
#         plt.plot(CMx,CMy, 'r*')
#         plt.grid(True)
#     return CMx,CMy
'''

def give_curve_center_mass (x_array, y_array, degree_fit = 4, plot_curve = True):
    '''
    Here is how to exchange the equations if the center of mass is between 2 curves
    https://tutorial.math.lamar.edu/classes/calcii/centerofmass.aspx#:~:text=The%20center%20of%20mass%20or%20centroid%20of%20a%20region%20is,interval%20%5Ba%2Cb%5D%20.
    '''
    from sympy import integrate, lambdify, sympify, Symbol
        # fit the curve to a trendline of a polynomial equation grade 4
    coeff = np.polyfit(x_array,y_array,degree_fit)
    xtrend = np.linspace(x_array[0],x_array[-1],100)
    ytrend = np.polyval(coeff, xtrend)
        #build the expression to integrate 
    expression= get_poly_func(degree_fit, coeff)
    
    x = Symbol('x')
        #solve the integral for the first and last value of the x-axis
    res = integrate(expression , (x,x_array[0],x_array[-1]))
    
    mx_expression = sympify (0.5*(expression)**2)
    my_expression = sympify(x*(expression))
    
    
        #solve the integral for the Mx weight
    res_mx = integrate(mx_expression,(x,x_array[0],x_array[-1]))
        #solve the integral for the My weight
    res_my = integrate(my_expression,(x,x_array[0],x_array[-1]))
    
        # calculate x and y coordinates for the center of mass
    CMx = res_my/res
    CMy = res_mx/res

        # plot the curve if necessary
    if plot_curve:
        
        plt.title('Curve Fit and Center of Mass')
        plt.plot(x_array,y_array,'b*')
        #plt.ylim([0,round(np.max(y_array)+1)])
        plt.xlabel(' Wavelenth '+ r'[$\AA$]')
        plt.plot(xtrend,ytrend,'r-')
        plt.plot(CMx,CMy, 'r*')
        plt.grid(True)
    return np.float(CMx), np.float(CMy)


# =============================================================================
#                           trapezoid_weight
# =============================================================================
def trapezoid_weight (curve_values, weight_triang = [0,1], weight_top = [1,1], zones_percentage = [0.375,0.25]):
    '''
    takes a curve and gives a weight to their valiues according to a trapezoid shape of neutrons arrival time

    Parameters
    ----------
    curve_values : list
        y-axis values of the curve
    weight_triang : list of 2 int values
        Triangle weights, from the bottom to the top. The default is [0,1].
    weight_top : list of 2 int values
        top of the trapezoid weightening. The default is [1,1].
    zones_percentage : list of 2 int values in percentage
        first value is the triangles, the second is the rectangle. Twice the first value plus the second should give 1.0 . The default is [0.375,0.25].

    Returns
    -------
    weighted_array : list of values
        array weightened with a the shape of a trapezoid.

    '''
    w1 = np.linspace(weight_triang[0],weight_triang[1],int(len(curve_values)*zones_percentage[0]))
    w2 = np.linspace(weight_top[0],weight_top[1],round(len(curve_values)*zones_percentage[1]))
    w3 = np.linspace(weight_triang[1],weight_triang[0],(len(curve_values)-len(w1)-len(w2)))
    weights = [*w1,*w2,*w3]
    weighted_array = np.array(weights)*np.array(curve_values)
    
    return weighted_array


# =============================================================================
#                           get_eff_wavelegnth
# =============================================================================
    

def get_eff_wavelength (table_tof, table_profile, list_boundaries, each_window=False, degree_fit = 4, plot_curve = True, weight_triang = [0,1], weight_top = [1,1], zones_percentage = [0.375,0.25]):

    start = 0
    coordinates_x = []
    coordinates_y = []
    
    if type(list_boundaries) != list:
        x_array = table_tof.tolist()[start : list_boundaries]
        y_array = table_profile.tolist()[start : list_boundaries]
        y_new = trapezoid_weight (y_array,weight_triang = weight_triang, weight_top = weight_top, zones_percentage = zones_percentage)
        cx,cy = give_curve_center_mass (x_array, y_new, degree_fit = degree_fit, plot_curve = plot_curve)
        coordinates_x = cx
        coordinates_y = cy
    else:
        
        if each_window:
            # see coments below
            for idx in list_boundaries:
                x_array = table_tof.tolist()[start : idx]
                y_array = table_profile.tolist()[start : idx]
                y_new = trapezoid_weight (y_array,weight_triang = weight_triang, weight_top = weight_top, zones_percentage = zones_percentage)
                cx,cy = give_curve_center_mass (x_array, y_new, degree_fit = degree_fit, plot_curve = plot_curve)
                coordinates_x.append(cx)
                coordinates_y.append(cy)
    
                start = idx
        else:
                # takes the profile full and make the trapezoid weightening
            y_array = table_profile.tolist()
            y_new = trapezoid_weight (y_array,weight_triang = weight_triang, weight_top = weight_top, zones_percentage = zones_percentage)
                # for each window takes the center of mass 
            for idx in list_boundaries:
                    # extract each window data
                x_array = table_tof.tolist()[start : idx]
                    # take the samevalues that correspond to the x (toF) axis
                y_new_slice = y_new[start : idx]
                    # computes the center of mass
                cx,cy = give_curve_center_mass (x_array, y_new_slice, degree_fit = degree_fit, plot_curve = plot_curve)
                    # save the data in an ordered list
                coordinates_x.append(cx)
                coordinates_y.append(cy)
                    # resets the variable start to take the next point in the boundaries
                start = idx
    return coordinates_x, coordinates_y


# =============================================================================
#                          convert_tof
# =============================================================================

def convert_tof (tof_values, flight_path = 1, result = 'wavelength'):
    conv_res = []
    if result == '':
        print('Please give me the format you want to convert the ToF (wavelength or energy)')
        return
    if type(tof_values) != list:
        if result == 'wavelength':
            conv_res = 3956.03401026312/(flight_path/tof_values)
        if result == 'energy':
            conv_res = ((flight_path/tof_values)/437.393364333048)**2
    else:
        
        if result == 'wavelength':
            for value in tof_values:
                conv_res.append(3956.03401026312/(flight_path/value))
        if result == 'energy':
            for value in tof_values:
                conv_res.append(((flight_path/value)/437.393364333048)**2)
    return conv_res


# =============================================================================
#                          convert_wvl
# =============================================================================

def convert_wvl (wvl_values, flight_path = 1):
    conv_res_tof = []

    if type(wvl_values) != list:
        conv_res_tof = wvl_values / (3956.03401026312/flight_path)

    else:
        for value in wvl_values:
            conv_res_tof.append(value / (3956.03401026312/flight_path))

    return conv_res_tof



# =============================================================================
#                          get_relative_att_stack
# =============================================================================
    
def get_relative_att_stack (trans_imgs_dict, dst_dir, HE_n_LE, proc_folder = []):
    
    import warnings
    warnings.filterwarnings("ignore")
    
        # for the case where no specific folders are given, works with all folders
    if proc_folder == []:
        
            # unpacks the key arguments and transform them into a list
        proc_folder = [*trans_imgs_dict]
        
        # constructs the list of targeted folders and reference
    keep_key(trans_imgs_dict,proc_folder)
    
    HE = HE_n_LE[0]
    LE = HE_n_LE[1]
    
    for key,values in trans_imgs_dict.items():

            # find the border values according to the given boundaries HE and LE
        borders_HE = [item for item in values[0] if format(HE[0], '05d') in item or format(HE[1], '05d') in item]
        borders_LE = [item for item in values[0] if format(LE[0], '05d') in item or format(LE[1], '05d') in item]
        
            #give all the values (including borders) 
        list_HE = [i for i in values[0] if i >= borders_HE[0] and i <= borders_HE[1]]
        list_LE = [i for i in values[0] if i >= borders_LE[0] and i <= borders_LE[1]]
        
            # initialize the averga of images with a matrix of zeros with the img shape
        img_HE = np.zeros(get_img(list_HE[0]).shape)
        img_LE = np.zeros(get_img(list_LE[0]).shape)
        
            # get the average image for HE and LE
        for loc_HE, loc_LE in zip(list_HE,list_LE):
            
            img_HE = img_HE + get_img(loc_HE)
            img_LE = img_LE + get_img(loc_LE)

        img_HE = np.log(img_HE/len(list_HE))
        img_LE = np.log(img_LE/len(list_LE))
        
            # get the relative attenuation image for each point PIXEL WISE
        att_rel_img = np.nan_to_num(img_LE/img_HE)
       
        file_name = os.path.join(dst_dir, key + '_rel_att.fits' )
        write_img(np.flip(att_rel_img, 0), file_name, base_dir='', overwrite=False)
        # slice the dictionary into HE and LE
    return
    


# =============================================================================
#                          eff_wvl_beamline
# =============================================================================
    # gives the effective wavelength of a beamline spectrum according to the number of windows and center of mass

def eff_wvl_beamline( beamline = 'BOA', beam_windows = 5, degree_fit = 4, plot_curve = True, weight_triang = [0,0.99], weight_top = [1,1], zones_percentage = [0.375,0.25]):
    
    if beamline != 'BOA':
        
        print('I do not have additional beamlines information, just BOA')
        return
        
        # get the spectrum and separate it into x and y lists
    spectrum = BoaSpectrum()
    x = spectrum[:,0]
    y = spectrum[:,1]

        # slice the beam according to the number of windows
    sliced_x = np.array_split(x,beam_windows)
    sliced_y = np.array_split(y,beam_windows)
        
        # initialize the variables
    coordinates_x = []
    coordinates_y = []
    
    for x_arr, y_arr in zip (sliced_x, sliced_y):
            
            # calculates the weight according to a trapezoid function 
        y_new = trapezoid_weight (y_arr,weight_triang = weight_triang, weight_top = weight_top, zones_percentage = zones_percentage)
            # calculates the center of mass of the new curve
        cx,cy = give_curve_center_mass (x_arr, y_new, degree_fit = degree_fit, plot_curve = plot_curve)
            # saves the values
        coordinates_x.append(cx)
        coordinates_y.append(cy)
        
    return coordinates_x, coordinates_y
    
    

# =============================================================================
#                          eff_wveff_wvl_expl_beamline
# =============================================================================
    # slices the beam spectrum (experiment) to get the ToF points, which contain the most weight
    # this will give you a list of X Tof points that can be converted to effective wavelenngths with the 'eff_wvl_exp_values' function'

def eff_wvl_exp( spectrum_arr, beam_windows = 5, degree_fit = 4, plot_curve = True, weight_triang = [0,0.99], weight_top = [1,1], zones_percentage = [0.375,0.25]):
    
        # get the spectrum and separate it into x and y lists
    x = spectrum_arr[:,0]
    y = spectrum_arr[:,1]

        # slice the beam according to the number of windows
    sliced_x = np.array_split(x,beam_windows)
    sliced_y = np.array_split(y,beam_windows)

    coordinates_x = []
    coordinates_y = []
    
    for x_arr, y_arr in zip (sliced_x, sliced_y):
        
        y_new = trapezoid_weight (y_arr,weight_triang = weight_triang, weight_top = weight_top, zones_percentage = zones_percentage)
        cx,cy = give_curve_center_mass (x_arr, y_new, degree_fit = degree_fit, plot_curve = plot_curve)
        coordinates_x.append(cx)
        coordinates_y.append(cy)
        
    return coordinates_x, coordinates_y




# =============================================================================
#                          eff_wvl_exp_values
# =============================================================================

    # receives a beam spectrum (array) and the effective wavelenght values in the beamline spectrum and transform the experiment ToF points into effective wavelengths
def eff_wvl_exp_values(tof_values_exp, beamline_wvl_values, flight_path=1):
       
        # transform the points of the beamline wavelengths into ToF
    beamline_tof_values = convert_wvl (beamline_wvl_values, flight_path = flight_path)
    
    new_tof_exp_values = []
    for val in tof_values_exp:
        
            # search for the closest ToF value in the beamline spectrum for each experimental ToF value
        new_tof_exp_values.append(min(beamline_tof_values, key=lambda x:abs(x-val)))

        #transforms the ToF values into wavelengths (effective)
    new_tof_exp_values = convert_tof (new_tof_exp_values, flight_path = flight_path, result = 'wavelength')
    
    return new_tof_exp_values



def BoaSpectrum():
    spectrum = np.array([[0.78878, 0],[0.81036, 0],[0.83194, 0.00028697],[0.85351, 0.0022001],[0.87509, 0.003635],[0.89667, 0.0011479],[0.91825, 0.0023914],[0.93983, 0.0059307],[0.96141, 0.0084178],
    [0.98298, 0.0046872],[1.0046, 0.0088005],[1.0261, 0.011861],[1.0477, 0.015783],[1.0693, 0.021332],[1.0909, 0.031949],[1.1125, 0.038837],[1.134, 0.040654],[1.1556, 0.040559],[1.1772, 0.044863],
    [1.1988, 0.056342],[1.2203, 0.064664],[1.2419, 0.081596],[1.2635, 0.083126],[1.2851, 0.086857],[1.3067, 0.099675],[1.3282, 0.11259],[1.3498, 0.12598],[1.3714, 0.12627],[1.393, 0.13909],[1.4146, 0.15324],
    [1.4361, 0.16903],[1.4577, 0.17132],[1.4793, 0.18165],[1.5009, 0.18557],[1.5224, 0.1917],[1.544, 0.207],[1.5656, 0.21647],[1.5872, 0.23293],[1.6088, 0.2532],[1.6303, 0.25875],[1.6519, 0.27368],
    [1.6735, 0.29328],[1.6951, 0.29175],[1.7166, 0.30639],[1.7382, 0.3304],[1.7598, 0.34035],[1.7814, 0.35259],[1.803, 0.3722],[1.8245, 0.39344],[1.8461, 0.41305],[1.8677, 0.43132],[1.8893, 0.45169],
    [1.9109, 0.43888],[1.9324, 0.46977],[1.954, 0.46872],[1.9756, 0.50182],[1.9972, 0.52822],[2.0187, 0.51684],[2.0403, 0.54993],[2.0619, 0.56112],[2.0835, 0.57069],[2.1051, 0.58427],[2.1266, 0.61661],
    [2.1482, 0.63775],[2.1698, 0.65209],[2.1914, 0.68672],[2.213, 0.69399],[2.2345, 0.72183],[2.2561, 0.74364],[2.2777, 0.74976],[2.2993, 0.76564],[2.3208, 0.77874],[2.3424, 0.80314],[2.364, 0.80113],
    [2.3856, 0.8391],[2.4072, 0.85498],[2.4287, 0.87632],[2.4503, 0.8789],[2.4719, 0.90511],[2.4935, 0.92797],[2.515, 0.93687],[2.5366, 0.94241],[2.5582, 0.95868],[2.5798, 0.97197],[2.6014, 0.96442],[2.6229, 0.97408],
[2.6445, 0.95973],[2.6661, 0.95782],[2.6877, 0.96872],[2.7093, 0.97063],[2.7308, 0.96719],[2.7524, 0.97092],[2.774, 0.95093],[2.7956, 0.97723],[2.8171, 0.9955],[2.8387, 0.97972],[2.8603, 0.96729],[2.8819, 0.97389],
[2.9035, 0.97054],[2.925, 0.98881],[2.9466, 1],[2.9682, 0.97398],[2.9898, 0.98747],[3.0114, 0.97647],[3.0329, 0.98345],[3.0545, 0.98967],[3.0761, 0.95284],[3.0977, 0.95657],[3.1192, 0.96088],[3.1408, 0.93036],
[3.1624, 0.92682],[3.184, 0.90893],[3.2056, 0.9207],[3.2271, 0.9097],[3.2487, 0.90922],[3.2703, 0.90951],[3.2919, 0.87861],[3.3134, 0.88894],[3.335, 0.89277],[3.3566, 0.89573],[3.3782, 0.87278],[3.3998, 0.84484],
[3.4213, 0.84838],[3.4429, 0.83834],[3.4645, 0.85058],[3.4861, 0.83155],[3.5077, 0.82944],[3.5292, 0.82016],[3.5508, 0.81462],[3.5724, 0.80486],
[3.594, 0.81022],[3.6155, 0.78764],[3.6371, 0.78152],[3.6587, 0.74699],[3.6803, 0.7444],[3.7019, 0.73924],[3.7234, 0.7335],[3.745, 0.72087],[3.7666, 0.74536],[3.7882, 0.70471],[3.8098, 0.70719],[3.8313, 0.69256],[3.8529, 0.685],[3.8745, 0.68787],[3.8961, 0.66807],[3.9176, 0.65611],[3.9392, 0.64243],[3.9608, 0.64033],
[3.9824, 0.62244],[4.004, 0.60245],[4.0255, 0.59614],[4.0471, 0.60321],[4.0687, 0.61488],[4.0903, 0.63889],[4.1118, 0.62694],[4.1334, 0.63277],[4.155, 0.62598],[4.1766, 0.61134],[4.1982, 0.61909],[4.2197, 0.60101],[4.2413, 0.58561],
[4.2629, 0.59843],[4.2845, 0.5661],[4.3061, 0.57375],[4.3276, 0.56447],[4.3492, 0.54285],[4.3708, 0.53329],[4.3924, 0.52831],[4.4139, 0.53501],[4.4355, 0.51645],[4.4571, 0.50344],[4.4787, 0.52583],[4.5003, 0.49168],[4.5218, 0.48297],
[4.5434, 0.48603],[4.565, 0.47169],[4.5866, 0.46824],[4.6082, 0.46642],[4.6297, 0.45131],[4.6513, 0.46068],[4.6729, 0.46384],[4.6945, 0.46432],[4.716, 0.46709],[4.7376, 0.45418],[4.7592, 0.44777],[4.7808, 0.44691],[4.8024, 0.44146],
[4.8239, 0.43381],[4.8455, 0.4252],[4.8671, 0.42166],[4.8887, 0.41955],[4.9102, 0.41974],[4.9318, 0.40205],[4.9534, 0.40549],[4.975, 0.40348],[4.9966, 0.40501],[5.0181, 0.38789],[5.0397, 0.38215],[5.0613, 0.37832],[5.0829, 0.37383],
[5.1045, 0.37306],[5.126, 0.37153],[5.1476, 0.37871],[5.1692, 0.36598],[5.1908, 0.36723],[5.2123, 0.34561],[5.2339, 0.35259],[5.2555, 0.36187],[5.2771, 0.34207],[5.2987, 0.33518],[5.3202, 0.32552],[5.3418, 0.33337],[5.3634, 0.32122],
[5.385, 0.32236],[5.4065, 0.31663],[5.4281, 0.31328],[5.4497, 0.29663],[5.4713, 0.30036],[5.4929, 0.28716],[5.5144, 0.2908],[5.536, 0.28525],[5.5576, 0.28372],[5.5792, 0.27884],[5.6008, 0.27874],[5.6223, 0.26956],[5.6439, 0.26401],
[5.6655, 0.26564],[5.6871, 0.26315],[5.7086, 0.2598],[5.7302, 0.24928],[5.7518, 0.2511],[5.7734, 0.2401],[5.795, 0.2467],[5.8165, 0.24584],[5.8381, 0.2444],[5.8597, 0.238],[5.8813, 0.22747],[5.9029, 0.22958],[5.9244, 0.22891],[5.946, 0.21848],[5.9676, 0.21599],[5.9892, 0.21925],
[6.0107, 0.22164],[6.0323, 0.20595],[6.0539, 0.20949],[6.0755, 0.20212],[6.0971, 0.20078],[6.1186, 0.2026],[6.1402, 0.19581],[6.1618, 0.20212],[6.1834, 0.19093],[6.2049, 0.19208],[6.2265, 0.18089],[6.2481, 0.18462],[6.2697, 0.18797],[6.2913, 0.17467],[6.3128, 0.17324],[6.3344, 0.1762],[6.356, 0.17658],[6.3776, 0.17897],
[6.3992, 0.17716],[6.4207, 0.17381],[6.4423, 0.16549],[6.4639, 0.16797],[6.4855, 0.16003],[6.507, 0.17008],[6.5286, 0.15956],[6.5502, 0.15678],[6.5718, 0.15487],[6.5934, 0.14942],[6.6149, 0.14626],[6.6365, 0.14999],[6.6581, 0.14224],
[6.6797, 0.13631],[6.7013, 0.15171],[6.7228, 0.13277],[6.7444, 0.13029],[6.766, 0.13335],[6.7876, 0.1299],[6.8091, 0.13134],[6.8307, 0.12349],[6.8523, 0.12388],[6.8739, 0.13],[6.8955, 0.11661],[6.917, 0.12789],[6.9386, 0.11861],
[6.9602, 0.11144],[6.9818, 0.11861],[7.0033, 0.11536],[7.0249, 0.11909],[7.0465, 0.10771],[7.0681, 0.1079],[7.0897, 0.10321],[7.1112, 0.10943],[7.1328, 0.10207],[7.1544, 0.10312],[7.176, 0.10522],[7.1976, 0.10446],[7.2191, 0.094318],
[7.2407, 0.092883],[7.2623, 0.092787],[7.2839, 0.096518],[7.3054, 0.084178],[7.327, 0.093361],[7.3486, 0.085996],[7.3702, 0.077578],[7.3918, 0.082074],[7.4133, 0.081596],[7.4349, 0.083604],[7.4565, 0.073752],[7.4781, 0.078917],
[7.4997, 0.07576],[7.5212, 0.081883],[7.5428, 0.075378],[7.5644, 0.071456],[7.586, 0.068299],[7.6075, 0.067056],[7.6291, 0.072126],[7.6507, 0.069734],[7.6723, 0.065812],[7.6939, 0.06476],[7.7154, 0.066099],[7.737, 0.063516],
[7.7586, 0.066577],[7.7802, 0.062847],[7.8017, 0.064856],[7.8233, 0.062082],[7.8449, 0.053664],[7.8665, 0.062368],[7.8881, 0.058829],[7.9096, 0.056629],[7.9312, 0.054333],[7.9528, 0.055768],[7.9744, 0.055385],[7.996, 0.052898],
[8.0175, 0.053472],[8.0391, 0.049359],[8.0607, 0.049455],[8.0823, 0.047446],[8.1038, 0.049742],[8.1254, 0.050316],[8.147, 0.050316],[8.1686, 0.043715],[8.1902, 0.045724],[8.2117, 0.040846],[8.2333, 0.040176],[8.2549, 0.047159],
[8.2765, 0.045437],[8.2981, 0.04075],[8.3196, 0.037498],[8.3412, 0.046298],[8.3628, 0.036254],[8.3844, 0.035967],[8.4059, 0.037115],[8.4275, 0.033384],[8.4491, 0.038359],[8.4707, 0.039219],[8.4923, 0.034628],[8.5138, 0.037785],
[8.5354, 0.034915],[8.557, 0.037306],[8.5786, 0.033863],[8.6001, 0.035202],[8.6217, 0.03568],[8.6433, 0.034628],[8.6649, 0.026593],[8.6865, 0.032523],[8.708, 0.032045],[8.7296, 0.026401],[8.7512, 0.027071],[8.7728, 0.027071],
[8.7944, 0.026019],[8.8159, 0.028506],[8.8375, 0.025062],[8.8591, 0.025732],[8.8807, 0.025732],[8.9022, 0.028601],[8.9238, 0.026306],[8.9454, 0.027071],[8.967, 0.023723],[8.9886, 0.022958],
[9.0101, 0.025062],[9.0317, 0.026019],[9.0533, 0.022288],[9.0749, 0.021523],[9.0964, 0.024297],[9.118, 0.02181],[9.1396, 0.017697],[9.1612, 0.022192],[9.1828, 0.018557],[9.2043, 0.019514],
[9.2259, 0.020566],[9.2475, 0.020758],[9.2691, 0.019992],[9.2907, 0.021714],[9.3122, 0.01961],[9.3338, 0.020662],[9.3554, 0.015783],[9.377, 0.020758],[9.3985, 0.020088],[9.3985, 0.020088],
    ])
    return spectrum