#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from celerite2 import terms, GaussianProcess
from numpy.fft import fftfreq, ifft
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from astropy.timeseries import LombScargle
from astropy import units as u
import emcee
import corner

################################
# Code to simulate time-series #
################################

def simulate_times_regular(T, dt):
    ''' Simulate regularly sampled time-array with duration T and cadence dt
        Inputs:
            T: integer
            dt: integer 
        Outputs: 
            t: array of floats
    '''

    N = int(T / dt) # total no. obs
    t = np.linspace(dt, T, N)
    return t

def get_GP(params, components, t, y_err = None, log = True):
    '''Create GP object with desired kernel and parameters
       Inputs:
           params: array of floats
           components: array of strings
           t: array of floats 
           y_err: array of floats 
           log: True or Flase 
       Outputs:
           gp: object 
    '''
   
    if log:
        params_not_log = np.exp(params)
    else:
        params_not_log = np.copy(params)
    n_comp = len(components)
    i = 0
    for i_comp in range(n_comp):
        S0 = params_not_log[i]
        w0 = params_not_log[i+1]
        if components[i_comp] == 'SHO_periodic' or components[i_comp] == 'SHO_aperiodic':
            if components[i_comp] == 'SHO_periodic':
                Q = params_not_log[i+2]
                i += 3
            elif components[i_comp] == 'SHO_aperiodic':
                Q = 1.0 / np.sqrt(2)
                i += 2
            kernel_comp = terms.SHOTerm(S0 = S0,
                                    w0 = w0,
                                    Q = Q)
        elif components[i_comp] == 'Rotation':
            sigma = params_not_log[i]
            period = params_not_log[i+1]
            Q = params_not_log[i+2]
            dQ = params_not_log[i+3]
            f = params_not_log[i+4]
            i += 5
            kernel_comp = terms.RotationTerm(sigma = sigma,
                                    period = period,
                                    Q0 = Q, 
                                    dQ = dQ, 
                                    f = f)
            
        else:
            stop('Error: not set up for this type of component yet: ',
                  components[i_comp])
            

        if i_comp == 0: 
            kernel = kernel_comp
        else:
            kernel += kernel_comp
    gp = GaussianProcess(kernel)
    # white noise
    if len(params) == i:
        # no jitter term in parameter array, just [optional] known uncertainties
        gp.compute(t, yerr = y_err)
    else:
        # parameter array includes jitter term 
        wnV = params_not_log[-1]**2
        if y_err is not None:
            wnV += y_err**2
        gp.compute(t, diag = wnV)
    return gp

def get_labels(components, log = True, wn = True):
    n_comp = len(components)
    if log:
        prefix = '\ln '
    else:
        prefix = ''
    labels = []
    for i_comp in range(n_comp):
        if components[i_comp] == 'SHO_periodic' or components[i_comp] ==  'SHO_aperiodic':
            labels.append('${:s}S_{:d}$'.format(prefix,i_comp))
            labels.append('${:s}\\nu_{:d}$'.format(prefix,i_comp))
        if components[i_comp] == 'SHO_periodic':
            labels.append('${:s}Q_{:d}$'.format(prefix,i_comp))
        elif components[i_comp] ==  'Rotation':
            labels.append('${:s}\sigma_{:d}$'.format(prefix,i_comp))
            labels.append('${:s}P_{:d}$'.format(prefix,i_comp))
            labels.append('${:s}Q_{:d}$'.format(prefix,i_comp))
            labels.append('${:s}dQ_{:d}$'.format(prefix,i_comp))
            labels.append('${:s}f_{:d}$'.format(prefix,i_comp))
    if wn:
        labels.append('${:s}\sigma_n$'.format(prefix))
    return labels

def simulate_GP(t, components, params, nsim, log = False, y_err = None):
    # Simulate nsim samples from GP with specified components and
    # parameters over time array t
    print(y_err)
    gp = get_GP(params, components, t, y_err = y_err, log = log)
    samples = gp.sample(size = nsim).T
    return samples

def resample(t_in, y_in, mode = 'reg', dur_frac = 1, 
             exposure = 1, readout = 0,
             n_desired = 10):

    # mode: one of 'reg' or 'random' (default: reg)
    # dur_frac: float between 0 and 1, used to define the time range 
    #     within which the samples are selected. The start of this window is 
    #     selected at random, so that the end of the window is before the end 
    #     of the input time array. (default: 1)
    # exposure: integer >= 1, duration of each "exposure" in number of time-steps 
    #     in the input array (default: 1)
    # readout: integer >= 0, duration of any gap after each exposure before the 
    #     next one starts (same units as exposure) (default: 0)
    # n_desired: integer > 0. number of elements in output time-array when mode='random'
    #     Must be < n_in / (exposure + readout). Ignored when mode = 'reg'
    
    
    # First, check if we were given one time-series as input or several
    S = y_in.shape
    if len(S) == 1:
        nsim = 1
        flat = True
    else:
        flat = False
        nsim = S[1]
        
    # Trim overall duration
    tmin_in, tmax_in = min(t_in), max(t_in)
    trange_in = tmax_in - tmin_in
    trange_out = dur_frac * trange_in
    rng = np.random.default_rng()
    tmin_out = rng.uniform() * (trange_in - trange_out)
    tmax_out = tmin_out + trange_out
    l = (t_in >= tmin_out) * (t_in <= tmax_out)
    t_trim = t_in[l]
    if flat:
        y_trim = y_in[l].reshape((l.sum(),1))
    else:
        y_trim = y_in[l,:]
    n_trim = len(t_trim)
    
    # Select exposure start times
    cadence = exposure + readout
    n_bin = int(np.floor(n_trim / cadence))
    if mode == 'reg':
        bin_start_indices = np.array(range(n_bin)) * cadence
        n_selected = len(bin_start_indices)
    elif mode == 'random':
        if n_desired > n_bin:
            stop('Error: you asked for more observations than available')
        if readout > 0:
            print("Warning: random mode ignores readout for the time being")
        print("Warning: exposures may overlap in random mode")
        bin_start_indices = np.sort(rng.choice(n_trim, n_desired, replace = False))
    n_selected = len(bin_start_indices)

    # now bin
    t_out = t_trim[bin_start_indices] # note the output times correspond to the start of the exposure
    y_out = np.zeros((n_selected, nsim))
    for i in range(n_selected):
        ii = bin_start_indices[i]
        y_out[i,:] = y_trim[ii:ii+exposure,:].mean(0)
    if flat:
        y_out = y_out.flatten()
    return t_out, y_out

def bin_and_trim(t, y, bin_fac, dur_frac = 1):
    # Bin and trim one or more samples
    S = y.shape
    if len(S) == 1:
        nsim = 1
    else:
        nsim = S[1]
    # first, trim
    tmax = max(t)
    tmax_new = max(t) * dur_frac
    l = t <= tmax_new
    t_new = t[l]
    if nsim == 1:
        y_new = y[l]
    else:
        y_new = y[l,:]
    # then bin
    n_in = len(t_new)
    n_out = int(np.floor(n_in / bin_fac))
    n_in_adjusted = n_out * bin_fac
    t_nnew = t_new[:n_in_adjusted].reshape(n_out, bin_fac).mean(1)
    if nsim == 1:
        y_nnew = y_new[:n_in_adjusted].reshape(n_out, bin_fac).mean(1)
    else:
        y_nnew = y_new[:n_in_adjusted,:].reshape(n_out, bin_fac,nsim).mean(1)
    return t_nnew, y_nnew

def bin_and_trim_error(t, y, err, bin_fac, dur_frac = 1):
    # Bin and trim one or more samples
    S = y.shape
    if len(S) == 1:
        nsim = 1
    else:
        nsim = S[1]
    # first, trim
    tmax = max(t)
    tmax_new = max(t) * dur_frac
    l = t <= tmax_new
    t_new = t[l]
    if nsim == 1:
        y_new = y[l]
        e_new = err[l]
    else:
        y_new = y[l,:]
        e_new = err[l,:]
    # then bin
    n_in = len(t_new)
    n_out = int(np.floor(n_in / bin_fac))
    n_in_adjusted = n_out * bin_fac
    t_nnew = t_new[:n_in_adjusted].reshape(n_out, bin_fac).mean(1)
    if nsim == 1:
        y_nnew = y_new[:n_in_adjusted].reshape(n_out, bin_fac).mean(1)
        e_nnew = e_new[:n_in_adjusted].reshape(n_out, bin_fac).mean(1)
    else:
        y_nnew = y_new[:n_in_adjusted,:].reshape(n_out, bin_fac,nsim).mean(1)
        e_nnew = e_new[:n_in_adjusted,:].reshape(n_out, bin_fac,nsim).mean(1)
    return t_nnew, y_nnew, e_nnew

def add_wn(samples, sig):
    rng = np.random.default_rng()
    noise = rng.normal(0, 1, samples.shape) * sig
    return samples + noise

#######################################
# Code to evaluate PSD using FFT      #
#######################################

def get_FFT_PSD(t, y):
    # Evaluate PSD of time-series using FFT
    dt = t[1] - t[0]
    freq = fftfreq(len(t), dt)
    freq_step = freq[1] - freq[0]
    ft_amp = np.absolute(ifft(y))
    ft_power = ft_amp**2
    psd = ft_power / freq_step
    return freq, psd

def bin_PSD_logfreq(freq, psd, n_bins = 50):
                    # , bin_width = 0.1):
    lpos = freq > 0
    pfreq = freq[lpos]
    lfreq = np.log10(pfreq)
    ppsd = psd[lpos]
    lfmin, lfmax = lfreq.min(), lfreq.max()
    bins = np.linspace(lfmin, lfmax, n_bins)
    bin_width = bins[1] - bins[0]
    if max(bins) <= lfmax:
        bins = np.append(bins,max(bins)+bin_width)
    indices = np.digitize(lfreq, bins)
    binned_freq = np.array([])
    binned_psd = np.array([])
    for i in range(len(bins)-1):
        l = indices == i
        if l.sum() < 3:
            if l.sum() > 0:
                binned_freq = np.concatenate([binned_freq, pfreq[l]])
                binned_psd = np.concatenate([binned_psd, ppsd[l]])
            else: 
                continue
        else:
            binned_freq = np.append(binned_freq, 10**(np.mean(lfreq[l])))
            binned_psd = np.append(binned_psd, np.mean(ppsd[l]))
    return binned_freq, binned_psd

def convert_psd_normal_to_cel(psd_in):
    # multiply by sqrt(2pi) to account for asymmetric FT definition
    # divide by 2pi to convert from freq to omega
    return psd_in / np.sqrt(2 * np.pi)

def convert_psd_cel_to_normal(psd_in):
    # multiply by 2pi to convert from omega to freq
    # divide by sqrt(2pi) to account for asymmetric FT definition
    return psd_in * np.sqrt(2 * np.pi)

#######################################
# Code to evaluate PSD using LS      #
#######################################

def get_WF_PS(t, freq):
    arg = 2 * np.pi * freq[:,None] * t[None, :]
    WF_pow = (np.cos(arg).sum(axis=1))**2 + (np.sin(arg).sum(axis=1))**2
    N = len(t)
    WF_pow = WF_pow / N**3 #/ 4
    return WF_pow

def get_LS_PSD(t, y, yerr = None):

    # Evaluate PSD of time-series using LS
    
    # Note that, as described in the astropy documentation, the LS periodogram 
    # with "psd" normalisation is only equivalent to the FFT power if we don't 
    # pass it any uncertainties. However, here we want to be able to give 
    # different relative weights to the different observations, if need be (for 
    # example if analysing a time series with variying signal-to-noise due to
    # weather etc...). Therefore, we re-normalise the uncertainties so that
    # their mean is unity.

    freq_max = 1/(max(t) - min(t))
    freq_min = 1/np.min(np.diff(t))
    window = np.ones(len(t))*np.sqrt(2)/2
    
    if yerr is not None:
        yerr_mean = np.mean(yerr)
        yerr_renorm = yerr / yerr_mean 
        ls = LombScargle(t, y, yerr_renorm)
    else:
        ls = LombScargle(t, y)
       
    # To evaluate the power spectrum, use the normalisation="psd" option.
    
    freq, power = ls.autopower(method="fast", normalization="psd")
    window_function = windowFT(freq.min(),freq.max(), len(freq),t,window)
    #power = power  * window_function[1]
    # TBD: modify the max frequency following section 4.1 of VanderPlas 2018
    
    # We also need to multiply the output by the number of data points N
    # in the input to make it comparable to the FFT-estimated power 
    # (cf "PSD normalisation" tutorial in docs for celerite v1, or 
    # astropy LombScargle docs under periodogram normalisation)
    power /= len(t)
    
    # To convert this to a PSD, we need to divide by the lowest frequency 
    # we're sensitive to, i.e. the inverse of the time span of the data.
    # NB: this typically differs from the minimum frequency returned by 
    # LS.autopower(), so we evaluate it directly
    T = max(t) - min(t)
    psd = power * T
    
    return freq, psd, window_function

#######################################
# Code to fit GP model to time-series #
# or analytic model to PSD estimate   #
#######################################

def get_log_prior(params_log, components, nyquist):
    n_comp = len(components)
    i = 0
    for i_comp in range(n_comp):
        if components[i_comp] == 'SHO_periodic' or components[i_comp] ==  'SHO_aperiodic':
            log_S0 = params_log[i]
            log_w0 = params_log[i+1]
            if components[i_comp] == 'SHO_periodic':
                log_Q = params_log[i+2]
                i += 3
            else:
                log_Q = np.log(1.0 / np.sqrt(2))
                i += 2
            par_comp_log = np.array([log_S0, log_w0, log_Q])
            if log_w0 > np.log(nyquist):
                #print( np.log(nyquist))
                #print(params_log)
                #print('nyquist')
                return -np.inf
            if i_comp > 0:
                if components[i_comp - 1 ] == 'SHO_periodic' or 'SHO_aperiodic':
                    if log_w0 > log_w0_old:
                        #print('wrong order')
                        return -np.inf
            log_w0_old = log_w0 + 0.0
        elif components[i_comp] ==  'Rotation':
            log_sigma = params_log[i]
            log_period = params_log[i+1]
            log_Q = params_log[i+2]
            log_dQ = params_log[i+3]
            log_f = params_log[i+4]
            if log_f > 0:
                return -np.inf
            if np.isnan(log_period) == True :
                return -np.inf
            i += 5
            
            par_comp_log = np.array([log_sigma, log_period, log_Q, log_dQ, log_f])
        
    if (abs(params_log) > 11).any():
        #print('over 10')
        return -np.inf
    
    # white noise term
    if len(params_log) == i:
        return 0
    elif abs(params_log[-1]) > 5:
        return -np.inf
    return 0

def analytic_psd(params_log, components, freq, freq_step = None):
    # NB: It's important to provide native freq_step if the PSD has
    # been re-binned before fitting!
    omega = 2 * np.pi * freq
    n_comp = len(components)
    i = 0
    for i_comp in range(n_comp):
        if components[i_comp] == 'SHO_periodic' or components[i_comp] ==  'SHO_aperiodic':
            S0 = np.exp(params_log[i])
            w0 = np.exp(params_log[i+1])
            if components[i_comp] == 'SHO_periodic':
                Q = np.exp(params_log[i+2])
                i += 3
            else:
                Q = 1.0 / np.sqrt(2)
                i += 2        
            kernel_comp = terms.SHOTerm(S0 = S0,
                                        w0 = w0,
                                        Q = Q)
        elif components[i_comp] ==  'Rotation':
            sigma = np.exp(params_log[i])
            period = np.exp(params_log[i+1])
            Q = np.exp(params_log[i+2])
            dQ = np.exp(params_log[i+3])
            f = np.exp(params_log[i+4])
            i += 5
            kernel_comp = terms.RotationTerm(sigma = sigma,
                                    period = period,
                                    Q0 = Q, 
                                    dQ = dQ, 
                                    f = f)

        psd_comp_cel = kernel_comp.get_psd(omega)
        psd_comp = convert_psd_cel_to_normal(psd_comp_cel)
        
        if i_comp == 0: 
            psd = psd_comp
        else:
            psd += psd_comp
    
    if len(params_log) > i:
        wn = np.exp(params_log[-1])
        if freq_step is None:
            freq_step = freq[1] - freq[0]
        wn_psd = wn * freq_step / 2
        psd += wn_psd
    return psd

def log_prob(params_log, components, x, y, use_GP = True, y_err = None):
    if use_GP: # x = time, y = observable, y_err = measurement uncertainties
        dt_min = (x[1:] - x[:-1]).min()
        nyquist = 1/(2*dt_min)
    else: # input x = freq, y = psd, y_err = freq_step
        nyquist = max(x)
        freq_step = y_err 
    lp = get_log_prior(params_log, components, nyquist)
    if np.isfinite(lp):
        if use_GP:
            gp = get_GP(params_log, components, x, y_err)
            try:
                ll = gp.log_likelihood(y)
                return ll + lp
            except:
                return -np.inf
        else:
            model_psd = analytic_psd(params_log, components, x, freq_step)
            resid = np.log(y) - np.log(model_psd) # fit in log space
            ll = -0.5 * (resid**2).sum()
            return ll + lp                
    else:
        return lp

def neg_log_prob(params_log, components, x, y, use_GP = True, y_err = None):
    nll = -log_prob(params_log, components, x, y,
                    use_GP = use_GP, y_err = y_err)
    #print(params_log)
    #print(nll)
    if np.isfinite(nll):
        return nll
    else:
        return 1e25

def fit_time_series(t, y, components, y_err = None,
                    par_initial = None, par_true = None, 
                    do_plot = True, use_GP = True, regular = True):
    # Fit time-series in time or frequency domain

    # Set up initial guess for parameters
    labels = get_labels(components)
    if par_initial is None:
        if par_true is None:
            print('Error: need to supply at least one of par_initial or par_true')
        else:        
            initial_params_log = np.log(par_true) + \
                np.random.uniform(-0.2,0.2, len(labels))
    else:
        initial_params_log = np.log(par_initial)


    # Evaluate PSD of time-series using FFT
    freq_step = 1/(t.max() - t.min())
    if regular:
        freq, psd = get_FFT_PSD(t, y)

    elif y_err is not None:
        freq, psd, wf = get_LS_PSD(t, y, y_err)
    else:
        freq, psd, wf = get_LS_PSD(t, y)
  
    if components == ['SHO_aperiodic'] or components == ['SHO_aperiodic', 'SHO_aperiodic']:
        binned_freq, binned_psd = \
            bin_PSD_logfreq(freq, psd, n_bins = 50)
    else:
        binned_freq, binned_psd = \
            bin_PSD_logfreq(freq, psd, n_bins = 200)
    
    freq_step_binned = binned_freq[-1] - binned_freq[-2]
    
        
    if use_GP:
        args_ = (components, t, y, use_GP, y_err)
    else:
        args_ = (components, binned_freq, binned_psd, use_GP, freq_step)
 
 
    
    # Find maximum a posteriori solution first
    r = minimize(neg_log_prob, initial_params_log,
                 args = args_ , method = 'L-BFGS-B') #, method = "L-BFGS-B" Nelder-Mead')
 
    
    # Do MCMC

    coords = r.x + 1e-5 * np.random.randn(4 * len(labels), len(r.x))
    
    sampler = emcee.EnsembleSampler(coords.shape[0], coords.shape[1],
                                     log_prob, 
                                     args = args_)
    state = sampler.run_mcmc(coords, 10000, progress = False)
    try:
        tau = sampler.get_autocorr_time()
        discard = 3 * np.max(tau)
        discard = int(np.trunc(discard))
        thin = 0.25 * np.max(tau)
        thin =  int(np.trunc(thin))
        ran = True
    except:
        discard = 5000
        thin = 30
        print('Skipping error - check chains')
        do_plot = True
        ran = False
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    flat_log_probs = sampler.get_log_prob(discard=discard, thin=thin, flat=True)
    
    i_best = np.argmax(flat_log_probs)
    best_fit_par = flat_samples[i_best]
    best_fit_log_prob = flat_log_probs[i_best]
    n_par = len(best_fit_par)
    BIC = n_par * np.log(len(y)) - 2 * best_fit_log_prob
 
    n_comp = len(components)
    plt.tight_layout()
    flat_samples_nu = []
    par_true_nu = []
    i = 0
            
    for i_comp in range(n_comp):
        if components[i_comp] == 'SHO_periodic' or components[i_comp] ==  'SHO_aperiodic':
            S0_samples = flat_samples[:,i]
            w0_samples = flat_samples[:,i + 1]
            nu_samples = np.log(np.exp(w0_samples)/(2 * np.pi))
            flat_samples_nu.append(S0_samples)
            flat_samples_nu.append(nu_samples)
            if par_true is not None:
                S0_true = par_true[i]
                nu_true = par_true[i + 1]/(2*np.pi)
                par_true_nu.append(S0_true)
                par_true_nu.append(nu_true)
            if components[i_comp] == 'SHO_periodic':
                Q_samples = flat_samples[:, i + 2]
                flat_samples_nu.append(Q_samples)
                if par_true is not None:
                    Q_true = par_true[i + 3]
                    par_true_nu.append(Q_true)
                i += 3
                    
            else:
                 i += 2
        elif components[i_comp] == 'Rotation':
            sigma_samples = flat_samples[:,i]
            period_samples = flat_samples[:,i + 1]
            Q0_samples = flat_samples[:,i + 2]
            dQ_samples = flat_samples[:,i + 3]
            f_samples = flat_samples[:,i + 4]
            flat_samples_nu.append(sigma_samples)
            flat_samples_nu.append(period_samples)
            flat_samples_nu.append(Q0_samples)
            flat_samples_nu.append(dQ_samples)
            flat_samples_nu.append(f_samples)
            
            if par_true is not None:
                par_true_nu.append(par_true[i])
                par_true_nu.append(par_true[i + 1])
                par_true_nu.append(par_true[i + 2])
                par_true_nu.append(par_true[i + 3])
                par_true_nu.append(par_true[i + 4])
                
            i += 5
    if len(initial_params_log) > i:         
        wn_samples = flat_samples[:, -1]
    if par_true is not None:
        wn_true = par_true[-1]
        par_true_nu.append(wn_true)
            
        
    par_true_nu = np.array(par_true_nu)
    if len(initial_params_log) > i:  
             flat_samples_nu.append(wn_samples)
    flat_samples_nu = np.array(flat_samples_nu)
    flat_samples_nu = np.transpose(flat_samples_nu)
    lpos = freq > 0
    
    mcmc = np.percentile(flat_samples, [5, 50,  95], axis = 0)
    psd_5 = analytic_psd(mcmc[0], components, freq, freq_step)
    psd_50 = analytic_psd(mcmc[1], components, freq, freq_step)
    psd_95 = analytic_psd(mcmc[2], components, freq, freq_step)
    
        
                     
    if do_plot:
        
         # Plot chains
        samples = sampler.get_chain(discard=discard, thin=thin)
        nsteps, nwalkers, ndim = samples.shape
        fig, axes = plt.subplots(len(labels) + 1, figsize=(6, 8), sharex=True)
        
        for i in range(len(labels)):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            
            if par_true is not None:
                ax.axhline(np.log(par_true[i]), color = 'C0')
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        
        log_prob1 = sampler.get_log_prob(discard=discard, thin=thin)
        axes[-1].plot(log_prob1, "k", alpha=0.3)
        axes[-1].set_ylabel(r'$\log P$')
        axes[-1].set_xlabel("step number")
        plt.tight_layout()

         # Plot time-series and predictive distribution for MAP solution
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        if y_err is None:
             ax2.plot(t , y, 'k.', ms = 3, alpha = 0.5)
        else:
             ax2.errorbar(t, y, yerr = y_err,
                          fmt = 'k.', ms = 3, alpha = 0.5)
        gp = get_GP(r.x, components, t, y_err)
        mu, var = gp.predict(y, return_var = True)
        sig = np.sqrt(var)
        plt.plot(t, mu, 'C2-')        
        plt.fill_between(t, mu + 2 * sig, mu - 2 * sig,
                          color = 'C2', alpha = 0.3, lw = 0)
        ax2.set_xlabel("time")
        ax2.set_ylabel("observable")
        plt.tight_layout()
        
         # Plot PSD 
        fig1, ax1 = plt.subplots(figsize=(6, 5))        
        ax1.loglog(freq[lpos], psd[lpos], 'k-', lw = 1, alpha = 0.5,  label = 'Simulated PSD')
        ax1.set_xlim(np.min(freq[lpos]), np.max(freq[lpos]))
        ax1.plot(binned_freq, binned_psd, 'k.', ms = 4)
         # plot truth
        if (par_true is not None):
            psd_true = analytic_psd(np.log(par_true), components, freq, freq_step)
            ax1.loglog(freq[lpos], psd_true[lpos], 'C0-', label = 'True PSD')
         # plot MAP estimate
        psd_map = analytic_psd(r.x, components, freq, freq_step)
        ax1.loglog(freq[lpos], psd_map[lpos], 'C3-', lw = 0.5, label = 'map estimate')        
         # plot posterior samples
        
        mcmc = np.percentile(flat_samples, [5, 50,  95], axis = 0)
        psd_5 = analytic_psd(mcmc[0], components, freq, freq_step)
        psd_50 = analytic_psd(mcmc[1], components, freq, freq_step)
        psd_95 = analytic_psd(mcmc[2], components, freq, freq_step)
     
        if use_GP:
            ax1.loglog(freq[lpos], psd_5[lpos], 'C6-', lw = 0.5)
            ax1.loglog(freq[lpos], psd_50[lpos], 'C6-', lw = 0.5)
            ax1.loglog(freq[lpos], psd_95[lpos], 'C6-', lw = 0.5)
            ax1.fill_between(freq[lpos], psd_5[lpos], psd_95[lpos], color = 'C6', alpha = 0.3, lw = 0)
 
        else:
            ax1.loglog(freq[lpos], psd_5[lpos], 'C2-', lw = 0.5)
            ax1.loglog(freq[lpos], psd_50[lpos], 'C2-', lw = 0.5)
            ax1.loglog(freq[lpos], psd_95[lpos], 'C2-', lw = 0.5)
            ax1.fill_between(freq[lpos], psd_5[lpos], psd_95[lpos], color = 'C2', alpha = 0.3, lw = 0)

        ax1.set_xlabel("frequency")
        ax1.set_ylabel("PSD")
        ax1.legend()
  
        
        if par_true is None:
            fig = corner.corner( flat_samples_nu, labels=labels, color='C1', show_titles = True)
        else:
            fig = corner.corner( flat_samples_nu, labels=labels, color='C1', truths = np.log(par_true_nu), truth_color='C0', show_titles = True)
    return flat_samples, freq[lpos], psd[lpos], binned_freq, binned_psd,  lpos, freq_step, flat_samples, BIC, flat_samples_nu, best_fit_par,  ran
    


# TO DO NEXT

# ALLOW FOR IRREGULAR TIME SAMPLING (USING LSPER)

if __name__ == '__main__':
    T = 100 
    dt = 0.01
    
    components = ['SHO_aperiodic']
    par_true = [2, 5]

    nsim = 2

    bin_fac = 2
    dur_frac = 0.5

    wn = 0.5

    t = simulate_times_regular(T, dt)
    samples = simulate_GP(t, components, par_true, nsim)

    t_bt, samples_bt = bin_and_trim(t, samples, bin_fac, dur_frac)
    samples_noisy = add_wn(samples_bt, wn)

    par_true = np.append(par_true, wn)

    # Fit with GP. Note we're not giving it the measurement
    # uncertainties because we're fitting for them.
    posterior_samples_GP = fit_time_series(t_bt, samples_noisy[:,0], components,
                                           y_err = None,
                                           use_GP = True,
                                           par_true = par_true,
                                           do_plot = True, 
                                          regular = False)

    # Now fit analytic function to FFT-estimated PSD. Note in this
    # case we can't pass individual measurement uncertainties anyway.
    posterior_samples_FFT = fit_time_series(t_bt, samples_noisy[:,0], components,
                                            y_err = None,
                                            use_GP = False,
                                            par_true = par_true,
                                            do_plot = True, 
                                           regular = False)

    # Compare the results of the two method to the true parameters
    labels = get_labels(components, log = True)    
    fig = corner.corner(posterior_samples_GP, labels = labels,
                        color = 'C0',
                        truths = np.log(par_true),
                        truth_color='k', show_titles = False)
    corner.corner(posterior_samples_FFT, color = 'C1',
                  fig = fig)
    plt.show()
    



