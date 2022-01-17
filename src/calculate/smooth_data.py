import numpy as np
from scipy import stats
from scipy.signal import butter, filtfilt, wiener
from scipy.ndimage import median_filter

import dtcwt
from timeit import default_timer as timer


def dual_tree_complex_wavelet_filter(signal, filt_signal=[], filter=None, filt_opts={}, 
               max_iter=150, baseline_opts={},
               lvls=4, biort='near_sym_b', qshift='qshift_d', noise_type='+-'):

    transform = dtcwt.Transform1d(biort=biort, qshift=qshift)
    gain_mask = np.zeros(lvls)
    gain_mask[-1] = 1
    
    signal_mod = np.array(signal, copy=True)
    # signal_mod = signal # overwrites signal
    
    isfiltered = True
    if len(filt_signal) > 0:    # if filtered signal is provided, pass
        pass
    elif filter == 'wavelet':     # Wavelet filter
        filt_lvls = 11
        if 'nlevels' in filt_opts:
            filt_lvls = filt_opts['nlevels']
        
        filt_mask = np.zeros(filt_lvls)
        filt_mask[-1] = 1
            
        signal_t = transform.forward(signal_mod, nlevels=filt_lvls)
        filt_signal = transform.inverse(signal_t, gain_mask=filt_mask)
    
    elif filter == 'median':   # Median Filter
        window_perc = 0.10
        if 'window_perc' in filt_opts:
            window_perc = filt_opts['window_perc']      

        filt_window = int(np.floor(len(signal_mod)*window_perc))
        filt_signal = median_filter(signal_mod, size=filt_window)
    
    elif filter == 'wiener':    # Wiener Filter
        window_perc = 0.05
        if 'window_perc' in filt_opts:
            window_perc = filt_opts['window_perc']      

        filt_window = int(np.floor(len(signal_mod)*window_perc))
        
        filt_signal = wiener(signal_mod, mysize=filt_window, noise=4000)    # noise-power should be changed if used
    
    elif filter == 'butter':    # Butterworth Filter
        fc = 30                 # Cut-off frequency of the filter
        N = 5                   # Order of the filter
        if 'fc' in filt_opts:   
            fc = filt_opts['fc']
        
        if 'N' in filt_opts:    
            N = filt_opts['N']

        w = fc/(len(signal_mod)/2)  # Normalize the frequency
        b, a = butter(N, w, 'low')      # Create Filter
        filt_signal = filtfilt(b, a, signal_mod)
    else:
        isfiltered = False
    
    if noise_type == '+-':
        df = np.shape(signal_mod)[0] - 1
        t_crit = 0
        alpha = 0.05        # 95% confidence interval
        if 'alpha' in baseline_opts:
            alpha = baseline_opts['alpha']
        if df != 0:
            t_crit = stats.t.ppf(1-alpha/2, df=df)        
    
    wavelet_coef = []
    for n in range(max_iter):
        if n != 0 or not isfiltered:
            signal_t = transform.forward(signal_mod, nlevels=lvls)
            wavelet_coef.append(signal_t.highpasses[-1])
            filt_signal = transform.inverse(signal_t, gain_mask=gain_mask)
        
        # if np.mod(n, 25) == 0:
            # baseline.append(filt_signal)
        
        # Break condition
        if len(wavelet_coef) > 2:   # clear first if len > 2 to keep rolling list
            del wavelet_coef[0]
        if len(wavelet_coef) == 2:
            rel_err = np.abs(np.subtract(wavelet_coef[1], wavelet_coef[0]))/np.abs(wavelet_coef[0])*100
            if np.max(rel_err) < 0.5:   # if change is less than 0.5% then break
                break
                
        if noise_type == '+-':  # outlier rejection
            outlier_detection = 'percentile'
            if 'outlier_detection' in baseline_opts:
                outlier_detection = baseline_opts['outlier_detection']
            
            if 'residuals' in baseline_opts:
                if baseline_opts['residuals'] == 'signal':
                    res = signal - filt_signal
                elif baseline_opts['residuals'] == 'signal_mod':
                    res = signal_mod - filt_signal
            else:
                res = signal_mod - filt_signal
            
            if outlier_detection == 'conf_int':
                mean = np.mean(res)
                std = np.std(res)
                # SE = std/np.sqrt(df+1)
                limits = np.sort([mean-std*t_crit, mean+std*t_crit])    # limits is conf interval
                
            elif outlier_detection == 'MADe':
                median = np.median(res)
                MADe = np.median(np.abs(res - median))*1.483
                limits = np.sort([median-MADe, median+MADe])
                
            elif outlier_detection == 'adjusted_boxplot':    # (G. Bray et al. (2005))
                median = np.median(res)
                q1, q3 = np.percentile(res, 25), np.percentile(res, 75)
                iqr = q3 - q1 
                
                h = []
                for i in range(len(res)):
                    for j in range(len(res)):
                        if res[i] != res[j] and res[i] <= median and median <= res[j]:
                            h = ((res[j] - median) - (median - res[i]))/(res[j] - res[i])
                MC = np.median(h)
                if MC < 0:
                    limits = np.sort([q1-1.5*np.exp(-4*MC)*iqr, q3+1.5*np.exp(3.5*MC)*iqr])
                else:
                    limits = np.sort([q1-1.5*np.exp(-3.5*MC)*iqr, q3+1.5*np.exp(4*MC)*iqr])
                    
            elif outlier_detection == 'percentile':
                percentile = 25
                if n == 0:
                    percentile = 35
                q1, q3 = np.percentile(res, percentile), np.percentile(res, 100-percentile)
                iqr = q3 - q1         
                cut_off = iqr*1.5
                limits = np.sort([q1-cut_off, q3+cut_off])
                
            mask = np.logical_or(np.less(res, limits[0]), np.greater(res, limits[1]))
                
        elif noise_type == '+':
            mask = np.greater(signal_mod, filt_signal)  # how it's done in Galloway paper
        elif noise_type == '-':
            mask = np.less(signal_mod, filt_signal)
        
        signal_mod[mask] = filt_signal[mask]

    return filt_signal