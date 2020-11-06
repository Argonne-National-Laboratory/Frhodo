# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import re
from tabulate import tabulate

import matplotlib as mpl
import numpy as np
from scipy import stats

from plot.base_plot import Base_Plot
from plot.draggable import Draggable


class Plot(Base_Plot):
    def __init__(self, parent, widget, mpl_layout):
        super().__init__(parent, widget, mpl_layout)
        
        self.start_ind = 300
        self.end_ind = 800
    
    def info_table_text(self, preshock, postshock, prec=2):
        def fix_g_format(value, prec):
            text = '{:.{dec}g}'.format(value, dec=prec)
            text = text.replace("e+", "e")
            return re.sub("e(-?)0*(\d+)", r"e\1\2", text)
        
        shock_zone = 2
        # table = [['pre-shock',  fix_g_format(preshock, prec)],
                 # ['post-shock', fix_g_format(postshock, prec)],
                 # ['difference', fix_g_format(postshock-preshock, prec)]]
        
        table = [['pre-shock',  '{:.{dec}f}'.format(preshock, dec=prec)],
                 ['post-shock', '{:.{dec}f}'.format(postshock, dec=prec)],
                 ['difference', '{:.{dec}f}'.format(postshock-preshock, dec=prec)]]
        
        table = tabulate(table).split('\n')[1:-1] # removes header and footer
        table.insert(0, 'Baseline Averages')
        
        table_left_justified = []
        max_len = len(max(table, key=len))
        for line in table:
            table_left_justified.append('{:<{max_len}}'.format(line, max_len=max_len))
            
        return '\n'.join(table_left_justified)
    
    def create_canvas(self):
        self.ax = []
        
        self.ax.append(self.fig.add_subplot(1,1,1))
        self.ax[0].item = {}

        self.ax[0].item['exp_data'] = self.ax[0].scatter([],[], color='#0C94FC', facecolors='#0C94FC',
            s=16, linewidth=0.5, alpha = 0.85)
        self.ax[0].item['start_divider'] = self.ax[0].add_line(mpl.lines.Line2D([],[], 
            marker='$'+'\u2336'+'$', markersize=18, markerfacecolor='0', markeredgecolor='0', markeredgewidth=0.5, 
            zorder=2))
        self.ax[0].item['start_avg_l'] = self.ax[0].add_line(mpl.lines.Line2D([],[], ls='--', c= '0'))
        self.ax[0].item['end_divider'] = self.ax[0].add_line(mpl.lines.Line2D([],[], 
            marker='$'+'\u2336'+'$', markersize=18, markerfacecolor='0', markeredgecolor='0', markeredgewidth=0.5, 
            zorder=2))
        self.ax[0].item['end_avg_l'] = self.ax[0].add_line(mpl.lines.Line2D([],[], ls='--', c= '0'))
                
        self.ax[0].item['textbox'] = self.ax[0].text(.98,.98, self.info_table_text(0, 0), fontsize=10, fontname='DejaVu Sans Mono',
            horizontalalignment='right', verticalalignment='top', transform=self.ax[0].transAxes)
        
        self.fig.subplots_adjust(left=0.06, bottom=0.065, right=0.98,
                        top=0.97, hspace=0, wspace=0.12)
        
        # Create canvas from Base
        super().create_canvas()
        
        # Add draggable lines
        draggable_items = [[0, 'start_divider'], [0, 'end_divider']]
        for pair in draggable_items:
            n, name = pair  # n is the axis number, name is the item key
            update_fcn = lambda x, y, name=name: self.draggable_update_fcn(name, x, y)
            self.ax[n].item[name].draggable = Draggable(self, self.ax[n].item[name], update_fcn)
      
    def draggable_update_fcn(self, name, x, y):
        if self.parent.display_shock['raw_data'].size == 0: return
        x0, xpress, xnew, xpressnew = x['0'], x['press'], x['new'], x['press_new']
        y0, ypress, ynew, ypressnew = y['0'], y['press'], y['new'], y['press_new']
        
        if name is 'start_divider':
            self.start_ind = np.argmin(np.abs(self.t - xnew))
            if self.start_ind < 3:
                self.start_ind = 3
            
        elif name is 'end_divider':
            self.end_ind = np.argmin(np.abs(self.t - xnew))

        self.update(estimate_ind=False)
    
    def update(self, estimate_ind=True, update_lim=False):
        def shape_data(x,y): return np.transpose(np.vstack((x,y)))
        def set_xy(plot, x, y): 
            plot.set_xdata(x)
            plot.set_ydata(y)
        
        def estimateInd(data, frac, alpha):
            def pred_int(i_old, i):
                SD = np.std(data[:i_old])    # Standard deviation of sample
                sigma = SD**2                   # Variance of Sample
                return t*np.sqrt((sigma/i_old + sigma/(i-i_old)))   # Prediction interval for 2 means
            
            def calc_mu_t(i):
                mu = np.mean(data[:i])   # Mean of sample
                df = i - 1
                t = stats.t.ppf(1-alpha/2, df=df)
                return mu, t
            
            i_old = int(np.round(np.shape(data)[0]*frac))
            i = i_old + 1
            mu, t = calc_mu_t(i_old)
            i_max = np.shape(data)[0] - 1
            j = 0
            while i != i_old:   # sorta bisection, boolean hybrid monstrosity
                if np.abs(mu - np.mean(data[i_old:i])) > pred_int(i_old, i):
                    j = 0
                    i = int(np.floor((i+i_old)/2))
                else:
                    i_old = i
                    mu, t = calc_mu_t(i_old)    # calculate new mu, t
                    j += 1
                    i += j**3   # this is to speed up the search
                
                if i > i_max:
                    i = i_max
                    break
            
            return i 
        
        parent = self.parent
        
        if np.isnan(parent.display_shock['Sample_Rate']):
            self.clear_plot()
            return
        
        data = parent.display_shock['raw_data'].reshape(-1,)
        self.t = np.arange(np.shape(data)[0])/parent.display_shock['Sample_Rate']
        t = self.t
                
        if estimate_ind:
            self.start_ind = estimateInd(data, frac=0.1, alpha=0.002)   # 2-tail 99.9%
            self.end_ind = -estimateInd(data[::-1], frac=0.1, alpha=0.001) # 2-tail 99.95%
          
        start_ind = self.start_ind
        end_ind = self.end_ind
          
        self.ax[0].item['exp_data'].set_offsets(shape_data(t, data))
        
        start_avg = np.mean(data[:start_ind])
        set_xy(self.ax[0].item['start_divider'], (t[start_ind]), (start_avg))
        # self.ax[0].item['start_divider']_l.set_xdata(t[start_ind]*np.ones((2,1)))
        set_xy(self.ax[0].item['start_avg_l'], (t[0], t[start_ind]), start_avg*np.ones((2,1)))
        
        end_avg = np.mean(data[end_ind:])
        set_xy(self.ax[0].item['end_divider'], (t[end_ind]), (end_avg))
        # self.ax[0].item['end_divider']_l.set_xdata(t[end_ind]*np.ones((2,1)))
        set_xy(self.ax[0].item['end_avg_l'], (t[end_ind], t[-1]), end_avg*np.ones((2,1)))
        
        self.ax[0].item['textbox'].set_text(self.info_table_text(start_avg, end_avg))
        
        if update_lim:
            self.update_xylim(self.ax[0])
    
    