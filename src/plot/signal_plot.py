# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

from tabulate import tabulate

import matplotlib as mpl
import numpy as np

from plot.base_plot import Base_Plot
from plot.draggable import Draggable


class Plot(Base_Plot):
    def __init__(self, parent, widget, mpl_layout):
        super().__init__(parent, widget, mpl_layout)

        # Connect Signals
        self.canvas.mpl_connect('resize_event', self._resize_event)
        parent.num_sim_lines_box.valueChanged.connect(self.set_history_lines)
    
    def info_table_text(self):
        parent = self.parent
        # TODO: Fix variables when implementing zone 2 and 5 option
        shock_zone = parent.display_shock['zone']
        if shock_zone == 2:
            display_vars = ['T2', 'P2']
        elif shock_zone == 5:
            display_vars = ['T5', 'P5']
        
        table = [['Shock {:d}'.format(parent.var['shock_choice']), '']]
        
        # This sets the info table to have the units selected in the shock properties window
        if not np.isnan([parent.display_shock[key] for key in display_vars]).all():
            T_unit = eval('str(parent.' + display_vars[0] + '_units_box.currentText())')
            P_unit = eval('str(parent.' + display_vars[1] + '_units_box.currentText())')
            T_value = parent.convert_units(parent.display_shock[display_vars[0]], T_unit, 'out')
            P_value = parent.convert_units(parent.display_shock[display_vars[1]], P_unit, 'out')
            table.append(['T{:.0f} {:s}'.format(shock_zone, T_unit), '{:.2f}'.format(T_value)])
            table.append(['P{:.0f} {:s}'.format(shock_zone, P_unit), '{:.2f}'.format(P_value)])
        
        for species, mol_frac in parent.display_shock['thermo_mix'].items():
            table.append(['{:s}'.format(species), '{:g}'.format(mol_frac)])

        table = tabulate(table).split('\n')[1:-1] # removes header and footer
        
        table_left_justified = []
        max_len = len(max(table, key=len))
        for line in table:
            table_left_justified.append('{:<{max_len}}'.format(line, max_len=max_len))
            
        return '\n'.join(table_left_justified)
        
    def create_canvas(self):
        self.ax = []
        
        ## Set upper plots ##
        self.ax.append(self.fig.add_subplot(4,1,1))
        self.ax[0].item = {}
        self.ax[0].item['weight'] = self.ax[0].add_line(mpl.lines.Line2D([],[], c = '#800000', zorder=1))
        self.ax[0].item['weight_max'] = [self.ax[0].add_line(mpl.lines.Line2D([],[], marker='$'+u'\u2195'+'$', 
            markersize=12, markerfacecolor='#BF0000', markeredgecolor='None', linestyle='None', zorder=4))]
        lines = {'weight_shift': {'marker': 'o',               'markersize': 7}, 
                 'weight_k':     {'marker': '$'+'\u2194'+'$',  'markersize': 12}, 
                 'weight_min':   {'marker': '$'+u'\u2195'+'$', 'markersize': 12}}
        for name, attr in lines.items():        # TODO: Redo with fewer plots since Draggable does not forget point dragged
            self.ax[0].item[name] = []
            for i in range(0, 2):
                self.ax[0].item[name].append(self.ax[0].add_line(mpl.lines.Line2D([],[], marker=attr['marker'], 
                    markersize=attr['markersize'], markerfacecolor='#BF0000', markeredgecolor='None', 
                    linestyle='None', zorder=2)))
                self.ax[0].item[name][-1].info = {'n': i}

        self.ax[0].item['sim_info_text'] = self.ax[0].text(.98,.92, '', fontsize=10, fontname='DejaVu Sans Mono',
            horizontalalignment='right', verticalalignment='top', transform=self.ax[0].transAxes)
        
        self.ax[0].set_ylim(-0.1, 1.1)
        self.ax[0].tick_params(labelbottom=False)
        
        self.ax[0].text(.5,.95,'Weight Function', fontsize='large',
            horizontalalignment='center', verticalalignment='top', transform=self.ax[0].transAxes)
        
        self.fig.subplots_adjust(left=0.06, bottom=0.065, right=0.98,
                        top=0.98, hspace=0, wspace=0.12)
        
        ## Set lower plots ##
        self.ax.append(self.fig.add_subplot(4,1,(2,4), sharex = self.ax[0]))
        self.ax[1].item = {}
        self.ax[1].item['exp_data'] = self.ax[1].scatter([],[], color='0', facecolors='0',
            linewidth=0.5, alpha = 0.85)
        self.ax[1].item['sim_data'] = self.ax[1].add_line(mpl.lines.Line2D([],[], c='#0C94FC'))
        self.ax[1].item['history_data'] = []
        self.lastRxnNum = None
        
        self.ax[1].text(.5,.98,'Observable', fontsize='large',
            horizontalalignment='center', verticalalignment='top', transform=self.ax[1].transAxes)
                
        self.parent.rxn_change_history = []
        self.set_history_lines()
        
        # Create colorbar legend
        self.cbax = self.fig.add_axes([0.90, 0.575, 0.02, 0.15], zorder=3)
        self.cb = mpl.colorbar.ColorbarBase(self.cbax, cmap=mpl.cm.gray,
            ticks=[0, 0.5, 1], orientation='vertical')
        self.cbax.invert_yaxis()
        self.cbax.set_yticklabels(['1', '0.5', '0'])  # horizontal colorbar
        self.cb.set_label('Weighting')        
        
        # Create canvas from Base
        super().create_canvas()
        self._set_scale('y', 'abslog', self.ax[1])  # set Signal/SIM y axis to abslog
        
        # Add draggable lines
        draggable_items = [[0, 'weight_shift'], [0, 'weight_k'], [0, 'weight_min'],
                           [1, 'sim_data']]
        for pair in draggable_items:
            n, name = pair  # n is the axis number, name is the item key
            items = self.ax[n].item[name]   
            if not isinstance(items, list):     # check if the type is a list
                items = [self.ax[n].item[name]]
            for item in items:    
                update_fcn = lambda x, y, item=item: self.draggable_update_fcn(item, x, y)
                press_fcn = lambda x, y, item=item: self.draggable_press_fcn(item, x, y)
                release_fcn = lambda item=item: self.draggable_release_fcn(item)
                item.draggable = Draggable(self, item, update_fcn, press_fcn, release_fcn)       
    
    def set_history_lines(self):
        old_num_hist_lines = len(self.ax[1].item['history_data'])
        num_hist_lines = self.parent.num_sim_lines_box.value() - 1
        numDiff = np.abs(old_num_hist_lines - num_hist_lines)
        
        if old_num_hist_lines > num_hist_lines:
            del self.ax[1].item['history_data'][0:numDiff]
        elif old_num_hist_lines < num_hist_lines:
            for n in range(old_num_hist_lines, old_num_hist_lines+numDiff):
                line = mpl.lines.Line2D([],[])
                self.ax[1].item['history_data'].append({'line': self.ax[1].add_line(line), 'rxnNum': None})
        
        color = mpl.cm.nipy_spectral(np.linspace(0.05, 0.95, num_hist_lines)[::-1])
        for n, item in enumerate(self.ax[1].item['history_data']):
                item['line'].set_color(color[n])
        
        if hasattr(self, 'canvas'): # this can be deleted after testing color changes
            self._draw_items_artist() 
    
    def draggable_press_fcn(self, item, x, y):
        x0, xpress, xnew, xpressnew = x['0'], x['press'], x['new'], x['press_new']
        y0, ypress, ynew, ypressnew = y['0'], y['press'], y['new'], y['press_new']
        xy_data = item.get_xydata()
        
        distance_cmp = []
        for xy in xy_data:  # calculate distance from press and points, don't need sqrt for comparison
            distance_cmp.append((xy[0] - xpress)**2 + (xy[1] - ypress)**2)
        
        item.draggable.nearest_index = np.argmin(distance_cmp) # choose closest point to press
    
    def draggable_release_fcn(self, item):
        item.draggable.nearest_index = 0                       # reset nearest_index
    
    def draggable_update_fcn(self, item, x, y):
        parent = self.parent
        
        x = {key: np.array(val)/parent.var['reactor']['t_unit_conv'] for key, val in x.items()}    # scale with unit choice
        x0, xpress, xnew, xpressnew = x['0'], x['press'], x['new'], x['press_new']
        y0, ypress, ynew, ypressnew = y['0'], y['press'], y['new'], y['press_new']
        exp_data = parent.display_shock['exp_data']
        
        if item is self.ax[1].item['sim_data']:
            time_offset = np.round(xnew[0]/0.01)*0.01
            for box in parent.time_offset_box.twin:
                box.blockSignals(True)
                box.setValue(time_offset)
                box.blockSignals(False)

            parent.var['time_offset'] = parent.time_offset_box.value()*parent.var['reactor']['t_unit_conv']
            
            parent.tree._copy_expanded_tab_rates()  # update rates/time offset autocopy
            self.update_sim(parent.SIM.independent_var, parent.SIM.observable)
                
        elif item in self.ax[0].item['weight_shift']:
            t_conv = parent.var['reactor']['t_unit_conv']
            xnew = (xnew*t_conv - exp_data[0,0])/(exp_data[-1,0] - exp_data[0,0])*100
            
            # shift must be within the experiment
            n = item.info['n']
            if n == 0:
                if xnew < 0.0:
                    xnew = 0.0
                elif xnew > parent.display_shock['weight_shift'][1]:
                    xnew = parent.display_shock['weight_shift'][1]
            elif n == 1:
                if xnew < parent.display_shock['weight_shift'][0]:
                    xnew = parent.display_shock['weight_shift'][0]
                elif xnew > 100:
                    xnew = 100
            
            parent.weight.boxes['weight_shift'][n].setValue(xnew)
            
        elif item in self.ax[0].item['weight_k']:   # save n on press, erase on release
            xy_data = item.get_xydata()
            i = item.draggable.nearest_index
            n = item.info['n']
            
            shift = parent.display_shock['weight_shift'][n]
            shift = shift/100*(exp_data[-1,0] - exp_data[0,0]) + exp_data[0,0]
            shift /= parent.var['reactor']['t_unit_conv']
            
            # Calculate new sigma, shift - sigma or sigma - shift based on which point is selected
            sigma_new = -((-1)**(i))*(xnew[i] - shift)
            
            if sigma_new < 0:   # Sigma must be greater than 0
                sigma_new = 0
            
            parent.weight.boxes['weight_k'][n].setValue(sigma_new)
            
        elif item in self.ax[0].item['weight_min']:
            xnew = x0
            min_new = ynew
            # Must be greater than 0 and less than 0.99
            if min_new < 0:
                min_new = 0   # Let the GUI decide low end
            elif min_new > 1:
                min_new = 1
            
            parent.weight.boxes['weight_min'][1].setValue(min_new*100)     

        # Update plot if data exists
        if exp_data.size > 0:
            parent.update_user_settings()
            self.update()
        
    def _resize_event(self, event=None):
        canvas_width = self.canvas.size().width()
        left = -7.6E-08*canvas_width**2 + 2.2E-04*canvas_width + 7.55E-01   # Might be better to adjust by pixels
        self.cbax.set_position([left, 0.575, 0.02, 0.15])
    
    def _clear_event(self, event=None): # unused
        self.fig.clear()
    
    def update(self, update_lim=False):
        def shape_data(t,x): return np.transpose(np.vstack((t, x)))
        
        parent = self.parent

        t = parent.display_shock['exp_data'][:,0]
        data = parent.display_shock['exp_data'][:,1]
        #weight_shift = np.array(parent.display_shock['weight_shift'])*t_conv
        weight_shift = np.array(parent.display_shock['weight_shift'])/100*(t[-1] - t[0]) + t[0]
        weight_k = np.array(parent.display_shock['weight_k'])*self.parent.var['reactor']['t_unit_conv']
        
        weight_fcn = parent.series.weights
        weights = parent.display_shock['weights'] = weight_fcn(t)
        
        # Update lower plot
        self.ax[1].item['exp_data'].set_offsets(shape_data(t, data))
        self.ax[1].item['exp_data'].set_facecolor(np.char.mod('%f', 1-weights))
        
        # Update upper plot
        self.ax[0].item['weight'].set_xdata(t)
        self.ax[0].item['weight'].set_ydata(weights)
        
        for i in range(0, 2):   # TODO: need to intelligently reorder to fix draggable bug
            mu = weight_shift[i]
            f_mu = weight_fcn([mu], calcIntegral=False)
            sigma = np.sort(np.ones(2)*mu + np.array([1, -1])*weight_k[i]*(-1)**(i))
            f_sigma = weight_fcn(sigma, calcIntegral=False)

            if mu == sigma[0] and mu == sigma[1]:   # this happens if growth rate is inf
                f = weight_fcn(np.array([(1.0-1E-3), (1.0+1E-3)])*mu, calcIntegral=False)               
                
                f_mu = np.mean(f)
                perc = 0.1824
                f_sigma = [(1-perc)*f[0] + perc*f[1], perc*f[0] + (1-perc)*f[1]]
            
            self.ax[0].item['weight_shift'][i].set_xdata(mu)
            self.ax[0].item['weight_shift'][i].set_ydata(f_mu)
            
            self.ax[0].item['weight_k'][i].set_xdata(sigma)
            self.ax[0].item['weight_k'][i].set_ydata(f_sigma)
            
            x_weight_min = np.max(t)*0.95  # put arrow at 95% of x data
            self.ax[0].item['weight_min'][i].set_xdata(x_weight_min)
            self.ax[0].item['weight_min'][i].set_ydata(weight_fcn([x_weight_min], calcIntegral=False))
              
        self.update_info_text()
        
        if update_lim:
            self.update_xylim(self.ax[1])
    
    def update_info_text(self, redraw=False):
        self.ax[0].item['sim_info_text'].set_text(self.info_table_text())
        if redraw:
            self._draw_items_artist()
    
    def clear_sim(self):
        self.ax[1].item['sim_data'].raw_data = np.array([])
        self.ax[1].item['sim_data'].set_xdata([])
        self.ax[1].item['sim_data'].set_ydata([])
    
    def update_sim(self, t, observable, rxnChanged=False):
        time_offset = self.parent.display_shock['time_offset']
        exp_data = self.parent.display_shock['exp_data']
        
        self.ax[0].item['sim_info_text'].set_text(self.info_table_text())
        
        if len(self.ax[1].item['history_data']) > 0:
            self.update_history()
        
        # logic to update lim
        self.sim_update_lim = False
        if hasattr(self.ax[1].item['sim_data'], 'raw_data'):
            old_data = self.ax[1].item['sim_data'].raw_data
            if old_data.size == 0 or old_data.ndim != 2 or old_data[-1,0] != t[-1]:
                self.sim_update_lim = True
        else:
            self.sim_update_lim = True
        
        self.ax[1].item['sim_data'].raw_data = np.array([t, observable]).T
        self.ax[1].item['sim_data'].set_xdata(t + time_offset)
        self.ax[1].item['sim_data'].set_ydata(observable)
        
        if exp_data.size == 0 and not np.isnan(t).any(): # if exp data doesn't exist rescale
            self.set_xlim(self.ax[1], [t[0], t[-1]])
            if np.count_nonzero(observable) > 0:    # only update ylim if not all values are zero
                self.set_ylim(self.ax[1], observable)
                self._draw_event()
        else:
            if self.sim_update_lim:
                self.update_xylim(self.ax[1])
            else:
                self._draw_items_artist()
   
    def update_history(self):
        def reset_history_lines(line):  
            for n in range(0,len(line)):
                line[n]['line'].set_xdata([])
                line[n]['line'].set_ydata([])
                line[n]['rxnNum'] = None
            
        numHist = self.parent.num_sim_lines_box.value()
        rxnHist = self.parent.rxn_change_history
        
        if len(rxnHist) > 0:
            if self.lastRxnNum != rxnHist[-1]:    # only update if the rxnNum changed
                self.lastRxnNum = rxnHist[-1]
            else:
                if self.lastRxnNum is None:       # don't update from original mech
                    self.lastRxnNum = rxnHist[-1]
                return        
        else:
            self.lastRxnNum = None
            reset_history_lines(self.ax[1].item['history_data'])
            return
        
        histRxnNum = [item['rxnNum'] for item in self.ax[1].item['history_data']]  
        
        if rxnHist[-1] in histRxnNum:           # if matching rxnNum, replace that
            n = histRxnNum.index(rxnHist[-1])
        else:
            firstNone = next((n for n, x in enumerate(histRxnNum) if x is None), None)
            
            if firstNone is not None:
                n = firstNone
            else:                               # if no matching rxnNums, replace differing rxnNum
                s = set(histRxnNum)                     
                n = [n for n, x in enumerate(rxnHist[:-numHist:-1]) if x not in s][0]
            
        hist = self.ax[1].item['history_data'][n]
        hist['rxnNum'] = rxnHist[-1]
        hist['line'].set_xdata(self.ax[1].item['sim_data'].get_xdata())
        hist['line'].set_ydata(self.ax[1].item['sim_data'].get_ydata())    
        
