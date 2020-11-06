import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity
#from sklearn.model_selection import GridSearchCV, LeaveOneOut
from tabulate import tabulate
from copy import deepcopy
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backend_bases import key_press_handler
from matplotlib import scale as mplscale
from matplotlib import figure as mplfigure
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from PyQt5.QtWidgets import QMenu, QAction
from PyQt5 import QtCore, QtGui


class All_Plots:    # container to hold all plots
    def __init__(self, main):
        global parent
        parent = main
        
        self.raw_sig = Raw_Signal_Plot(parent, parent.raw_signal_plot_widget, parent.mpl_raw_signal)
        self.signal = Signal_Plot(parent, parent.signal_plot_widget, parent.mpl_signal)
        self.sim_explorer = Sim_Explorer_Plot(parent, parent.sim_explorer_plot_widget, parent.mpl_sim_explorer)
        self.opt = Optimization_Plot(parent, parent.opt_plot_widget, parent.mpl_opt)


class Base_Plot:
    def __init__(self, parent, widget, mpl_layout):
        self.widget = widget
        self.mpl_layout = mpl_layout
        self.fig = mplfigure.Figure()
        mpl.scale.register_scale(AbsoluteLogScale)
        mpl.scale.register_scale(BiSymmetricLogScale)
        
        # Set plot variables
        self.x_zoom_constraint = False
        self.y_zoom_constraint = False
        
        self.create_canvas()        
        self.NavigationToolbar(self.canvas, self.widget, coordinates=True)
        
        # AutoScale
        self.autoScale = [True, True]
        
        # Connect Signals
        self._draw_event_signal = self.canvas.mpl_connect('draw_event', self._draw_event)
        self.canvas.mpl_connect('button_press_event', lambda event: self.click(event))
        self.canvas.mpl_connect('key_press_event', lambda event: self.key_press(event))
        # self.canvas.mpl_connect('key_release_event', lambda event: self.key_release(event))
        
        self._draw_event()
    
    def create_canvas(self):
        self.canvas = FigureCanvas(self.fig)
        self.mpl_layout.addWidget(self.canvas)
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.canvas.draw()
        
        # Set scales
        scales = {'linear': True, 'log': 0, 'abslog': 0, 'bisymlog': 0}
        for ax in self.ax:
            ax.scale = {'x': scales, 'y': deepcopy(scales)}
        
        # Get background
        for ax in self.ax:
            ax.background = self.canvas.copy_from_bbox(ax.bbox)
    
    def _find_calling_axes(self, event):
        for axes in self.ax:    # identify calling axis
            if axes == event or (hasattr(event, 'inaxes') and event.inaxes == axes):
                return axes
    
    def set_xlim(self, axes, x):
        if not self.autoScale[0]: return    # obey autoscale right click option
    
        if axes.get_xscale() in ['linear']:
            range = np.abs(np.max(x) - np.min(x))
            min = np.min(x) - range*0.05
            if min < 0:
                min = 0
            xlim = [min, np.max(x) + range*0.05]
        if 'log' in axes.get_xscale():
            abs_x = np.abs(x)
            abs_x = abs_x[np.nonzero(abs_x)]    # exclude 0's
            
            if axes.get_xscale() in ['log', 'abslog', 'bisymlog']:
                min_data = np.ceil(np.log10(np.min(abs_x)))
                max_data = np.floor(np.log10(np.max(abs_x)))
                
                xlim = [10**(min_data-1), 10**(max_data+1)]
        
        if np.isnan(xlim).any() or np.isinf(xlim).any():
            pass
        elif xlim != axes.get_xlim():   # if xlim changes
            axes.set_xlim(xlim)
    
    def set_ylim(self, axes, y):
        if not self.autoScale[1]: return    # obey autoscale right click option
        
        min_data = np.min(y)
        max_data = np.max(y)
        
        if min_data == max_data:
            min_data -= 10**-1
            max_data += 10**-1
        
        if axes.get_yscale() == 'linear':
            range = np.abs(max_data - min_data)
            ylim = [min_data - range*0.1, max_data + range*0.1]
            
        elif axes.get_yscale() in ['log', 'abslog']:
            abs_y = np.abs(y)
            abs_y = abs_y[np.nonzero(abs_y)]    # exclude 0's
            
            if abs_y.size == 0:             # if no data, assign 
                ylim = [10**-7, 10**-1]
            else:            
                min_data = np.ceil(np.log10(np.min(abs_y)))
                max_data = np.floor(np.log10(np.max(abs_y)))
                
                ylim = [10**(min_data-1), 10**(max_data+1)]
                
        elif axes.get_yscale() == 'bisymlog':
            min_sign = np.sign(min_data)
            max_sign = np.sign(max_data)
            
            if min_sign > 0:
                min_data = np.ceil(np.log10(np.abs(min_data)))
            elif min_data == 0 or max_data == 0:
                pass
            else:
                min_data = np.floor(np.log10(np.abs(min_data)))
            
            if max_sign > 0:
                max_data = np.floor(np.log10(np.abs(max_data)))
            elif min_data == 0 or max_data == 0:
                pass
            else:
                max_data = np.ceil(np.log10(np.abs(max_data)))
            
            # TODO: ylim could be incorrect for neg/neg, checked for pos/pos, pos/neg
            ylim = [min_sign*10**(min_data-min_sign), max_sign*10**(max_data+max_sign)]
        
        if ylim != axes.get_ylim():   # if ylim changes, update
            axes.set_ylim(ylim)
    
    def update_xylim(self, axes):
        data = self._get_data(axes)         
        
        # on creation, there is no data, don't update
        if np.shape(data['x'])[0] < 2 or np.shape(data['y'])[0] < 2:   
            return
        
        self.set_ylim(axes, data['y'])
        self.set_xlim(axes, data['x'])
        self._draw_event()  # force a draw
    
    def _get_data(self, axes):      # NOT Generic
        # get experimental data for axes
        data = {'x': [], 'y': []}
        if 'exp_data' in axes.item:
            data_plot = axes.item['exp_data'].get_offsets().T
            if np.shape(data_plot)[1] > 1:
                data['x'] = data_plot[0,:]
                data['y'] = data_plot[1,:]
        elif 'weight_l' in axes.item:
            data['x'] = axes.item['weight_l'].get_xdata()
            data['y'] = axes.item['weight_l'].get_ydata()
        
        return data
    
    def _set_scale(self, coord, type, event, update_xylim=False):
        # find correct axes
        axes = self._find_calling_axes(event)
        # for axes in self.ax:
            # if axes == event or (hasattr(event, 'inaxes') and event.inaxes == axes):
                # break
                
        # Set scale menu boolean
        if coord == 'x':
            shared_axes = axes.get_shared_x_axes().get_siblings(axes)               
        else:
            shared_axes = axes.get_shared_y_axes().get_siblings(axes)
        
        for shared in shared_axes:
            shared.scale[coord] = dict.fromkeys(shared.scale[coord], False) # sets all types: False
            shared.scale[coord][type] = True                                # set selected type: True

        # Apply selected scale
        if type == 'linear':
            str = 'axes.set_{:s}scale("{:s}")'.format(coord, 'linear')
        elif type == 'log':
            str = 'axes.set_{0:s}scale("{1:s}", nonpos{0:s}="mask")'.format(coord, 'log')
        elif type == 'abslog':
            str = 'axes.set_{:s}scale("{:s}")'.format(coord, 'abslog')
        elif type == 'bisymlog':
            data = self._get_data(axes)[coord]
            data_max = np.max(data)
            data_min = np.min(data)
            
            if len(data) == 0 or data_min == data_max:  # backup in case set on blank plot
                str = 'axes.set_{0:s}scale("{1:s}")'.format(coord, 'bisymlog')
            else:
                # if zero is within total range, find largest pos or neg range
                if np.sign(data_max) != np.sign(data_min):  
                    pos_range = np.max(data[data>=0])-np.min(data[data>=0])
                    neg_range = np.max(-data[data<=0])-np.min(-data[data<=0])
                    C = np.max([pos_range, neg_range])
                else:
                    C = np.abs(data_max-data_min)
                C /= 5E2                             # scaling factor, debating between 100 and 1000
                
                str = 'axes.set_{0:s}scale("{1:s}", C={2:e})'.format(coord, 'bisymlog', C)
        
        eval(str)
        if type == 'linear':
            # axes.yaxis.set_major_formatter(OoMFormatter(4, "%1.3f"))
            formatter = mpl.ticker.ScalarFormatter(useOffset=False, useMathText=True)
            formatter.set_powerlimits([-3, 4])
            axes.yaxis.set_major_formatter(formatter)
        if update_xylim:
            self.update_xylim(axes)
 
    def _animate_items(self, bool=True):
        for axis in self.ax:
            if axis.get_legend() is not None:
                axis.get_legend().set_animated(bool)
            
            for item in axis.item.values():
                if isinstance(item, list):
                    for subItem in item:
                        if isinstance(subItem, dict):
                            subItem['line'].set_animated(bool)
                        else:
                            subItem.set_animated(bool)
                else:
                    item.set_animated(bool)
    
    def _draw_items_artist(self):   
        for axis in self.ax:     # restore background first (needed for twinned plots)
            self.canvas.restore_region(axis.background)   
        
        for axis in self.ax:
            for item in axis.item.values():
                if isinstance(item, list):
                    for subItem in item:
                        if isinstance(subItem, dict):
                            axis.draw_artist(subItem['line'])
                        else:
                            axis.draw_artist(subItem) 
                else:
                    axis.draw_artist(item)
           
            if axis.get_legend() is not None:
                axis.draw_artist(axis.get_legend())
            
        self.canvas.update()
        # self.canvas.flush_events()    # unnecessary?
    
    def set_background(self):
        self.canvas.draw_idle() # for when shock changes
        for axis in self.ax:
            # axis.background = self.canvas.copy_from_bbox(axis.bbox)
            axis.background = self.canvas.copy_from_bbox(self.fig.bbox)
    
    def _draw_event(self, event=None):   # After redraw (new/resizing window), obtain new background
        self._animate_items(True)
        self.set_background()
        self._draw_items_artist()     
        # self.canvas.draw_idle()   # unnecessary?
    
    def clear_plot(self):
        for axis in self.ax:
            if axis.get_legend() is not None:
                axis.get_legend().remove()
                
            for item in axis.item.values():
                if hasattr(item, 'set_offsets'):    # clears all data points
                    item.set_offsets(([np.nan, np.nan]))
                elif hasattr(item, 'set_xdata') and hasattr(item, 'set_ydata'):
                    item.set_xdata([np.nan]) # clears all lines
                    item.set_ydata([np.nan])
                elif hasattr(item, 'set_text'): # clears all text boxes
                    item.set_text('')
        
        self._draw_event()

    def click(self, event):
        if event.button == 3: # if right click
            if self.toolbar._active is None:
                self._popup_menu(event)
            # if self.toolbar._active is 'ZOOM':  # if zoom is on, turn off
                # self.toolbar.press_zoom(event)  # cancels current zooom
                # self.toolbar.zoom()             # turns zoom off
            elif event.dblclick:                  # if double right click, go to default view
                self.toolbar.home()

    def key_press(self, event):
        if event.key == 'escape':
            if self.toolbar._active is 'ZOOM':  # if zoom is on, turn off
                self.toolbar.zoom()             # turns zoom off
            elif self.toolbar._active is 'PAN':
                self.toolbar.pan()
        # elif event.key == 'shift':
        elif event.key == 'x':  # Does nothing, would like to make sticky constraint zoom/pan
            self.x_zoom_constraint = not self.x_zoom_constraint
        elif event.key == 'y':  # Does nothing, would like to make sticky constraint zoom/pan
            self.y_zoom_constraint = not self.y_zoom_constraint
        elif event.key in ['s', 'l', 'L', 'k']: pass
        else:
            key_press_handler(event, self.canvas, self.toolbar)
    
    # def key_release(self, event):
        # print(event.key, 'released')
    
    def NavigationToolbar(self, *args, **kwargs):
        ## Add toolbar ##
        self.toolbar = NavigationToolbar2QT(self.canvas, self.widget, coordinates=True)
        # print(self.toolbar.iteritems)
        self.mpl_layout.addWidget(self.toolbar)

    def _popup_menu(self, event):
        axes = self._find_calling_axes(event)   # find axes calling right click
        if axes is None: return
        
        pos = parent.mapFromGlobal(QtGui.QCursor().pos())
        
        popup_menu = QMenu(parent)
        xScaleMenu = popup_menu.addMenu('x-scale')
        yScaleMenu = popup_menu.addMenu('y-scale')
        
        for coord in ['x', 'y']:
            menu = eval(coord + 'ScaleMenu')
            for type in axes.scale[coord].keys():
                action = QAction(type, menu, checkable=True)
                if axes.scale[coord][type]: # if it's checked
                    action.setEnabled(False)
                else:
                    action.setEnabled(True)
                menu.addAction(action)
                action.setChecked(axes.scale[coord][type])
                fcn = lambda bool, coord=coord, type=type: self._set_scale(coord, type, event, True)
                action.triggered.connect(fcn)
        
        # Create menu for AutoScale options X Y All
        popup_menu.addSeparator()
        autoscale_options = ['AutoScale X', 'AutoScale Y', 'AutoScale All']
        for n, text in enumerate(autoscale_options):
            action = QAction(text, menu, checkable=True)
            if n < len(self.autoScale):
                action.setChecked(self.autoScale[n])
            else:
                action.setChecked(all(self.autoScale))
            popup_menu.addAction(action)
            action.toggled.connect(lambda event, n=n: self._setAutoScale(n, event, axes))
                    
        popup_menu.exec_(parent.mapToGlobal(pos))    
    
    def _setAutoScale(self, choice, event, axes):
        if choice == len(self.autoScale):
            for n in range(len(self.autoScale)):
                self.autoScale[n] = event
        else:
            self.autoScale[choice] = event
        
        if event:   # if something toggled true, update limits
            self.update_xylim(axes)

                
class Raw_Signal_Plot(Base_Plot):
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
        
        self.fig.subplots_adjust(left=0.06, bottom=0.05, right=0.98,
                        top=0.98, hspace=0, wspace=0.12)
        
        # Create canvas from Base
        super().create_canvas()
        
        # Add draggable lines
        draggable_items = [[0, 'start_divider'], [0, 'end_divider']]
        for pair in draggable_items:
            n, name = pair  # n is the axis number, name is the item key
            update_fcn = lambda press, name=name: self.draggable_update_fcn(name, press)
            self.ax[n].item[name].draggable = Draggable(self, self.ax[n].item[name], update_fcn)
      
    def draggable_update_fcn(self, name, press):
        if parent.display_shock['raw_data'].size == 0: return
        
        x0, y0, xpress, ypress, xnew, ynew = press   
        
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
    
    
class Signal_Plot(Base_Plot):
    def __init__(self, parent, widget, mpl_layout):
        super().__init__(parent, widget, mpl_layout)
        
        # Connect Signals
        self.canvas.mpl_connect('resize_event', self._resize_event)
        parent.num_sim_lines_box.valueChanged.connect(self.set_history_lines)
    
    def info_table_text(self):
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
        self.ax[0].item['weight_l'] = self.ax[0].add_line(mpl.lines.Line2D([],[], c = '#800000', zorder=1))
        self.ax[0].item['weight_shift'] = self.ax[0].add_line(mpl.lines.Line2D([],[], marker='o', markersize=7,
            markerfacecolor='#BF0000', markeredgecolor='None', zorder=2))
        self.ax[0].item['weight_k'] = self.ax[0].add_line(mpl.lines.Line2D([],[], marker='$'+'\u2194'+'$', 
            markersize=12, markerfacecolor='#BF0000', markeredgecolor='None', linestyle='None', zorder=3))    
        self.ax[0].item['weight_min'] = self.ax[0].add_line(mpl.lines.Line2D([],[], marker='$'+u'\u2195'+'$', 
            markersize=12, markerfacecolor='#BF0000', markeredgecolor='None', linestyle='None', zorder=4))
        self.ax[0].item['sim_info_text'] = self.ax[0].text(.98,.92, '', fontsize=10, fontname='DejaVu Sans Mono',
            horizontalalignment='right', verticalalignment='top', transform=self.ax[0].transAxes)
        
        self.ax[0].set_ylim(-0.1, 1.1)
        self.ax[0].tick_params(labelbottom=False)
        
        self.ax[0].text(.5,.95,'Weight Function', fontsize='large',
            horizontalalignment='center', verticalalignment='top', transform=self.ax[0].transAxes)
        
        self.fig.subplots_adjust(left=0.06, bottom=0.05, right=0.98,
                        top=0.98, hspace=0, wspace=0.12)
        
        ## Set lower plots ##
        self.ax.append(self.fig.add_subplot(4,1,(2,4), sharex = self.ax[0]))
        self.ax[1].item = {}
        self.ax[1].item['exp_data'] = self.ax[1].scatter([],[], color='0', facecolors='0',
            linewidth=0.5, alpha = 0.85)
        self.ax[1].item['sim_data'] = self.ax[1].add_line(mpl.lines.Line2D([],[], c='#0C94FC'))
        self.ax[1].item['history_data'] = []
        self.lastRxnNum = None
        self.ax[1].item['cutoff_l'] = self.ax[1].axvline(x=1, ls='--', c = '#BF0000')
        
        self.ax[1].text(.5,.98,'dρ/dx', fontsize='large',
            horizontalalignment='center', verticalalignment='top', transform=self.ax[1].transAxes)
        
        parent.rxn_change_history = []
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
                           [1, 'cutoff_l'], [1, 'sim_data']]
        for pair in draggable_items:
            n, name = pair  # n is the axis number, name is the item key
            update_fcn = lambda press, name=name: self.draggable_update_fcn(name, press)
            self.ax[n].item[name].draggable = Draggable(self, self.ax[n].item[name], update_fcn)       
    
    def set_history_lines(self):
        old_num_hist_lines = len(self.ax[1].item['history_data'])
        num_hist_lines = parent.num_sim_lines_box.value() - 1
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
    
    def draggable_update_fcn(self, name, press):
        x0, y0, xpress, ypress, xnew, ynew = press
        exp_data = parent.display_shock['exp_data']
        
        def setBoxValue(parent, object, value):
            # set GUI value and temporarily turns box from sending signal
            signal = eval('parent.' + object + '.blockSignals(True)')
            eval('parent.' + object + '.setValue(value)')
            eval('parent.' + object + '.blockSignals(False)')        
        
        if name is 'cutoff_l':
            exp_time = exp_data[:,0]
            
            xnew_closest_ind = np.argmin(np.abs(exp_time - xnew[0]))
            
            # Update ind box
            setBoxValue(parent, 'start_ind_box', xnew_closest_ind)
            
        elif name is 'sim_data':
            time_offset = np.round(xnew[0]/0.01)*0.01
            for box in parent.time_offset_box.twin:
                box.blockSignals(True)
                box.setValue(time_offset)
                box.blockSignals(False)

            parent.var['time_offset'] = parent.time_offset_box.value()
            
            parent.tree._copy_expanded_tab_rates()  # update rates/time offset autocopy
            self.update_sim(parent.SIM.t_lab, parent.SIM.observable)
                
        elif name is 'weight_shift':
           # shift must be within the experiment
            if xnew < exp_data[0,0]:
                xnew = exp_data[0,0]
            elif xnew > exp_data[-1,0]:
                xnew = exp_data[-1,0]
            
            weight_shift = xnew - exp_data[0,0]
            setBoxValue(parent, 'weight_shift_box', weight_shift)
            
        elif name is 'weight_k':
            shift = exp_data[0,0] + parent.display_shock['weight_shift']

            xy_data = self.ax[0].item['weight_k'].get_xydata()
            distance_cmp = []
            for xy in xy_data:  # calculate distance from press and points, don't need sqrt for comparison
                distance_cmp.append((xy[0] - xpress)**2 + (xy[1] - ypress)**2)
            
            n = np.argmin(distance_cmp) # choose closest point to press
            
            # Calculate new sigma, shift - sigma or sigma - shift based on which point is selected
            sigma_new = (-(-1)**(n))*(xnew[n] - shift)
            
            if sigma_new < 0:   # Sigma must be greater than 0
                sigma_new = 0   # Let the GUI decide low end
            
            setBoxValue(parent, 'weight_k_box', sigma_new)
            
        elif name is 'weight_min':
            xnew = x0
            min_new = ynew
            # Must be greater than 0 and less than 0.99
            if min_new < 0:
                min_new = 0   # Let the GUI decide low end
            elif min_new > 1:
                min_new = 1
            
            setBoxValue(parent, 'weight_min_box', min_new*100)     

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
        
        t = parent.display_shock['exp_data'][:,0]
        data = parent.display_shock['exp_data'][:,1]
        start_ind = parent.display_shock['start_ind']
        weight_shift = parent.display_shock['weight_shift']
        weight_k = parent.display_shock['weight_k']
        
        weight_fcn = parent.series.weights
        
        mu = t[0] + weight_shift
        sigma = [mu - weight_k, mu + weight_k]
        weights = parent.display_shock['weights'] = weight_fcn(t)
        
        # Update lower plot
        self.ax[1].item['exp_data'].set_offsets(shape_data(t, data))
        self.ax[1].item['cutoff_l'].set_xdata(t[start_ind]*np.ones((2,1)))
        self.ax[1].item['exp_data'].set_facecolor(np.char.mod('%f', 1-weights))
        
        # Update upper plot
        self.ax[0].item['weight_l'].set_xdata(t)
        self.ax[0].item['weight_l'].set_ydata(weights)
        
        self.ax[0].item['weight_shift'].set_xdata(mu)
        self.ax[0].item['weight_shift'].set_ydata(weight_fcn([mu], removePreCutoff=False))
        
        self.ax[0].item['weight_k'].set_xdata(sigma)
        self.ax[0].item['weight_k'].set_ydata(weight_fcn(sigma, removePreCutoff=False))
        
        x_weight_min = np.max(t)*0.95  # put arrow at 95% of x data
        self.ax[0].item['weight_min'].set_xdata(x_weight_min)
        self.ax[0].item['weight_min'].set_ydata(weight_fcn([x_weight_min], removePreCutoff=False))
              
        self.update_info_text()
        
        if update_lim:
            self.update_xylim(self.ax[1])
    
    def update_info_text(self, redraw=False):
        self.ax[0].item['sim_info_text'].set_text(self.info_table_text())
        if redraw:
            self._draw_items_artist()
    
    def clear_sim(self):
        self.ax[1].item['sim_data'].set_xdata([])
        self.ax[1].item['sim_data'].set_ydata([])
    
    def update_sim(self, t, observable, rxnChanged=False):
        time_offset = parent.display_shock['time_offset']
        exp_data = parent.display_shock['exp_data']
        
        self.ax[0].item['sim_info_text'].set_text(self.info_table_text())
        
        if len(self.ax[1].item['history_data']) > 0:
            self.update_history()
        
        self.ax[1].item['sim_data'].set_xdata(t + time_offset)
        self.ax[1].item['sim_data'].set_ydata(observable)
        
        if exp_data.size == 0: # if exp data doesn't exist rescale
            self.set_xlim(self.ax[1], [np.round(np.min(t))-1, np.round(np.max(t))+1])
            if np.count_nonzero(observable):    # only update ylim if not all values are zero
                self.set_ylim(self.ax[1], observable)
        
        self._draw_items_artist()
   
    def update_history(self):
        def reset_history_lines(line):  
            for n in range(0,len(line)):
                line[n]['line'].set_xdata([])
                line[n]['line'].set_ydata([])
                line[n]['rxnNum'] = None
            
        numHist = parent.num_sim_lines_box.value()
        rxnHist = parent.rxn_change_history
        
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
        

class Sim_Explorer_Plot(Base_Plot):
    def __init__(self, parent, widget, mpl_layout):
        super().__init__(parent, widget, mpl_layout)

    def create_canvas(self):
        self.ax = []
        self.ax.append(self.fig.add_subplot(1,1,1))
        self.ax.append(self.ax[0].twinx())
        self.ax[0].set_zorder(1)
        self.ax[1].set_zorder(0)    # put secondary axis behind primary
        
        max_lines = parent.sim_explorer.max_history + 1
        color = [plt.rcParams['axes.prop_cycle'].by_key()['color'],
                 plt.rcParams['axes.prop_cycle'].by_key()['color']]
        for n in range(3):
            color[1].append(color[1].pop(0))
        # color = mpl.cm.jet(np.linspace(0.05, 0.95, max_lines)[::-1])
        ls = ['-', '--']
        for n, ax in enumerate(self.ax):
            ax.item = {'property': [], 'legend': []}
            for i in range(max_lines):
                ax.item['property'].append(ax.add_line(mpl.lines.Line2D([],[], ls=ls[n], c=color[n][i])))

        # Create canvas from Base
        super().create_canvas()
        self.toggle_y2_axis(show_y2=False)
    
    def toggle_y2_axis(self, show_y2=False):
        if show_y2:
            self.ax[1].get_yaxis().set_visible(True)
            self.fig.subplots_adjust(left=0.06, bottom=0.05, right=0.94,
                                     top=0.97, hspace=0, wspace=0.12)
        else:
            self.ax[1].get_yaxis().set_visible(False)
            self.fig.subplots_adjust(left=0.06, bottom=0.05, right=0.98,
                                     top=0.97, hspace=0, wspace=0.12)
    
    def update(self, data, labels=[], update_lim=True):
        def set_xy(line, x, y): 
            line.set_xdata(x)
            line.set_ydata(y)

        legend_loc = {'y': 'upper left', 'y2': 'upper right'}
        # if reason:
            # self.clear_plot()
            # return
        for n, key in enumerate(data):
            axisData = data[key]
            if n == 0:
                xData = axisData[0] # if x axis is data set it and continue to y axes
                continue
            
            label = labels[key]
            axes = self.ax[n-1]
            
            # clear lines, labels and legend
            if isinstance(axes.get_legend(), mpl.legend.Legend):    # if legend exists, remove it
                leg = axes.get_legend()
                legend_loc[key] = leg._loc
                leg.remove()
            for line in axes.item['property']:  # clear lines and labels
                set_xy(line, [], [])
                line.set_label('')
            
            for i, yData in enumerate(axisData):        # update lines, won't update if empty list
                set_xy(axes.item['property'][i], xData, yData)
                if len(label) > 1:
                    axes.item['property'][i].set_label(label[i])
                    
            if len(label) > 1:
                leg = axes.legend(loc=legend_loc[key])
                leg._draggable = DraggableLegend(self, leg, use_blit=True, update='loc')
                if n == 2:  # if legend is on secondary y plot, set click passthrough
                    Matplotlib_Click_Passthrough(leg)
                        
            if n == 2:  # resize if y2 shown
                if np.size(axisData) > 0:
                    self.toggle_y2_axis(show_y2=True)
                else:
                    self.toggle_y2_axis(show_y2=False)
        
            if update_lim:  # update limits
                self.update_xylim(axes)
            
    def _get_data(self, axes):
        # get experimental data for axes
        data = {'x': [], 'y': []}
        if 'property' in axes.item:
            data['x'] = axes.item['property'][0].get_xdata()        # xdata always the same
            for line in axes.item['property']:
                data['y'] = np.append(data['y'], line.get_ydata())  # combine all lines on y
        
        return data


class Optimization_Plot(Base_Plot):
    def __init__(self, parent, widget, mpl_layout):
        super().__init__(parent, widget, mpl_layout)
        
        self.kde = KernelDensity(kernel='gaussian', bandwidth=1, rtol=1E-4)
        # self.kde = KernelDensity(kernel='epanechnikov', bandwidth=1, rtol=1E-4)
        
        # FOR TESTING
        self.resid = [-5.0978353223E-06, -5.1353181518E-06, -4.2204915790E-06, -4.4650813208E-06, -3.4848230945E-06, -3.5958056172E-06, -2.6862486061E-06, -1.1648507969E-06, -3.1188520844E-06, -4.5290841884E-06, -2.4663129888E-06, -3.0614163658E-06, -3.3170411993E-06, -2.3583336929E-07, -3.5739592182E-07, -3.0004969336E-06, -3.8053645570E-06, -4.4068867618E-06, -4.1238329291E-06, -3.2286674403E-06, -6.3143167636E-07, -3.2804624294E-06, -3.1374498379E-06, -2.0416478720E-06, -1.9004325032E-06, -2.7137877025E-06, -3.3236374414E-06, -3.1168766929E-06, -3.3877901013E-06, -2.2970949905E-06, -3.5233189432E-06, -2.5704525416E-06, -2.5593736825E-07, 2.8666899448E-07, -2.5775270359E-06, -3.2625810417E-06, -2.8584157491E-06, -3.4767752280E-06, -3.4827412823E-06, -2.9444247232E-06, -2.2024173620E-06, -1.4610740099E-06, -1.7421994781E-06, -1.7514805779E-06, -2.5788441290E-06, -1.9762846217E-06, -5.5687969008E-07, 8.6192391992E-07, 5.7710846239E-07, 1.1091651914E-06, -1.9862063884E-07, -1.9156967744E-06, -3.4289639610E-06, -3.7165951319E-06, -2.7104633418E-06, -9.5552806140E-07, -9.7226376211E-07, -1.4663619153E-06, -3.1190259924E-06, -1.5704964647E-06, -4.9930680368E-07, -5.8666149565E-07, -4.7012850315E-07, -1.4440053519E-06, -1.1921666002E-06, -1.7582388063E-06, -2.1885205285E-06, -1.0524613250E-06, -2.3692007542E-06, -1.0296493743E-06, -6.4421891814E-07, -1.0085418064E-06, -1.7820056293E-06, -1.1253359772E-06, -9.4591844040E-07, 1.1867539080E-07, 8.4226992612E-07, 6.7990157526E-07, 4.0303747939E-08, 1.0352358539E-06, 1.5529482599E-06, 2.6152617870E-06, 1.4973378784E-06, 1.7414789704E-06, 1.5765384989E-06, 2.7055508999E-06, 2.2379960959E-07, -1.8495659361E-06, -3.6507823012E-06, -3.4768170495E-06, -2.3494847479E-06, -2.5848962221E-06, -1.3219627646E-06, -3.3182871688E-07, 1.4754465797E-06, 2.7374447838E-06, 4.4759855540E-06, 4.3749565488E-06, 3.5924194268E-06, 1.3790448468E-06, -2.2152953286E-07, 4.9372787397E-07, -8.3493072337E-07, -7.3332623718E-07, -9.0448257985E-07, -8.0342866376E-07, 4.5541359872E-07, 1.5777462952E-06, 1.4055135133E-06, -5.3813065930E-07, -6.4276213506E-07, -3.3892183839E-07, -3.0781891574E-07, 8.8109229175E-07, -1.1011756370E-07, -7.9755829825E-08, -1.0714528544E-06, -2.9223998519E-07, 7.5922143004E-07, 1.1292380435E-06, 3.0658085076E-06, 1.5279734739E-06, 1.0798504785E-06, 3.5901978944E-07, 1.6134796367E-06, 3.5489312503E-06, 1.6693838603E-06, 1.9694976966E-06, 1.7244289895E-06, 2.3647249688E-06, 2.0511178646E-06, 2.2141519070E-06, -9.6095267405E-07, -9.3456564848E-07, -4.9965078741E-07, 1.0250050050E-06, 1.3232845694E-06, 2.5894871781E-07, 2.4642352620E-06, 2.5575770139E-06, 6.7521878528E-07, -9.3484161187E-07, -1.8638743657E-06, -4.0885166437E-07, 1.3184753040E-06, 1.0019893513E-06, 2.3202332895E-06, 1.5265479287E-06, 2.0951109778E-06, 1.4373219838E-06, 6.4311947896E-07, 5.3111995870E-08, 4.1663306697E-07, -1.7370777527E-07, -1.4880998388E-08, 7.5687293007E-07, 2.3416554258E-07, 1.8911723716E-06, 1.9812359496E-06, 1.1855688090E-06, 6.6223048045E-07, 1.7055244169E-06, 1.3181219685E-06, 1.1311547780E-07, -1.0239197122E-06, -1.2755282587E-06, -5.0546781896E-07, -6.8924805008E-07, 1.5791863907E-06, 1.6893884628E-07, -7.6460334059E-07, -2.0388908271E-06, -1.7974627035E-07, -7.7309933121E-07, -7.5349877163E-07, 1.5154153386E-07, 7.1584170484E-07, -4.9114013910E-07, 4.1350012163E-07, -5.2126039337E-07, -2.9809056454E-07, -5.5189827226E-07, 6.9283160304E-07, 1.3924671809E-06, -2.9226241894E-07, -8.8717807704E-07, -5.2852167825E-07, 1.0249780149E-07, 1.7552133015E-06, 1.0916907579E-06, -5.2564689320E-07, 1.7301528428E-07, 1.8933772264E-06, 1.9105948694E-06, 7.6963814928E-07, 1.1953530021E-06, 1.0759853641E-06, 2.7957741713E-06, 1.6543663598E-06, 2.4202378617E-06, 3.5266055486E-06, 4.9734692448E-06, 4.0359857734E-06, 3.5752439576E-06, 2.2969406205E-06, 1.9722285852E-06, 1.1705636748E-06, 9.1376271242E-07, 1.8149185212E-06, 6.0421592419E-07, -2.3096146554E-06, -4.1336091945E-06, -1.3254670700E-06, -1.1060260588E-06, 8.8446466224E-07, 6.9498191626E-07, 2.1403105263E-06, 2.0868783156E-06, 4.2813471039E-06, 4.2277256400E-06, 4.5146166038E-06, 3.0983856726E-06, 2.2270325236E-06, 2.5136458340E-06, 4.1593028120E-07, -3.1945345773E-07, 1.2611902945E-06, 2.2967772151E-06, 2.4467009813E-06, 1.2341152703E-06, 1.4519857594E-06, 2.0784971258E-06, 1.1381380468E-06, -1.4291180055E-07, -1.9786573887E-07, 4.9642890902E-07, 8.5003482036E-07, 2.0210126707E-06, 6.7142808182E-07, 2.1828486108E-06, 2.7404938110E-06, 3.2299382358E-06, 2.3568814386E-06, 1.7562299726E-06, 1.7004693914E-06, -6.7148675182E-07, -5.2303890364E-07, 3.4059489297E-08, 1.8235498034E-07, 6.0306012285E-07, 8.1932647019E-07, 6.2679257572E-07, 3.6606299279E-07, -1.8021303252E-06, -1.4499150250E-06, -6.8904535312E-07, 1.7747808431E-06, 3.5573250816E-06, 4.0454978287E-06, 4.6698425465E-06, 2.4330306971E-06, 1.4904517425E-06, 1.7058621448E-06, 1.6487203661E-06, 1.5915108685E-06, 1.7385971141E-06, 2.2943435649E-06, 1.8282066831E-06, 1.4301259306E-06, -1.0797758303E-06, -2.7041677377E-06, -1.7400217295E-06, -2.6607543465E-08, 6.6492628235E-07, 1.2201550101E-06, 5.4914110154E-07, -8.0314630112E-07, -1.7998289270E-07, 9.1996682805E-07, 1.7473705624E-06, 3.2559247115E-06, 4.1513296766E-06, 2.3218298590E-06, 1.7865716598E-06, 1.2512544805E-06, 2.0783007220E-06, 2.7009257858E-06, 3.1872510730E-06, 2.3792179849E-06, 2.8654289227E-06, 1.9210392876E-06, 1.5914048088E-07, 1.0021390378E-07, 3.1371595752E-07, 1.8214650433E-06, 1.4217675463E-06, 4.0193448147E-06, 2.3252391913E-06, 2.0616220189E-06, 9.8049864026E-07, 8.5301839826E-07, 4.5300063567E-07, 3.2541569533E-07, 8.1086892002E-07, -1.3427334744E-07, -3.9825476423E-07, -2.5356098756E-07, 3.6793032540E-07, 1.8749465174E-06, 4.1312439314E-06, 4.7525869100E-06, 3.3983667962E-06, 4.4964589327E-06, 1.3028726623E-06, 1.7196563159E-06, 2.1363931887E-06, 2.0081135695E-06, 9.9421174699E-07, 1.8195330101E-06, 1.8954766474E-06, 1.6988879479E-06, 1.6384962001E-06, 6.2436269300E-07, 1.1088487152E-06, 3.6369225556E-06, 1.6008375029E-06, 1.3358568459E-06, 7.3022587329E-07, 4.6515587391E-07, -7.2441863487E-08, 1.3654299499E-06, 1.3045926027E-06, 1.4480773839E-06, 1.1827915821E-06, 1.2580694861E-06, 3.9004384705E-08, -7.0325443333E-07, 2.5747332076E-07, 8.0943293570E-07, 8.1638268638E-07, 1.0957758146E-06, 1.7157345571E-06, 1.5182001508E-06, 1.7974728326E-06, 7.8240483937E-07, 3.9782407939E-08, -9.0724262482E-07, -6.9624792204E-07, 7.4088785314E-07, 1.8373800376E-06, 1.2989269682E-06, -3.2950161824E-07, -5.9554598479E-07, -1.4065967946E-06, -7.1902031084E-07, 1.3990621034E-06, 1.9503224849E-06, 2.7740312707E-06, 2.5077646975E-06, 2.3777050022E-06, 7.4894442170E-07, 1.3681451929E-06, 4.8864655251E-07, 8.5959737465E-08, 5.0069197316E-07, 7.7914646289E-07, 5.8071940737E-07, 1.7446800073E-06, 1.4099414634E-06, 8.7080597638E-07, -1.4521065304E-07, 8.8237197584E-07, -3.3807483625E-07, 1.4447261138E-07, 1.4444404195E-06, -5.2543761130E-07, 9.3254019801E-08, 1.2568822135E-06, 2.7610832705E-06, 2.0853773914E-06, 1.8183667771E-06, 1.5513236283E-06, 2.1017031456E-06, 2.7201725297E-06, 1.4993409814E-06, 1.3684167014E-06, -3.9744510968E-07, 6.2962874900E-07, 1.3841874779E-06, 2.7518062662E-06, 7.8146128453E-07, 5.8223570200E-07, 2.2903706877E-06, 2.0910864106E-06, 2.6411050398E-06, 1.5561877443E-06, 1.1524536933E-06, 6.8056905572E-07, -1.2218874993E-06, 4.8604669722E-07, -1.7570710856E-06, -1.4797358787E-06, 5.6872098689E-07, 7.0975838032E-07, 1.8044643706E-06, 1.1961151268E-06, 2.0864028180E-06, 1.4098786132E-06, -1.5224701845E-07, 8.0608219201E-07, 6.0632631367E-07, -2.1832844164E-09, -6.5749533183E-08, 1.0287170362E-06, 1.4419472814E-06, 1.1739399460E-06, -3.8839302770E-07, 4.3348710257E-07, -5.1580652081E-07, 1.3278406445E-06, 5.1473854090E-07, 4.2218210693E-08, -1.5202649038E-06, -4.9416996010E-07, 5.3190028401E-07, 8.7673597093E-07, 1.9708796430E-06, 8.8512344259E-07, 7.1828012037E-08, -1.2840090631E-07, -1.4867121701E-06, -5.2892883699E-07, -3.2050086462E-07, -1.1339127106E-06, -1.8792264327E-06, 1.5308255115E-06, 2.7609788644E-06, 2.5605667684E-06, 1.9514053653E-06, 1.2059787878E-06, 2.5722861584E-06, 3.3936015987E-06, 2.2393832306E-06, 3.1287771760E-06, 1.4295465565E-06, 1.9782914941E-06, 1.3008341105E-06, 6.2335652757E-07, 1.3764018671E-06, 4.9451825092E-07, 1.4518848008E-06, 5.0183863862E-07, -1.0762211388E-07, -6.4898222485E-07, 1.8750870975E-06, 2.2873823050E-06, 5.1978140948E-07, -9.7535496730E-07, 3.9057579650E-07, 8.7091232269E-07, 6.7001723308E-07, 1.3546776495E-06, 2.5842876937E-06, 1.1571554871E-06, -3.3811725853E-07, -9.4783440351E-07, -7.4011733937E-07, 1.3749716351E-06, 1.0376827628E-06, 2.2352676685E-07, 8.3989625017E-07, 1.7287308158E-06, 1.1188830666E-06, 3.0465370568E-07, 3.5375136015E-08, -1.6644650394E-06, -2.9887241765E-07, -1.5947799567E-07, 7.9735172951E-07, 1.4135586609E-06, 2.8472021014E-06, 2.9865261541E-06, 2.1721379220E-06, 3.3591510801E-07, 5.4330901516E-07, 8.1880654643E-07, -3.3625539518E-07, -4.6951770669E-07, 6.9150431489E-07, 2.0568729726E-06, 1.5148351693E-06, 2.3352020081E-06, 5.6695059197E-07, -6.5634809613E-07, -1.0851379320E-07, -1.6724501562E-06, -3.7531526405E-07, 4.4495597029E-07, 9.2460642677E-07, 9.2739298537E-07, 4.5331572605E-07, -3.6138207123E-07, -1.3804585265E-06, -1.9908231598E-06, -8.9817599115E-07, 6.7130425938E-07, 9.4646887179E-07, 6.3558526015E-08, 6.6208302036E-08, 7.5005427981E-07, 1.2976435393E-06, 3.1395191605E-06, 2.6652622233E-06, 1.4416588078E-06, 1.0354949938E-06, -2.5625873866E-07, 2.1305134904E-06, -3.8744783907E-07, 2.2814765289E-07, -1.7808765375E-07, -9.2494247903E-07, -2.6936281430E-06, -1.3287556657E-06, -1.3888704573E-06, -1.9893146415E-06, -2.3195569456E-06, -6.3825233348E-07, -3.2822158972E-06, -3.9836667495E-06, -4.9011159599E-06, -3.9451265596E-06, -2.8564200390E-06, -7.9910934695E-07, 1.1824520409E-06, 2.7403102965E-06, 1.5765656233E-06, -6.3725393538E-07, -9.0647089022E-07, 6.9944897428E-07, 1.8825961384E-06, 2.5735115113E-06, 9.6171300269E-07, 5.9896852195E-07, 9.2642359340E-08, -6.9650232474E-08, 1.8531797618E-06, 1.1601353654E-08, 5.3396239582E-07, 9.1338772461E-07, 1.1499739912E-06, 3.6115577197E-06, 3.3538651841E-06, 1.3519550384E-06, 1.9235371009E-06, 2.4920406955E-06, 1.3862351460E-06, 2.7148997548E-06, 2.5087093631E-06, 1.6730288876E-06, 9.7397190752E-07, 3.4757320017E-06, 6.0446227491E-06, 5.9647857702E-06, 3.3059338215E-06, 2.6642971530E-06, 2.9953560142E-06, 2.9760206548E-06, 2.4670806844E-06, 3.4881094731E-06, 3.7410868719E-06, 2.9475026923E-06, 3.4136174552E-07, 4.4926370682E-07, 7.6424670484E-07, 2.6791370765E-06, 2.6423779731E-06, 3.3699395463E-06, 3.1208839343E-06, 2.9398513546E-06, 1.5037181063E-06, 2.2944800108E-06, 4.4068628892E-06, 4.7071525630E-06, 3.0560948266E-06, 3.8409932838E-06, 2.8138977586E-06, 3.8049777002E-06, 2.7055525578E-06, 2.9976217807E-06, 1.4081687785E-06, 3.3690689242E-06, 4.2841586073E-06, 4.8498779805E-06, 4.6483911961E-06, 5.0028824067E-06, 4.9384157284E-06, 4.8032139860E-06, 3.6919983959E-06, 3.3457560736E-06, 2.5110091347E-06, 1.2574176949E-06, 2.3705628268E-06, 6.0593805059E-06, 6.4742205432E-06, 5.7042586936E-06, 7.2314737118E-06, 7.3650143526E-06, 5.6870592679E-06, 5.0528351198E-06, 5.1141567713E-06, 4.9657290206E-06, 6.8360266660E-06, 6.4074245056E-06, 6.0476872556E-06, 5.5478984046E-06, 4.6991504953E-06, 4.1978560465E-06, 4.1832985768E-06, 4.0287416046E-06, 3.0378076487E-06, 2.3247332276E-06, 2.2377258599E-06, 2.7071590642E-06, 2.1313481303E-06, 4.3404489170E-06, 3.2758787304E-06, 3.7427388759E-06, 5.5321236585E-06, 5.3710053835E-06, 4.0950713561E-06, 5.1166358813E-06, 6.5554612644E-06, 6.7402078101E-06, 4.2084903992E-06, 4.0439466731E-06, 2.6950010480E-06, 3.8525289400E-06, 4.9398957654E-06, 4.8428859405E-06, 5.0239268813E-06, 4.7866370042E-06, 4.4095807252E-06, 4.1016834440E-06, 4.2807867134E-06, 5.1558138604E-06, 5.1947131154E-06, 5.8599097092E-06, 5.8282688724E-06, 4.8908798357E-06, 5.3458358296E-06, 6.2878340848E-06, 4.7920488320E-06, 3.5047622749E-06, 2.8438162341E-06, 4.8287472438E-06, 4.1669968312E-06, 4.4797985237E-06, 1.0317058486E-06, 3.1543503333E-06, 2.9088915052E-06, 3.9165628914E-06, 5.0631460195E-06, 4.9558663309E-06, 4.3607577507E-06, 3.5563810108E-06, 2.4731018430E-06, 2.7822609790E-06, 3.1607221505E-06, 2.2157100892E-06, 3.1506235268E-06, 4.5726871951E-06, 6.0640718255E-06, 5.4659729429E-06, 6.3299834892E-06, 7.3329683036E-06, 6.5946722258E-06, 5.8560800950E-06, 6.0225007509E-06, 6.4671870328E-06, 6.9812207802E-06, 7.2860638326E-06, 7.5906280294E-06, 5.3182672100E-06, 6.0401152140E-06, 4.9510818536E-06, 4.8367304870E-06, 3.2596990965E-06, 5.3732766534E-06, 7.0687651287E-06, 5.2124124936E-06, 4.4003957191E-06, 5.5380247765E-06, 5.5611866368E-06, 4.2609672712E-06, 4.9800416507E-06, 6.3952717466E-06, 7.8102715298E-06, 7.3447878946E-06, 6.1130504509E-06, 4.3936167437E-06, 4.8327713179E-06, 6.3162927183E-06, 4.9443994897E-06, 3.5026541771E-06, 2.4785323253E-06, 2.9166214792E-06, 2.0313641837E-06, 3.7225469835E-06, 4.8564174236E-06, 4.5276710413E-06, 5.3825921855E-06, 7.0033490079E-06, 7.2311346512E-06, 5.5088392579E-06, 4.9702199704E-06, 5.5456399314E-06, 4.2406242833E-06, 3.4925401687E-06, 2.8835557302E-06, 2.0654781103E-06, 1.3300245143E-07, 7.7699789624E-07, 1.2815455318E-06, 3.0394271778E-06, 4.1703915736E-06, 4.1869674586E-06, 2.2534885719E-06, 3.1054086529E-06, 2.7733064409E-06, 5.4355236753E-06, 4.5459960952E-06, 4.7705384400E-06, 4.6467334490E-06, 6.1941138615E-06, 6.0700094144E-06, 4.4833387437E-06, 4.6374973525E-06, 5.6271797347E-06, 4.1793543841E-06, 3.9152507948E-06, 3.9992024604E-06, 4.5704888750E-06, 3.8881365324E-06, 4.4591509265E-06, 3.4283345511E-06, 1.7706349002E-06, 1.6448624676E-06, 3.6081307224E-06, 5.5016319762E-06, 6.2807824790E-06, 5.0402764807E-06, 3.7300072312E-06, 2.0714199803E-06, 1.3180179779E-06, 7.3834739157E-09, 1.8303847182E-06, 3.7229069606E-06, 3.4565034511E-06, 4.0256524394E-06, 4.8036041756E-06, 7.0438619094E-06, 7.0555598907E-06, 7.1367863694E-06, 7.2179035937E-06, 5.4186597253E-06, 4.8728108016E-06, 6.0678308503E-06, 3.5022398991E-06, 3.6524659759E-06, 1.8526971082E-06, 2.7687473240E-06, 3.4757796508E-06, 4.7398241165E-06, 4.1931557488E-06, 3.0196385753E-06, 1.9853026240E-06, 2.9704019224E-06, 2.0055144984E-06, 6.2269937958E-07, -4.8165240620E-07, 1.6172981475E-06, 1.2787929227E-06, 3.0293677563E-06, 3.3174344851E-06, 2.5608279457E-06, 3.4058309751E-06, 5.0864154099E-06, 5.7919680871E-06, 6.1492408433E-06, 6.3671515154E-06, 6.5849789401E-06, 5.2706659543E-06, 4.5830223946E-06, 5.2184390980E-06, 4.3913569011E-06, 2.8678056408E-06, 3.5726241525E-06, 3.2327802156E-06, 6.3748095343E-06, 5.4080618073E-06, 5.7643797335E-06, 4.1707310115E-06, 4.5965393403E-06, 4.8133574186E-06, 4.8211859452E-06, 5.3164156190E-06, 5.3241011386E-06, 5.1227992029E-06, 3.8768425108E-06, 4.0932357609E-06, 3.7524486521E-06, 3.4812328832E-06, 5.6473151530E-06, 6.7687461473E-06, 6.7062484813E-06, 6.6436857465E-06, 6.0239465345E-06, 3.1756954367E-06, 1.7898000447E-06, 2.8412069499E-06, 1.7337427440E-06, 9.0477301828E-07, 7.0249336441E-07, 2.9123637385E-07, 2.9775363810E-07, 1.4880747487E-06, 3.0265322971E-06, 3.3114298748E-06, 3.7355480734E-06, 3.3935804833E-06, 2.2158886546E-06, 1.5256140871E-06, 3.6208442774E-06, 3.2090157222E-06, 2.0311039178E-06, 1.5495283610E-06, 9.2862154826E-07, 1.2826079762E-06, -7.3118385874E-07, -9.8745459845E-08, 4.6400266939E-07, 2.0712850255E-06, 2.9124881052E-06, 4.7285874049E-06, 4.6643844212E-06, 5.5750786507E-06, 5.3715005787E-06, 3.4268996341E-06, 3.0143092285E-06, 1.0696137731E-06, 3.0873467934E-07, 8.0131135865E-07, 3.2437342224E-06, 4.5022486820E-06, 4.7857721489E-06, 4.8603340344E-06, 2.9153207499E-06, 3.6861847069E-06, 3.5516983167E-06, 2.6511399907E-06, 3.2129581403E-06, 4.4711241770E-06, 5.9381655121E-06, 6.0820245569E-06, 3.0920877230E-06, 1.5645294217E-06, 2.7528520643E-06, 1.8519650623E-06, 1.1599558083E-06, -5.0703835900E-07, -7.1165210001E-07, 1.9791988465E-07, 2.1520389544E-06, 3.4097304288E-06, 2.0907416371E-06, 2.0948569089E-06, 1.7507405735E-06, 1.8940609604E-06, 2.1069843991E-06, 2.3895112190E-06, 1.0006667494E-06, 1.2134843200E-06, 2.4708522600E-06, 4.5638538989E-06, 2.5481205662E-06, 1.1451959125E-07, 1.1825030357E-07, 2.2807570326E-06, 3.2593681077E-06, 1.8005818580E-06, 6.8995859113E-07, 5.5424957828E-07, 7.6670408721E-07, 4.2201538572E-07, 9.8260274156E-07, 2.5877444225E-06, 2.0340466964E-06, 7.1428983090E-07, 9.9620009386E-07, 3.0190557530E-06, 2.7437950762E-06, 3.5827293311E-06, 2.1931867856E-06, 3.3106197074E-06, 4.4280243643E-06, 5.4757620240E-06, 3.9468289544E-06, 4.6463164232E-06, 2.4905776982E-06, 9.6156304716E-07, 1.8698867379E-06, 4.1013250370E-06, 2.5722311872E-06, 2.6448084062E-06, 3.8315839108E-06, 6.5503919175E-06, 3.9069716431E-06, 3.4919743044E-06, 5.6994811799E-07, -1.2365469941E-07, 2.1771950689E-06, 2.2495726396E-06, 9.9878522949E-07, 4.4436405517E-07, 9.8836333359E-08, 5.1931428077E-07, 4.5229611411E-07, -3.8077394994E-07, 1.0927430534E-07, -3.7564590335E-07, -1.3480613593E-06, -2.1812208459E-06, -9.9487114630E-07, 2.0717099540E-06, 2.9794606465E-06, 2.8426051082E-06, 2.0789775162E-06, 4.1002204719E-07, 2.7310387820E-07, 6.2363818610E-07, 6.8845147795E-08, 8.3717294018E-07, -2.0513325984E-07, 9.8098872463E-07, 6.3503307050E-07, 9.1580895466E-07, 2.1715115540E-06, 3.1486390455E-06, 4.6132206059E-06, 2.3869164123E-06, 7.8734464148E-07, -1.8549452965E-07, 1.4182910758E-06, 5.8469363481E-07, 1.9795263242E-06, 3.6528973175E-06, 3.7941927677E-06, 4.5622218208E-06, 3.8678146222E-06, 2.4770003178E-06, 1.8521980533E-06, 2.7594369743E-06, 1.4382112266E-06, 2.5624695578E-07, 9.5451930765E-07, -7.1495057211E-07, -7.1310053781E-07, -7.1126644374E-07, -7.7908714420E-07, -1.5053349348E-07, 7.5656065411E-07, 1.0368884443E-06, 8.2972802274E-07, -4.1984648131E-09, -7.6850087268E-07, -7.6005515161E-10, 2.8561361335E-06, 2.7185408329E-06, 2.7202091629E-06, 1.8165562391E-06, 3.6288101770E-06, 4.9535770922E-06, 4.9551891003E-06, 4.3300363169E-06, 2.4513678575E-06, 2.2440218378E-06, 7.8316037329E-07, 2.3167625027E-06, 1.9004594961E-06, 9.1363391353E-08, 8.5889730420E-07, 1.2085843502E-06, -3.2199435517E-07, 5.8477930379E-07, 1.2826234426E-06, 1.9108161769E-06, -8.0367537782E-07, -1.9861211059E-06, -1.7061598918E-06, -1.5654886199E-06, 5.2506282530E-07, 3.1751555947E-07, 1.0849026982E-06, 3.8985935695E-07, -1.6591734858E-07, -9.3062230285E-07, -2.3220893919E-06, -1.5547585178E-06, -5.7852159196E-07, -4.3796350570E-07, 3.2933476954E-07, 3.3059338235E-07, 6.1039740130E-07, 4.0271791499E-07, 3.3984220120E-06, 3.8174727809E-06, 4.0972353104E-06, 2.8449296889E-06, 6.1766800505E-07, -7.9066525073E-09, 4.1109380478E-07, -1.5376415345E-06, -1.1882995818E-06, 2.5037047516E-06, 1.8084425541E-06, 2.3666729144E-06, 1.9499479211E-06, 9.7610166270E-07, -2.7630977214E-07, 4.9080070515E-07, 1.1882631831E-06, 2.5821067504E-06, 1.2600204956E-06, -1.1066594927E-06, -1.3067612601E-07, -1.2964731584E-07, 1.1248750167E-06, 1.1258869372E-06, 1.8232805100E-06, 2.1724707995E-06, 2.5216528702E-06, 1.7566027865E-06, 7.1298861276E-07, 8.5322941349E-07, 1.2720182531E-06, 9.9440919596E-07, -1.6163096443E-06, -1.7915253683E-06, 2.8144989214E-07, 3.0281128216E-06, 4.7370344961E-06, 3.4227641646E-06, 4.3585657696E-06, 5.0796215418E-06, 4.4907012448E-06, 4.0301856016E-06, 2.3288743055E-06, 2.6055931396E-06, 2.1899943391E-06, 3.8214916396E-06, 2.9808266801E-06, 3.4345622701E-06, 3.9503202593E-06, 3.4327019974E-06, 3.9362787565E-06, 5.5297367130E-06, 3.0089482029E-06, 2.1947355619E-06, 2.3340571900E-06, 4.9336081910E-06, 3.4881251298E-06, 3.4075345585E-06, 3.5963720420E-06, 5.1504371502E-06, 5.8100664799E-06, 6.5341162537E-06, 6.0216225977E-06, 6.9433634862E-06, 7.6559726826E-06, 7.2008825234E-06, 7.1531953449E-06, 5.0477905857E-06, 4.5826160618E-06, 5.2099266760E-06, 5.7656873309E-06, 6.1130320827E-06, 5.7727065059E-06, 4.5393612861E-06, 5.0837170500E-06, 6.9265044245E-06, 8.6982525608E-06, 5.8794613028E-06, 5.7973568743E-06, 4.5487654910E-06, 4.3935233685E-06, 3.8936447484E-06, 5.5144157713E-06, 5.4895905011E-06, 5.7365930011E-06, 5.9815873323E-06, 6.5670058865E-06, 6.5341965503E-06, 6.0201764572E-06, 7.3532327408E-06, 7.7943125178E-06, 7.8227812544E-06, 6.9593414199E-06, 7.3268751663E-06, 6.1177946455E-06, 6.1397618956E-06, 5.9547749141E-06, 6.0422066642E-06, 6.8814397938E-06, 5.8703369507E-06, 6.1589206494E-06, 5.7613610316E-06, 5.0885561962E-06, 4.8937941753E-06, 4.6292730010E-06, 5.3906707056E-06, 3.6171563212E-06, 3.8283148799E-06, 4.6546072501E-06, 5.4112775302E-06, 4.3863985109E-06, 4.1821629126E-06, 4.7985834558E-06, 5.8933028609E-06, 5.6174338481E-06, 5.9568691379E-06, 6.0898713071E-06, 6.1534406269E-06, 6.2845455097E-06, 8.0582043150E-06, 8.1189954023E-06, 7.7680271311E-06, 7.2792278610E-06, 6.6526259515E-06, 7.4632295986E-06, 7.3827945013E-06, 8.2602544110E-06, 7.4249730276E-06, 6.7258640506E-06, 7.6694611798E-06, 6.6949281151E-06, 5.8566085561E-06, 6.5925661438E-06, 7.9441236650E-06, 7.9254132618E-06, 5.7831970603E-06, 5.8316071869E-06, 6.8380457676E-06, 6.7481949288E-06, 5.6305267966E-06, 6.2241834187E-06, 3.3248467562E-06, 3.9172909767E-06, 3.4819682294E-06, 3.5938866639E-06, 5.3487084296E-06, 5.3225336760E-06, 4.4055885524E-06, 5.2685331269E-06, 6.1994273943E-06, 4.1852225908E-06, 3.4031119367E-06, 3.9215786526E-06, 4.4395429587E-06, 5.4363620754E-06, 5.6794362230E-06, 4.4839906221E-06, 5.1369884928E-06, 6.1319170448E-06, 6.8524818008E-06, 6.2715102002E-06, 5.0053185867E-06, 5.7245613042E-06, 3.8412046964E-06, 4.1487301071E-06, 4.9351888801E-06, 5.9951513593E-06, 4.6579738884E-06, 6.5388778112E-06, 5.9541763005E-06, 6.2593078860E-06, 5.1260189463E-06, 5.9097458601E-06, 6.4876710061E-06, 5.3532777630E-06, 6.8206995093E-06, 5.4116856238E-06, 4.6871074852E-06, 5.4687084720E-06, 3.0999769075E-06, 3.1276515718E-06, 1.9223919367E-06, 2.3602874707E-06, 2.9348226423E-06, 4.5362179199E-06, 6.0688257720E-06, 6.7793916670E-06, 5.8461810734E-06, 6.0768054596E-06, 4.8006182844E-06, 3.9350136461E-06, 4.5071691859E-06, 4.8736105154E-06, 5.8560802462E-06, 5.9480609901E-06, 4.8756443586E-06, 3.4605709634E-06, 4.0311054162E-06, 3.0948613285E-06, 3.0485803120E-06, 3.5498738609E-06, 4.2563551242E-06, 5.1680271875E-06, 3.4772861366E-06, 2.1286980570E-06, 2.1494390344E-06, 3.1971211545E-06, 1.8478375029E-06, 3.3744121653E-06, 1.9562022273E-06, 1.9073367304E-06, 3.0908613865E-06, 5.0959107584E-06, 5.3887934081E-06, 6.3662488977E-06, 8.0967547894E-06, 8.7314146452E-06, 7.9278230272E-06, 8.1512024976E-06, 8.1689556184E-06, 9.0082549516E-06, 9.5049730595E-06, 6.9884585041E-06, 6.8684948461E-06, 7.5700864937E-06, 6.3541075674E-06, 5.2064261558E-06, 4.8803063473E-06, 5.8550972305E-06, 6.4873238938E-06, 4.7226414257E-06, 5.2860499149E-06, 4.6851614496E-06, 5.7960631186E-06, 5.5372390103E-06, 5.4152092131E-06, 4.5397608157E-06, 4.0065478815E-06, 4.8427432376E-06, 2.8027015798E-06, 1.8581596024E-06, 2.7623800000E-06, 6.5425364670E-06, 5.8029856978E-06, 4.8578553870E-06, 2.2691052289E-06, 2.2139079180E-06, 1.6792231488E-06, 1.7607046157E-06, 2.7322670132E-06, 3.1558690353E-06, 3.3739042986E-06, 4.2081122394E-06, 3.6041472685E-06, 3.2739667969E-06, 1.6425732355E-06, 1.2436609953E-06, 2.0087554873E-06, 8.5633612224E-07, 2.3744463112E-06, 4.2348264651E-06, 5.1363939948E-06, 5.4900183113E-06, 5.7065708255E-06, 4.2795309382E-06, 4.5643339260E-06, 5.8077199812E-06, 6.9825172949E-06, 8.0887260578E-06, 8.2361324611E-06, 7.0138666955E-06, 5.4491039520E-06, 4.7744534215E-06, 4.4420902950E-06, 2.8085387633E-06, 3.0922750174E-06, 3.9922162483E-06, 3.4540156468E-06, 3.0526743599E-06, 2.5827584096E-06, 3.2083977954E-06, 4.3132895169E-06, 3.0213505739E-06, 1.7293189659E-06, 1.9437166925E-06, 2.4319357534E-06, 4.0157171483E-06, 3.7504978766E-06, 5.4025809382E-06, 4.1100143325E-06, 3.2967100592E-06, 3.5104931126E-06, 3.7241933977E-06, 3.8008537464E-06, 4.5622139888E-06, 3.9539299547E-06, 4.3727374740E-06, 3.7642923767E-06, 2.6764214926E-06, 2.6156446517E-06, 3.3080516838E-06, 4.4797304189E-06, 5.0350296868E-06, 4.9054713175E-06, 3.5432321409E-06, 1.5646159868E-06, 7.5005668524E-07, 2.0582510661E-06, 1.9283329592E-06, 1.8668231853E-06, 4.2019814698E-06, 4.1403344816E-06, 4.0786198887E-06, 1.6885783591E-06, 2.5169475611E-06, 2.7289461627E-06, 3.0093588319E-06, 3.2897072369E-06, 3.4330350457E-06, 4.2610829265E-06, 4.5412425473E-06, 4.4789495762E-06, 3.2524656814E-06, 4.0117895308E-06, 3.6754027926E-06, 2.1063491350E-06, 3.0709312258E-06, 2.8028477222E-06, 8.9123020311E-07, 1.1023812149E-06, 8.0868303524E-08, 3.3734270151E-06, 4.8854978958E-06, 4.4801234915E-06, 3.0475233484E-06, 2.5735640126E-06, 3.8799860301E-06, 3.3374449470E-06, 2.7263733093E-06, 2.9369896632E-06, 3.0105985547E-06, 2.6732885299E-06, 3.4315801349E-06, 2.2724329157E-06, 2.8936714184E-06, 2.5561671751E-06, 3.0403546513E-06, 3.6614512928E-06, 4.5564135451E-06, 5.4513308539E-06, 5.3875076649E-06, 4.9127694237E-06, 3.2738575759E-06, 3.1414225673E-06, 3.0774218435E-06, 2.0577085003E-07, 2.1016203268E-07, 1.0362488371E-06, 3.3003367088E-06, 3.5100370936E-06, 2.8294794371E-06, 1.0532291850E-06, 2.2215027828E-06, 3.4582146608E-06, 3.5307571976E-06, 3.0554357611E-06, 2.3746397195E-06, 3.1318494406E-06, 3.7520642924E-06, 2.3178946430E-06, 3.0065128604E-06, 3.7635733126E-06, 3.6303793674E-06, 4.1563039306E-07, 8.4527574100E-09, 2.3403678285E-06, 1.5907289743E-06, 2.2790975628E-06, 4.4054749621E-06, 3.8611675400E-06, 2.4266096646E-06, 2.1561496870E-06, 2.4334809191E-06, 2.8477376673E-06, 2.0293532380E-06, 2.5120239377E-06, 6.6924860725E-06, 6.3533539491E-06, 4.3022368736E-06, 1.1554371526E-06, 6.1078009234E-07, 1.4356559993E-06, 2.7398501797E-06, 2.6744509402E-06, 3.6361955869E-06, 3.6392164263E-06, 4.1900347649E-06, 4.6038679089E-06, 4.8807171644E-06, 3.4455828168E-06, 2.1473781168E-06, 2.3556673116E-06, 2.2215386486E-06, 1.9504263752E-06, 3.1858087386E-06, 3.0515999863E-06, 4.3554083655E-06, 4.8374531236E-06, 4.6346895079E-06, 4.1579877658E-06, 3.4758271446E-06, 1.4925558917E-06, 1.4266502543E-06, 3.9628924798E-06, 3.4860688157E-06, 3.7624825091E-06, 5.1345238074E-06, 3.6304579580E-06, 1.0991952083E-06, 6.9073480546E-07, 2.0626849969E-06, 1.9965710300E-06, 6.9782715196E-07, 3.2338415920E-06, 4.0578785452E-06, 4.1286342031E-06, 3.3776307571E-06, 3.5168223983E-06, 3.4505593183E-06, 3.1788407084E-06, 1.7429727600E-06, 3.3201256644E-06, 1.6787826130E-06, 1.4754617971E-06, 3.6003814081E-06, 3.5339776375E-06, 2.3719046764E-06, 2.5111691642E-07, 4.5856994878E-07, 2.5513476487E-07, 2.5853747561E-06, 2.3134257137E-06, 1.0142846292E-06, 2.6295089393E-07, 1.4289900992E-06, 3.4852280364E-06, 1.6381906968E-06, 1.0922232599E-06, 2.0527588794E-06, 1.5067567057E-06, 3.0835628891E-06, 3.9070915799E-06, 4.3197359284E-06, 4.4584500851E-06, 4.3917142002E-06, 3.8456144241E-06, 3.1625429071E-06, 2.4109777995E-06, 1.3170052518E-06, -1.2835022859E-06, -1.2818532631E-06, 2.9477967048E-07, 2.5561796652E-06, 3.5849571714E-06, 3.0387221393E-06, 3.7935577194E-06, 3.1788150620E-06, 1.9477543174E-06, -1.7353666410E-07, 5.1276516790E-07, -1.1976844363E-06, -1.1961920348E-06, -3.0449620682E-07, -4.3998688459E-07, 4.0813199502E-08, 2.0965995131E-06, 3.6045462239E-06, 1.2777011494E-06, -2.9589629255E-07, -7.0537688441E-07, 1.2818673915E-06, 3.1321433528E-06, 2.2432781172E-06, 2.3815748023E-06, 2.2459455258E-06, 1.7679144052E-06, 6.7356655838E-07, 2.6399010285E-07, 1.4978791563E-06, 4.5121898363E-06, 3.8286662607E-06, 2.5973055469E-06, 3.2833238127E-06, 4.0378101758E-06, 4.5183717537E-06, 3.2184896583E-06, 2.0555532820E-06, 2.0782492047E-07, -1.1605669344E-06, -1.5017962902E-06, 3.4826784546E-07, 1.8559304648E-06, 3.0211912602E-06, -5.9205975917E-08, 1.4734064868E-07, 1.8603984264E-06, 2.3408380495E-06, 2.4104001103E-06, 2.0690817013E-06, 1.3168859147E-06, 1.7288098429E-06, 7.7116057820E-07, -3.2345368705E-07, -5.9633896051E-07, -1.0061895498E-06, -2.0323525627E-06, -2.0313514068E-06, -1.0031859897E-06, -1.8924189192E-06, -1.9599214064E-06, -1.7535192746E-06, -2.4604014945E-07, 2.5625171434E-06, 3.3851985784E-06, 4.5502619299E-06, 5.2359702721E-06, 2.9086296796E-06, 3.7312792265E-06, 1.1984879873E-06, 1.1993830364E-06, 2.0220084480E-06, 2.3652792967E-06, 1.4759356566E-06, 4.4962700215E-07, -4.3973215774E-07, -2.8012552249E-08, 1.9586986965E-06, 1.1377967283E-06, 2.4840851758E-07, -4.3555162683E-07, 1.3456968659E-06, 2.7845481737E-06, 2.7168707364E-06, 2.9231006247E-06, 3.1293238961E-06, 5.1159736047E-06, 3.8156628046E-06, 1.8305635500E-06, 1.5574128949E-06, 3.0646888934E-06, 2.5176135998E-06, 2.3129220681E-06, 1.2864863525E-06, 2.1774345071E-06, 1.0825075860E-06, 4.3653532866E-07, 1.5752350369E-07, 7.6448914109E-07, -1.7504908003E-07, 3.9896389056E-07, 2.1396964009E-06, 3.1012015735E-06, 3.2841517926E-06, 4.4273654511E-06, 2.8465699476E-06, 3.3396429501E-06, 3.4737074621E-06, 2.6235612914E-06, 2.9450060099E-06, 3.0479587945E-06, 2.8633359013E-06, 3.7125658640E-06, 5.4569654451E-06, 5.8025214785E-06, 5.8620100727E-06, 8.2082337047E-06, 7.8355719826E-06, 7.4558631116E-06, 7.9037019932E-06, 7.4411743098E-06, 6.1380751110E-06, 6.5671174446E-06, 7.1989377913E-06, 5.6698390698E-06, 6.3602238562E-06, 6.4892097810E-06, 6.9608340642E-06, 7.9143124564E-06, 8.6545598583E-06, 8.4169461448E-06, 6.8539883174E-06, 7.6507329615E-06, 8.1653151872E-06, 6.5206567118E-06, 5.5674721139E-06, 5.5144559060E-06, 7.1960503136E-06, 8.7351557675E-06, 8.3937189063E-06, 8.2576263647E-06, 8.8136809843E-06, 7.9066441298E-06, 6.6490350031E-06, 6.7791028061E-06, 6.8368766529E-06, 7.2396096100E-06, 7.2920948990E-06, 7.2029948371E-06, 8.7105417417E-06, 9.4509076533E-06, 7.6859992174E-06, 7.9350881716E-06, 7.5562360742E-06, 6.4799844839E-06, 6.3055044002E-06, 2.9307720761E-06, 4.4904613709E-06, 4.3795961252E-06, 3.4325501801E-06, 4.9171256043E-06, 5.8437269130E-06, 7.5333966082E-06, 8.9432771923E-06, 7.3618911597E-06, 8.0732693929E-06, 9.1307391776E-06, 7.0579773126E-06, 4.1494205969E-06, 5.4110057572E-06, 6.9492870380E-06, 6.2613356345E-06, 6.3368225545E-06, 5.2290388060E-06, 7.1096053970E-06, 7.8069633353E-06, 5.7915536290E-06, 5.8607467531E-06, 6.6935594243E-06, 6.0651767422E-06, 7.5214638004E-06, 6.9603856922E-06, 8.8316675111E-06, 7.7817843505E-06, 6.8004112954E-06, 5.3313617886E-06, 7.1290756902E-06, 7.5353123874E-06, 7.8015512677E-06, 8.6926117185E-06, 9.1656131270E-06, 8.5253048808E-06, 8.0927162313E-06, 8.7021944070E-06, 6.5992950842E-06, 6.6509138999E-06, 5.8674164909E-06, 5.6393584939E-06, 8.0525355457E-06, 7.7534332832E-06, 7.8012269638E-06, 4.7891303023E-06, 6.1564966218E-06, 4.7421162456E-06, 4.2308954966E-06, 5.3876456979E-06, 4.5274721726E-06, 3.5276042206E-06, 4.1957461726E-06, 5.3499570597E-06, 8.1721978220E-06, 6.4050923995E-06, 3.5945047322E-06, 3.7034587604E-06, 4.9938014239E-06, 6.2835925030E-06, 4.2355665209E-06, 5.7328624080E-06, 4.7962040910E-06, 6.2229334964E-06, 4.3814085511E-06, 4.6947091818E-06, 4.5208363150E-06, 3.8597968776E-06, 3.3373407962E-06, 3.0925288092E-06, 4.7940054472E-06, 6.7731407638E-06, 5.7621948113E-06, 3.9860236421E-06, 3.6694833085E-06, 3.5611038629E-06, 5.6076353577E-06, 5.9851248453E-06, 7.7527433779E-06, 7.0865508561E-06, 5.7942263791E-06, 6.3092157825E-06, 3.9037169019E-06, 2.8188545726E-06, 3.1241616304E-06, 1.4128429106E-06, 3.1079762488E-06, 2.5779144804E-06, 4.4809444411E-06, 3.7416328006E-06, 3.3496265059E-06, 3.8611423073E-06, 4.3028129551E-06, 3.2841131996E-06, 4.7680597910E-06, 5.2783314795E-06, 5.2320920155E-06, 4.2121851492E-06, 2.5662506308E-06, 3.4229870532E-06, 5.9453444159E-07, 2.6876069332E-07, -5.7286294225E-08, 1.9107743763E-06, 2.0013516023E-06, 3.3431462812E-06, 3.5027313102E-06, 4.4963815868E-06, 3.9601970082E-06, 4.9533513429E-06, 3.8604659741E-06, 2.8368682135E-06, 4.0378833727E-06, 4.1957687634E-06, 3.4495776972E-06, 3.6070074857E-06, 2.2346284407E-06, 2.6001948737E-06, 4.2170230960E-06, 4.9993192664E-06, 5.1556661647E-06, 5.5203855127E-06, 3.5209970319E-06, 3.6072024441E-06, 5.2227934711E-06, 6.2819758346E-06, 5.5332722562E-06, 4.0195864577E-06, 3.2009781609E-06, 3.9117670874E-06, 1.4936789590E-06, 2.8298434973E-06, 3.6791453154E-06, 3.6244256650E-06, 4.8905377229E-06, 6.0869526657E-06, 6.3098256704E-06, 6.0458469136E-06, 6.1293365721E-06, 5.4478708228E-06, 3.3757148423E-06, 2.2072448074E-06, 3.3329958950E-06, 6.5443902817E-06, 4.8192471444E-06, 4.7625915995E-06, 4.0800504941E-06, 2.5630435997E-06, 1.3239996880E-06, 5.7149953042E-07, 1.8351288987E-06, 2.8900405643E-06, 5.8915572989E-06, 7.0852498741E-06, 5.1501130614E-06, 3.2843716325E-06, 3.3652443589E-06, 4.7669930122E-06, 6.2381423307E-06, 5.9014758511E-06, 5.4951600346E-06, 5.1582483423E-06, 5.3774282353E-06, 6.0136491745E-06, 4.2858506211E-06, 4.3656250361E-06, 4.7233918806E-06, 2.8561956157E-06, 2.4489457024E-06, 3.0149566019E-06, 3.3027527750E-06, 3.1732826644E-06, 2.8351265569E-06, 2.0101806625E-06, 2.5756611908E-06, 1.4028763511E-06, 1.6205203530E-06, 1.4209064059E-06, 3.5155687192E-06, 3.9414975025E-06, 3.8806429651E-06, 4.7930653166E-06, 2.9938587663E-06, 1.4031385238E-06, 2.3848097889E-06, 3.9921296483E-06, 4.5564611153E-06, 5.8854972022E-06, 4.7114889211E-06, 3.6764482841E-06, 3.8928003034E-06, 9.8037199101E-07, 2.6566143592E-06, 4.4022994201E-06, 2.4629971858E-06, 3.1656226684E-06, 2.9643248802E-06, 3.4582138278E-06, 2.2833874247E-06, 1.1780115057E-06, 2.1583569031E-06, 4.6682114492E-06, 3.6321379764E-06, 2.7350443169E-06, 2.8807763030E-06, 4.6255477672E-06, 5.1187695416E-06, 4.1518654586E-06, 4.0887363506E-06, 2.6350080498E-06, 8.3358038860E-07, 2.6474661992E-06, 3.7660213141E-06, 5.6493035654E-06, 3.6390347855E-06, 3.2973397627E-06, 3.3032151752E-06, 2.4747106843E-06, 1.2289849510E-06, 1.7908886367E-06, 2.0051004023E-06, 3.6097839091E-06, 3.8934038182E-06, 1.7435367908E-06, 2.4442004881E-06, 2.7276485712E-06, 3.2196207013E-06, 5.4497015395E-06, 4.6205587470E-06, 5.8076329850E-06, 5.8127019146E-06, 4.9138731970E-06, 4.0845184841E-06, 3.1160593477E-06, 2.4256563180E-06, 1.5266219251E-06, -1.1801515010E-06, -1.5925984304E-06, -8.2314263302E-07, 1.4063223211E-06, 1.8280500619E-06, 2.2497296194E-06, 3.3666285237E-06, 3.0929503047E-06, 2.5411194923E-06, 2.1282976167E-06, 4.6395320770E-07, 1.5110997954E-06, -3.6191219022E-07, 4.0704208013E-07, 6.1974179770E-07, -9.0576508404E-07, 1.5317006036E-06, -1.1063136506E-06, -5.9461138132E-08, -1.1679721302E-06, -1.9072629807E-07, 1.2731662869E-06, 3.2932318533E-06, 2.5321968300E-06, 1.6320696455E-06, 2.1919637287E-06, 5.2697050806E-07, 1.1341761242E-07, 1.3684658704E-06, 6.7673531074E-07, 3.3260116209E-07, 7.5322384256E-07, 3.3949371865E-07, 3.2630031411E-06, 8.3293446079E-07, -6.9332347159E-07, -2.0334440522E-07, 2.8660131067E-07, 5.6793572687E-07, 1.5445017942E-06, 1.8952978633E-06, 2.3851152851E-06, 3.2225344103E-06, 3.1560765898E-06, 4.1324871742E-06, 3.9964425144E-06, 3.8603679612E-06, 4.5585818653E-06, 2.4757055760E-06, 1.4356974106E-06, 3.1767236570E-06, 2.0671326017E-06, 2.2785185313E-06, 1.9336637325E-06, 5.4588349184E-07, 2.5648770959E-06, 2.7761538314E-06, 1.8054519848E-06, 3.4767328428E-06, 3.7574576919E-06, 3.6209958189E-06, 3.5581451026E-07, -5.4548884739E-07, -7.5155184744E-07, -1.2332076327E-07, 7.1346565170E-07, 2.1064404509E-06, 4.2641834708E-06, 4.1275269413E-06, 2.9479490924E-06, 1.6292941540E-06, 1.0754093561E-06, 9.3866092870E-07, 2.6095801017E-06, 3.6547381051E-06, 3.4483971689E-06, 4.2849315229E-06, 2.4794363972E-06, 2.3425570218E-06, 1.9970766265E-06, 8.8678344135E-07, 7.4984169632E-07, 2.7682026214E-06, 1.6578484464E-06, 2.2161124015E-06, 4.7997971649E-07, -2.8280057860E-07, 7.6209034618E-07, 1.2507501208E-06, 2.2260771745E-06, 2.0194337122E-06, 7.4608410182E-08, -6.8828363744E-07, -3.3876881634E-07, 9.1457320779E-07, 2.3764777892E-06, 3.6203690228E-07, 1.6153267212E-06, 4.0505504204E-06, 3.6351691740E-06, 2.6635581564E-06, 1.2747715418E-06, 2.1803445046E-06, 3.1554282191E-06, 3.5047568595E-06, 2.3244846002E-06, 2.2566226153E-06, 2.8144830793E-06, 4.9714391664E-06, 6.2940620509E-06, 6.2261379071E-06, 3.9333499092E-06, 1.2929142317E-06, -3.7416435400E-07, -1.6404020060E-07, -7.8824897380E-07, -3.6957353938E-07, -1.5949166315E-07, -8.8476710902E-08, -1.7475548434E-08, -1.6846518415E-06, -8.4093256024E-08, 4.0402724232E-07, 1.3788204877E-06, 2.3535997143E-06, 1.5902030564E-06, 3.8859626481E-06, 2.8444336236E-06, 2.0114731172E-06, 1.6651852631E-06, 1.6665181954E-06, 1.3897330483E-06, 1.5300949562E-06, 2.0180780531E-06, 2.2279424732E-06, 1.6730043508E-06, 9.0947482015E-07, -9.6649139042E-07, -9.6525137222E-07, -4.0780992085E-07, -1.7971230319E-06, -1.9680510103E-07, 1.2970876207E-08, 2.0304272042E-06, 7.8012438731E-07, 1.6156071300E-06, 9.2149653656E-07, 1.5784821147E-07, 8.5424775910E-07, 8.5537178384E-07, 1.6908048901E-06, 2.8738601822E-06, 3.1530607647E-06, 2.8065137418E-06, 4.6152792180E-06, 3.4343942977E-06, 6.5438808532E-07, 3.0888656852E-06, 1.7688992017E-06, -3.7761360760E-08, 1.7183890207E-07, 6.5953628495E-07, 4.0673414691E-06, 5.1112340347E-06, 1.9835250619E-06, 2.4711886310E-06, 2.0549988219E-06, 1.2911667151E-06, -3.7651750952E-07, -2.4613700716E-06, -1.4175347444E-06, -1.0689735415E-06, -1.0680529355E-06, -2.4241746328E-08, 7.4145540616E-07, -1.5520231978E-06, -1.6206603781E-06, 4.6601894540E-07, 1.7183718530E-06, 2.6230844247E-06, 2.0677327409E-06, 2.2076388816E-06, 1.9999049270E-06, 2.0702709573E-06, 4.0246525176E-07, -2.2244851978E-07, 1.0298477063E-06, 4.0492059342E-07, -1.6800705948E-06, -2.8874159481E-07, -1.4699011431E-06, -1.0519564763E-06, 1.3822519694E-06, 8.9633855732E-07, 1.8009498511E-06, -8.4029888586E-07, -3.4815541899E-06, -1.6035893976E-06, -1.9504806453E-06, -2.5754838696E-06, -1.7404350068E-06, -9.0539239361E-07, -8.2886634035E-10, -6.9538196150E-07, -2.0156794156E-06, -1.1806597650E-06, -6.9327814628E-07, -2.2221721959E-06, -1.6652743530E-06, 2.8214823151E-07, 2.3686193038E-06, -1.1765663901E-06, -3.1226461039E-06, -3.2610405916E-06, -3.1908612071E-06, -2.7009730425E-07, -8.2566643691E-07, -1.7984003590E-06, -1.1720276244E-06, -4.0660708689E-07, -7.5361620048E-07, -3.3583811901E-07, 7.0767380364E-07, -5.4319446641E-07, -2.1417002430E-06, 5.0090877987E-07, 3.2825662284E-06]
        
        self.update(self.resid)
        
    def create_canvas(self):
        self.ax = []
        
        ## Set left (QQ) plot ##
        self.ax.append(self.fig.add_subplot(1,2,1))
        self.ax[0].item = {}
        self.ax[0].item['data'] = self.ax[0].scatter([],[], color='0', facecolors='0',
            linewidth=0.5, alpha = 0.85)
        self.ax[0].item['ref_line'] = self.ax[0].add_line(mpl.lines.Line2D([],[], c = '#800000', zorder=1))
        
        self.ax[0].text(.5,1.03,'QQ-Plot of Residuals', fontsize='large',
            horizontalalignment='center', verticalalignment='top', transform=self.ax[0].transAxes)
        
        self.fig.subplots_adjust(left=0.06, bottom=0.05, right=0.98,
                        top=0.965, hspace=0, wspace=0.12)
        
        ## Set right (density) plot ##
        self.ax.append(self.fig.add_subplot(1,2,2))
        self.ax[1].item = {}
        
        self.ax[1].item['density'] = []
        for i in range(3):
            line = self.ax[1].add_line(mpl.lines.Line2D([],[], alpha=0.5))
            self.ax[1].item['density'].append(line)
        
        self.ax[1].item['outlier_l'] = []
        for i in range(2):
            self.ax[1].item['outlier_l'].append(self.ax[1].axvline(x=np.nan, ls='--', c = '#BF0000'))
        
        self.ax[1].text(.5,1.03,'Density Plot of Residuals', fontsize='large',
            horizontalalignment='center', verticalalignment='top', transform=self.ax[1].transAxes) 
        
        # Create canvas from Base
        super().create_canvas()
     
    def update(self, data, update_lim=False):
        def shape_data(t,x): return np.transpose(np.vstack((t, x)))
        
        def calc_density(x_grid, data):
            
            if isinstance(data, np.ndarray) and data.ndim == 1:
                data = data[:, np.newaxis]
            else:
                data = np.array(data)[:, np.newaxis]

            stdev = np.std(data)
            A = np.min([np.std(data), stats.iqr(data)/1.34])  # bandwidth is multiplied by std of sample
            bw = 0.9*A*len(data)**(-1./(data.ndim-1+4))
            # bw = 1.6*A*len(data)**(-1./(data.ndim-1+4))
            # OoM_guess = np.floor(np.log10(bw))
            
            # bandwidths = 10 ** np.linspace(OoM_guess-1, OoM_guess+1, 100)
            # grid = GridSearchCV(self.kde, {'bandwidth': bandwidths}, cv=LeaveOneOut())
            # grid.fit(x_grid[:, None])
            # print(bw, grid.best_params_)
            # bw = grid.best_params_['bandwidth']
            
            self.kde.set_params(bandwidth = bw)
            self.kde.fit(data)
            # score_samples() returns the log-likelihood of the samples
            log_pdf = self.kde.score_samples(x_grid[:, np.newaxis])
            density = np.exp(log_pdf)
            '''                
            dim = 1
            stdev = np.std(data)
            A = np.min([np.std(data), stats.iqr(data)/1.34])/stdev  # bandwidth is multiplied by std of sample
            bw_method = 0.9*A*len(data)**(-1./(dim+4))
            density = stats.gaussian_kde(data, bw_method=bw_method)(x_grid)
            '''
            return density
        
        from timeit import default_timer as timer
        start = timer()
        
        # for shock in [0]:
            # self.ax[0].add_line(mpl.lines.Line2D([],[], marker='$'+u'\u2195'+'$', 
                # markersize=12, markerfacecolor='#BF0000', markeredgecolor='None', linestyle='None', zorder=4))
        
        # Update left plot
        # self.ax[1].item['exp_data'].set_offsets(shape_data(t, data))
        # self.ax[1].item['cutoff_l'].set_xdata(t[start_ind]*np.ones((2,1)))
        # self.ax[1].item['exp_data'].set_facecolor(np.char.mod('%f', 1-weights))
        
        # Update right plot
        x_grid = np.linspace(np.min(data), np.max(data), 500)
        density = calc_density(x_grid, data)
        self.ax[1].item['density'][0].set_xdata(x_grid)
        self.ax[1].item['density'][0].set_ydata(density)
        self.ax[1].fill_between(x_grid, 0, density, alpha=0.20)
        
        self.ax[1].set_xlim(x_grid[0], x_grid[-1])
        self.ax[1].set_ylim(np.min(density), np.max(density)*1.05)
        
        if update_lim:
            self.update_xylim(self.ax[0])
            self.update_xylim(self.ax[1])
    
        print('{:0.1f} us'.format((timer() - start)*1E3))
        
    def clear_opt(self):
        for line in self.ax[0].item['data']:    # clear lines in QQ plot
            line.set_xdata([])
            line.set_ydata([])
        
        for line in self.ax[1].item['density'][1:]: # clear lines in density plot
            line.set_xdata([])
            line.set_ydata([])
    
        
class AbsoluteLogScale(mplscale.LogScale):
    name = 'abslog'

    def __init__(self, axis, **kwargs):
        super().__init__(axis, **kwargs)
        
    def get_transform(self):
        return self.AbsLogTransform()
    
    class AbsLogTransform(mpl.transforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self):
            mpl.transforms.Transform.__init__(self)

        def transform_non_affine(self, a):
            masked = np.ma.masked_where(a == 0, a)
            if masked.mask.any():
                # ignore any divide by zero errors, 0 shouldn't exist due to mask
                with np.errstate(divide='ignore'):
                    return np.log10(np.abs(masked))
            else:
                return np.log10(np.abs(a))

        def inverted(self): # link to inverted transform class
            return AbsoluteLogScale.InvertedAbsLogTransform()

    class InvertedAbsLogTransform(mpl.transforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self):
            mpl.transforms.Transform.__init__(self)

        def transform_non_affine(self, a):
            return np.power(10, a)
            
        def inverted(self):
            return AbsoluteLogScale.AbsLogTransform()

           
class BiSymmetricLogScale(mplscale.ScaleBase):
    name = 'bisymlog'

    def __init__(self, axis, **kwargs):
        def isNum(s):
            try:
                float(s)
                return True
            except ValueError:
                return False
                
        super().__init__(axis, **kwargs)
        self.subs = np.arange(1, 10)
        if 'C' in kwargs and isNum(kwargs['C']): # Maybe should check for a specific value?
            self.set_C(float(kwargs['C']))
        else:
            self.set_C(0)
        
    def get_transform(self):
        return self.BiSymLogTransform(self.C)
    
    def set_C(self, C):
        if C == 0:  # Default C value
            self.C = 1/np.log(1000)
        else:
            self.C = C
    
    def set_default_locators_and_formatters(self, axis):
        class Locator(mpl.ticker.SymmetricalLogLocator):
            def __init__(self, transform, C, subs=None):
                if subs is None:
                    self._subs = None
                else:
                    self._subs = subs
                    
                self._base = transform.base
                self.numticks = 'auto'
                self.C = C
                self.transform = transform.transform
                self.inverse_transform = transform.inverted().transform
            
            def tick_values(self, vmin, vmax):
                def OoM(x):
                    x[x==0] = np.nan
                    return np.floor(np.log10(np.abs(x)))                       
                
                if self.numticks == 'auto':
                    if self.axis is not None:
                        numticks = np.clip(self.axis.get_tick_space(), 2, 9)
                    else:
                        numticks = 9
                else:
                    numticks = self.numticks
                
                if vmax < vmin:
                    vmin, vmax = vmax, vmin
                
                vmin_scale = self.transform(vmin)
                vmax_scale = self.transform(vmax)
                
                # quicker way would only operate on min, second point and max
                scale_ticklocs = np.linspace(vmin_scale, vmax_scale, numticks)
                raw_ticklocs = self.inverse_transform(scale_ticklocs)
                raw_tick_OoM = OoM(raw_ticklocs)

                zero_OoM = np.nanmin(raw_tick_OoM) # nearest to zero
                min_OoM = raw_tick_OoM[0]
                max_OoM = raw_tick_OoM[-1]  
                min_dist = scale_ticklocs[2] - scale_ticklocs[1]

                if vmin <= 0 <= vmax:
                    if min_dist > self.transform(10**zero_OoM):
                        min_dist = self.inverse_transform(min_dist)
                        zero_OoM = np.round(np.log10(np.abs(min_dist)))
                    
                    if vmin == 0:
                        numdec = np.abs(max_OoM - 2*zero_OoM)
                    elif vmax == 0:
                        numdec = np.abs(min_OoM - 2*zero_OoM)
                    else:
                        numdec = np.abs(min_OoM + max_OoM - 2*zero_OoM)
                    
                    stride = 1
                    while numdec // stride + 2 > numticks - 1:
                        stride += 1
                    
                    if vmin < 0:
                        neg_dec = np.arange(zero_OoM, min_OoM + stride, stride)
                        neg_sign = np.ones_like(neg_dec)*-1
                        idx_zero = len(neg_dec)
                    else:
                        neg_dec = []
                        neg_sign = []
                        idx_zero = 0
                    
                    if vmax > 0:
                        pos_dec = np.arange(zero_OoM, max_OoM + stride, stride)
                        pos_sign = np.ones_like(pos_dec)
                    else:
                        pos_dec = []
                        pos_sign = []
                        idx_zero = len(neg_dec)
                    
                    decades = np.concatenate((neg_dec, pos_dec))
                    sign = np.concatenate((neg_sign, pos_sign))
                    
                    ticklocs = np.multiply(sign, np.power(10, decades))
                    
                    # insert 0
                    idx = ticklocs.searchsorted(0)
                    ticklocs = np.concatenate((ticklocs[:idx][::-1], [0], ticklocs[idx:]))
                    
                else:
                    numdec = np.abs(max_OoM - min_OoM)
                    stride = 1
                    while numdec // stride + 2 > numticks - 1:
                        stride += 1
                    
                    sign = np.sign(vmin_scale)
                    
                    if sign == -1:
                        decades = np.arange(max_OoM, min_OoM + stride, stride)
                    else:
                        decades = np.arange(min_OoM, max_OoM + stride, stride)
                        
                    ticklocs = sign*np.power(10, decades)
                        
                    scale_ticklocs = self.transform(ticklocs) 
                    diff = np.diff(scale_ticklocs) 
                    n = 0
                    for i in range(len(scale_ticklocs)-1):
                        if min_dist*0.25 > np.abs(diff[i]):
                            ticklocs = np.delete(ticklocs, n)
                        else:
                            n += 1
                
                # Add the subticks if requested
                if self._subs is None or stride != 1:
                    subs = np.array([1.0])
                else:
                    subs = np.asarray(self._subs)
                
                if len(subs) > 1 or subs[0] != 1.0:
                    decades = ticklocs
                    ticklocs = []
                    for decade in decades:
                        if decade == 0:
                            ticklocs.append(decade)
                        else:
                            ticklocs.extend(subs*decade)
                
                return self.raise_if_exceeds(np.array(ticklocs))

        axis.set_major_locator(Locator(self.get_transform(), self.C))
        axis.set_major_formatter(mpl.ticker.LogFormatterMathtext())
        axis.set_minor_locator(Locator(self.get_transform(), self.C, self.subs))
        axis.set_minor_formatter(mpl.ticker.NullFormatter())
    
    class BiSymLogTransform(mpl.transforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, C, base=10):
            mpl.transforms.Transform.__init__(self)
            self.base = base
            self.C = C

        def transform_non_affine(self, a):
            return np.sign(a)*np.log10(1 + np.abs(a/self.C))/np.log10(self.base)

        def inverted(self): # link to inverted transform class
            return BiSymmetricLogScale.InvertedBiSymLogTransform(self.C)

    class InvertedBiSymLogTransform(mpl.transforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, C, base=10):
            mpl.transforms.Transform.__init__(self)
            self.base = base
            self.C = C

        def transform_non_affine(self, a):
            return np.sign(a)*self.C*(-1 + np.power(self.base, np.abs(a)))
            
        def inverted(self):
            return BiSymmetricLogScale.BiSymLogTransform(self.C)            


class Matplotlib_Click_Passthrough(object):
    def __init__(self, artists):
        if not isinstance(artists, list):
            artists = [artists]
        self.artists = artists
        artists[0].figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        for artist in self.artists:
            artist.pick(event)
            

class DraggableLegend(mpl.legend.DraggableLegend):
    def __init__(self, parent, legend, use_blit=False, update='loc'):
        super().__init__(legend, use_blit, update)
        # print(self.parent)
        
        self.parent = parent
        
        self._animate_items = parent._animate_items
        self._draw_items_artist = parent._draw_items_artist
        self.set_background = parent.set_background
    
    def on_pick(self, evt):
        if self._check_still_parented() and evt.artist == self.ref_artist:
            self.parent.canvas.mpl_disconnect(self.parent._draw_event_signal)
            
            self.mouse_x = evt.mouseevent.x
            self.mouse_y = evt.mouseevent.y
            
            self.got_artist = True            

            if self._use_blit:
                self._animate_items(True)   # draw everything but the selected objects and store the pixel buffer
                self._draw_items_artist()   # redraw the changing objects
                
                self._c1 = self.canvas.mpl_connect('motion_notify_event', self.on_motion_blit)
            else:
                self._c1 = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
            
            self.save_offset()
    
    def on_motion_blit(self, evt):
        if self._check_still_parented() and self.got_artist:
            dx = evt.x - self.mouse_x
            dy = evt.y - self.mouse_y
            self.update_offset(dx, dy)
            self._draw_items_artist()
    
    def on_motion(self, evt):
        if self._check_still_parented() and self.got_artist:
            dx = evt.x - self.mouse_x
            dy = evt.y - self.mouse_y
            self.update_offset(dx, dy)
            self.canvas.draw()
    
    def on_release(self, event):
        if self._check_still_parented() and self.got_artist:
            self.finalize_offset()
            self.got_artist = False
            self.canvas.mpl_disconnect(self._c1)

            if self._use_blit:
                # reconnect _draw_event
                draw_event_signal = lambda event: self.parent._draw_event(event)
                self.parent._draw_event_signal = self.parent.canvas.mpl_connect(
                    'draw_event', draw_event_signal)
                # self.ref_artist.set_animated(False)
            
class Draggable:
    lock = None  # only one can be animated at a time
    def __init__(self, parent, obj, update_fcn):
        self.parent = parent    # this is a local parent (the plot)
        self.dr_obj = obj
        self.canvas = parent.canvas
        self.ax = parent.ax
        
        self.update_fcn = update_fcn
        self._animate_items = parent._animate_items
        self._draw_items_artist = parent._draw_items_artist
        self.set_background = parent.set_background
        
        self.press = None
        
        self.connect()

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.parent.canvas.mpl_connect(
            'button_press_event', lambda event: self._on_press(event))
        self.cidrelease = self.parent.canvas.mpl_connect(
            'button_release_event', lambda event: self._on_release(event))
        self.cidmotion = self.parent.canvas.mpl_connect(
            'motion_notify_event', lambda event: self._on_motion(event))
    
    def _on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.dr_obj.axes: return
        if Draggable.lock is not None: return
        contains, attrd = self.dr_obj.contains(event)
        if not contains: return
        
        # disconnect _draw_event
        parent = self.parent
        parent.canvas.mpl_disconnect(parent._draw_event_signal)
        
        x0 = self.dr_obj.get_xdata()
        y0 = self.dr_obj.get_ydata()
        self.press = x0, y0, event.xdata, event.ydata
        Draggable.lock = self

        self._animate_items(True)   # draw everything but the selected objects and store the pixel buffer
        # self.canvas.draw()
        # self.set_background()     # this is causing blank background on press if no movement
        self._draw_items_artist()   # redraw the changing objects

    def _on_motion(self, event):
        'on motion we will move the line if the mouse is over us'
        if Draggable.lock is not self:
            return
        if event.inaxes != self.dr_obj.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        xnew = x0+dx
        ynew = y0+dy
        
        self.update_fcn([x0, y0, xpress, ypress, xnew, ynew])
        self._draw_items_artist()

    def _on_release(self, event):
        'on release we reset the press data'
        if Draggable.lock is not self:
            return

        self.press = None
        Draggable.lock = None
        
        # reconnect _draw_event
        parent = self.parent
        draw_event_signal = lambda event: parent._draw_event(event)
        parent._draw_event_signal = parent.canvas.mpl_connect('draw_event', draw_event_signal)

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.parent.canvas.mpl_disconnect(self.cidpress)
        self.parent.canvas.mpl_disconnect(self.cidrelease)
        self.parent.canvas.mpl_disconnect(self.cidmotion)