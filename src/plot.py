# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import numpy as np
from scipy import stats
from tabulate import tabulate
from copy import deepcopy
import re, colors

import matplotlib as mpl
mpl.use("module://mplcairo.qt") # This implements mplcairo, faster/more accurate. Issues with other OSes?

from matplotlib import scale as mplscale, figure as mplfigure
from matplotlib.backend_bases import key_press_handler

# This should make plotting backend qt binding indifferent
from matplotlib.backends.qt_compat import is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
        
from qtpy.QtWidgets import QMenu, QAction
from qtpy import QtCore, QtGui

import plot_widget

colormap = colors.colormap(reorder_from=1, num_shift=4)
class All_Plots:    # container to hold all plots
    def __init__(self, main):
        global parent
        parent = main
        
        self.raw_sig = Raw_Signal_Plot(parent, parent.raw_signal_plot_widget, parent.mpl_raw_signal)
        self.signal = Signal_Plot(parent, parent.signal_plot_widget, parent.mpl_signal)
        self.sim_explorer = Sim_Explorer_Plot(parent, parent.sim_explorer_plot_widget, parent.mpl_sim_explorer)
        self.opt = Optimization_Plot(parent, parent.opt_plot_widget, parent.mpl_opt)
        
        self.observable_widget = plot_widget.Observable_Widgets(parent)
        
class Base_Plot(QtCore.QObject):
    def __init__(self, parent, widget, mpl_layout):
        super().__init__(parent)
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
            ax.ticklabel_format(scilimits=(-4, 4), useMathText=True)
        
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
            # range = np.abs(np.max(x) - np.min(x))
            # min = np.min(x) - range*0.05
            # if min < 0:
                # min = 0
            # xlim = [min, np.max(x) + range*0.05]
            xlim = [np.min(x), np.max(x)]
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
        
        min_data = np.array(y)[np.isfinite(y)].min()
        max_data = np.array(y)[np.isfinite(y)].max()
        
        if min_data == max_data:
            min_data -= 10**-1
            max_data += 10**-1
        
        if axes.get_yscale() == 'linear':
            range = np.abs(max_data - min_data)
            ylim = [min_data - range*0.1, max_data + range*0.1]
            
        elif axes.get_yscale() in ['log', 'abslog']:
            abs_y = np.abs(y)
            abs_y = abs_y[np.nonzero(abs_y)]    # exclude 0's
            abs_y = abs_y[np.isfinite(abs_y)]    # exclude nan, inf
            
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
    
    def update_xylim(self, axes, xlim=[], ylim=[]):
        data = self._get_data(axes)         

        # on creation, there is no data, don't update
        if np.shape(data['x'])[0] < 2 or np.shape(data['y'])[0] < 2:   
            return
        
        for (axis, lim) in zip(['x', 'y'], [xlim, ylim]):
            if len(lim) == 0:
                eval('self.set_' + axis + 'lim(axes, data["' + axis + '"])')
            else:
                eval('axes.set_' + axis + 'lim(lim)')
        self._draw_event()  # force a draw
    
    def _get_data(self, axes):      # NOT Generic
        # get experimental data for axes
        data = {'x': [], 'y': []}
        if 'exp_data' in axes.item:
            data_plot = axes.item['exp_data'].get_offsets().T
            if np.shape(data_plot)[1] > 1:
                data['x'] = data_plot[0,:]
                data['y'] = data_plot[1,:]
            
            # append sim_x if it exists
            if 'sim_data' in axes.item and hasattr(axes.item['sim_data'], 'raw_data'):
                if axes.item['sim_data'].raw_data.size > 0:
                    data['x'] = np.append(data['x'], axes.item['sim_data'].raw_data[:,0])
        
        elif 'weight' in axes.item:
            data['x'] = axes.item[name].get_xdata()
            data['y'] = axes.item[name].get_ydata()
        
        elif any(key in axes.item for key in ['density', 'qq_data', 'sim_data']):
            name = np.intersect1d(['density', 'qq_data'], list(axes.item.keys()))[0]
            for n, coord in enumerate(['x', 'y']):
                xyrange = np.array([])
                for item in axes.item[name]:
                    if name == 'qq_data':
                        coordData = item.get_offsets()
                        if coordData.size == 0:
                            continue
                        else:
                            coordData = coordData[:,n]
                    elif name == 'density':
                        coordData = eval('item.get_' + coord + 'data()')
                    
                    coordData = np.array(coordData)[np.isfinite(coordData)]
                    if coordData.size == 0:
                        continue
                    
                    xyrange = np.append(xyrange, [coordData.min(), coordData.max()])

                xyrange = np.reshape(xyrange, (-1,2))
                data[coord] = [np.min(xyrange[:,0]), np.max(xyrange[:,1])]

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
            min_data = np.array(data)[np.isfinite(data)].min()  # ignore nan and inf
            max_data = np.array(data)[np.isfinite(data)].max()
            
            if len(data) == 0 or min_data == max_data:  # backup in case set on blank plot
                str = 'axes.set_{0:s}scale("{1:s}")'.format(coord, 'bisymlog')
            else:
                # if zero is within total range, find largest pos or neg range
                if np.sign(max_data) != np.sign(min_data):  
                    pos_range = np.max(data[data>=0])-np.min(data[data>=0])
                    neg_range = np.max(-data[data<=0])-np.min(-data[data<=0])
                    C = np.max([pos_range, neg_range])
                else:
                    C = np.abs(max_data-min_data)
                C /= 5E2                             # scaling factor, debating between 100, 500 and 1000
                
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
    
    def clear_plot(self, ignore=[], draw=True):
        for axis in self.ax:
            if axis.get_legend() is not None:
                axis.get_legend().remove()
                
            for item in axis.item.values():
                if hasattr(item, 'set_offsets'):    # clears all data points
                    if 'scatter' not in ignore:
                        item.set_offsets(([np.nan, np.nan]))
                elif hasattr(item, 'set_xdata') and hasattr(item, 'set_ydata'):
                    if 'line' not in ignore:
                        item.set_xdata([np.nan, np.nan]) # clears all lines
                        item.set_ydata([np.nan, np.nan])
                elif hasattr(item, 'set_text'): # clears all text boxes
                    if 'text' not in ignore:
                        item.set_text('')
        if draw:
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
        self.toolbar = CustomNavigationToolbar(self.canvas, self.widget, coordinates=True)
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
                fcn = lambda event, coord=coord, type=type: self._set_scale(coord, type, axes, True)
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
        if parent.display_shock['raw_data'].size == 0: return
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
            # shift must be within the experiment
            n = item.info['n']
            exp_lim = np.array([exp_data[0,0], exp_data[-1,0]])/parent.var['reactor']['t_unit_conv']
            start = exp_lim[0]
            if n == 0:  # adjust limits
                exp_lim[1] = parent.weight.boxes['weight_shift'][1].value() + start - 0.01
            else:
                exp_lim[0] = parent.weight.boxes['weight_shift'][0].value() + start - 0.01
            if xnew < exp_lim[0]:
                xnew = exp_lim[0]
            elif xnew > exp_lim[1]:
                xnew = exp_lim[1]
            
            weight_shift = xnew - start
            parent.weight.boxes['weight_shift'][n].setValue(weight_shift)
            
        elif item in self.ax[0].item['weight_k']:   # save n on press, erase on release
            xy_data = item.get_xydata()
            i = item.draggable.nearest_index
            n = item.info['n']
            
            shift = exp_data[0,0]/parent.var['reactor']['t_unit_conv'] + parent.display_shock['weight_shift'][n]
            
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
        
        t = parent.display_shock['exp_data'][:,0]
        data = parent.display_shock['exp_data'][:,1]
        weight_shift = np.array(parent.display_shock['weight_shift'])*parent.var['reactor']['t_unit_conv']
        weight_k = np.array(parent.display_shock['weight_k'])*parent.var['reactor']['t_unit_conv']
        
        weight_fcn = parent.series.weights
        weights = parent.display_shock['weights'] = weight_fcn(t)
        
        # Update lower plot
        self.ax[1].item['exp_data'].set_offsets(shape_data(t, data))
        self.ax[1].item['exp_data'].set_facecolor(np.char.mod('%f', 1-weights))
        
        # Update upper plot
        self.ax[0].item['weight'].set_xdata(t)
        self.ax[0].item['weight'].set_ydata(weights)
        
        for i in range(0, 2):   # TODO: need to intelligently reorder to fix draggable bug
            mu = t[0] + weight_shift[i]
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
        time_offset = parent.display_shock['time_offset']
        exp_data = parent.display_shock['exp_data']
        
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
            self.set_xlim(self.ax[1], [np.round(np.min(t))-1, np.round(np.max(t))+1])
            if np.count_nonzero(observable):    # only update ylim if not all values are zero
                self.set_ylim(self.ax[1], observable)
        
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
        color = [mpl.rcParams['axes.prop_cycle'].by_key()['color'],
                 mpl.rcParams['axes.prop_cycle'].by_key()['color']]
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
            self.fig.subplots_adjust(left=0.06, bottom=0.065, right=0.94,
                                     top=0.97, hspace=0, wspace=0.12)
        else:
            self.ax[1].get_yaxis().set_visible(False)
            self.fig.subplots_adjust(left=0.06, bottom=0.065, right=0.98,
                                     top=0.97, hspace=0, wspace=0.12)
    
    def update(self, data, labels=[], label_order=[], update_lim=True):
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
                order = label_order[key] # reorder legend by label_order
                lines = np.array(axes.item['property'])
                leg = axes.legend(lines[order], label[order], loc=legend_loc[key])
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
        self.canvas.mpl_connect("motion_notify_event", self.hover)
        
    def create_canvas(self):
        self.ax = []
        
        ## Set left (QQ) plot ##
        self.ax.append(self.fig.add_subplot(1,2,1))
        self.ax[0].item = {}
        self.ax[0].item['qq_data'] = []
        self.ax[0].item['ref_line'] = self.ax[0].add_line(mpl.lines.Line2D([],[], c=colormap[0], zorder=2))
        
        self.ax[0].text(.5,1.03,'QQ-Plot of Residuals', fontsize='large',
            horizontalalignment='center', verticalalignment='top', transform=self.ax[0].transAxes)
        
        self.fig.subplots_adjust(left=0.06, bottom=0.065, right=0.98,
                        top=0.965, hspace=0, wspace=0.12)
        
        ## Set right (density) plot ##
        self.ax.append(self.fig.add_subplot(1,2,2))
        self.ax[1].item = {}
        
        self.ax[1].item['density'] = [self.ax[1].add_line(mpl.lines.Line2D([],[],
                                                          ls='-',  c=colormap[0], zorder=3))]
        
        # add exp scatter/line plots
        self.add_exp_plots()
        self.ax[1].item['shade'] = [self.ax[1].fill_between([0, 0], 0, 0)]
        
        self.ax[1].item['annot'] = self.ax[1].annotate("", xy=(0,0), xytext=(-100,20), 
                        textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), 
                        arrowprops=dict(arrowstyle="->"), zorder=10)
        self.ax[1].item['annot'].set_visible(False)
        
        self.ax[1].item['outlier_l'] = []
        for i in range(2):
            self.ax[1].item['outlier_l'].append(self.ax[1].axvline(x=np.nan, ls='--', c='k'))# c='#BF0000'))
        
        self.ax[1].text(.5,1.03,'Density Plot of Residuals', fontsize='large',
            horizontalalignment='center', verticalalignment='top', transform=self.ax[1].transAxes) 

        # Create canvas from Base
        super().create_canvas()
     
    def add_exp_plots(self):
        i = len(self.ax[1].item['density'])
        
        line = self.ax[1].add_line(mpl.lines.Line2D([],[], color=colormap[i+1], alpha=0.5, zorder=2))
        line.set_pickradius(line.get_pickradius()*2)
        self.ax[1].item['density'].append(line)
        
        scatter = self.ax[0].scatter([],[], color=colormap[i+1], facecolors=colormap[i+1], 
            s=16, linewidth=0.5, alpha = 0.85)
        scatter.set_pickradius(scatter.get_pickradius()*2)
        self.ax[0].item['qq_data'].append(scatter)
     
    def hover(self, event):
        def update_annot(line, ind):
            def closest_xy(point, points):
                dist_sqr = np.sum((points - point[:, np.newaxis])**2, axis=1)
                return points[:,np.argmin(dist_sqr)]
                
            annot = self.ax[1].item['annot']
            x,y = line.get_data()
            xy_mouse = self.ax[1].transData.inverted().transform([event.x, event.y])
            annot.xy = closest_xy(xy_mouse, np.array(line.get_data())[:, ind['ind']])   # nearest point to mouse
            
            text = '{:s}\nExp # {:d}'.format(line.shock_info['series_name'], line.shock_info['num'])
            annot.set_text(text)
            extents = annot.get_bbox_patch().get_extents()
            if np.mean(self.ax[1].get_xlim()) < annot.xy[0]: # if on left side of plot
                annot.set_x(-extents.width*0.8+20)
            else:
                annot.set_x(0)
            annot.get_bbox_patch().set_alpha(0.85)
            annot.set_visible(True)
    
        # if the event happened within axis and no toolbar buttons active do nothing
        if event.inaxes != self.ax[1] or self.toolbar._active is not None: return
        
        draw = False    # tells to draw or not based on annotation visibility change
        default_pick_radius = self.ax[1].item['density'][1].get_pickradius()
        contains_line = []
        for line in self.ax[1].item['density'][1:]:
            contains, ind = line.contains(event)
            if contains and hasattr(line, 'shock_info'):
                contains_line.append(line)     
        
        if len(contains_line) > 0:      # reduce pick radius until only 1 line contains event
            for n in range(1, default_pick_radius)[::-1]:
                if len(contains_line) == 1: # if only 1 item in list break
                    break
                    
                for i in range(len(contains_line))[::-1]:
                    if len(contains_line) == 1: # if only 1 item in list break
                        break
                    
                    contains_line[i].set_pickradius(n)
                    contains, ind = contains_line[i].contains(event)
                    
                    if not contains:
                        del contains_line[i]
            
            # update annotation based on leftover
            contains, ind = contains_line[0].contains(event)
            update_annot(contains_line[0], ind)
            draw = True
            
            for line in contains_line:  # resset pick radius
                line.set_pickradius(default_pick_radius)
                
        elif self.ax[1].item['annot'].get_visible():   # if not over a line, hide annotation
            draw = True
            self.ax[1].item['annot'].set_visible(False)
        
        if draw:
            self.canvas.draw_idle()
     
    def update(self, data, update_lim=True):
        def shape_data(x, y): return np.transpose(np.vstack((x, y)))
        
        shocks2run = data['shocks2run']
        resid = data['resid']
        weights = data['weights']
        resid_outlier = data['resid_outlier']
        num_shocks = len(shocks2run)
        
        # operations needed for both QQ and Density Plot
        allResid = np.concatenate(resid, axis=0)
        # weights = np.concatenate(weights, axis=0)
        # mu = np.average(allResid, weights=weights)
        # allResid -= mu
        # allResid = allResid[np.abs(allResid) < resid_outlier]
        # allResid += mu
        
        fitres = data['fit_result']
        x_grid = np.linspace(fitres[1]-4*fitres[2], fitres[1]+4*fitres[2], 300)
        xlim_density =[x_grid[0], x_grid[-1]]
        
        # add exp line/data if not enough
        for i in range(num_shocks):
            if len(self.ax[1].item['density'])-2 < i:   # add line if fewer than experiments
                self.add_exp_plots()
        
        self.clear_plot(ignore='text', draw=False)
        
        # from timeit import default_timer as timer
        # start = timer()
        
        # for shock in [0]:
            # self.ax[0].add_line(mpl.lines.Line2D([],[], marker='$'+u'\u2195'+'$', 
                # markersize=12, markerfacecolor='#BF0000', markeredgecolor='None', linestyle='None', zorder=4))
        
        # Update left plot
        xrange = np.array([])
        for i in range(num_shocks):
            QQ = data['QQ'][i]
            self.ax[0].item['qq_data'][i].set_offsets(QQ)
            
            xrange = np.append(xrange, [QQ[:,0].min(), QQ[:,0].max()])

            xrange = np.reshape(xrange, (-1,2))
            xrange = [np.min(xrange[:,0]), np.max(xrange[:,1])]
            
        self.ax[0].item['ref_line'].set_xdata(xrange)
        self.ax[0].item['ref_line'].set_ydata(xrange)
        
        # Update right plot
        # clear shades
        for shade in self.ax[1].item['shade'][::-1]:
            shade.remove()
            
        self.ax[1].item['shade'] = []
        
        # kernel density estimates
        gennorm_fit = stats.gennorm.pdf(x_grid, *fitres)
        self.ax[1].item['density'][0].set_xdata(x_grid)
        self.ax[1].item['density'][0].set_ydata(gennorm_fit)
        
        self.ax[1].item['outlier_l'][0].set_xdata(-resid_outlier*np.ones((2,1)))
        self.ax[1].item['outlier_l'][1].set_xdata(resid_outlier*np.ones((2,1)))
        
        for i in range(num_shocks):
            x_grid = data['KDE'][i][:,0]
            density = data['KDE'][i][:,1]
            self.ax[1].item['density'][i+1].set_xdata(x_grid)
            self.ax[1].item['density'][i+1].set_ydata(density)
            self.ax[1].item['density'][i+1].shock_info = shocks2run[i]
                   
            zorder = self.ax[1].item['density'][i+1].zorder
            color = self.ax[1].item['density'][i+1]._color
            shade = self.ax[1].fill_between(x_grid, 0, density, alpha=0.01, zorder=zorder, color=color)
            self.ax[1].item['shade'].append(shade)
        
        if update_lim:
            self.update_xylim(self.ax[0])
            self.update_xylim(self.ax[1], xlim=xlim_density)
    
        # print('{:0.1f} us'.format((timer() - start)*1E3))
            

class CustomNavigationToolbar(NavigationToolbar):
    # hide buttons
    NavigationToolbar.toolitems = (('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        (None, None, None, None),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        # ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
        (None, None, None, None),
        ('Save', 'Save the figure', 'filesave', 'save_figure'))
    
    
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
    def __init__(self, parent, obj, update_fcn, press_fcn=None, release_fcn=None):
        self.parent = parent    # this is a local parent (the plot)
        self.dr_obj = obj
        self.canvas = parent.canvas
        self.ax = parent.ax
        
        self.update_fcn = update_fcn
        self.press_fcn = press_fcn
        self.release_fcn = release_fcn
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
        
        if self.press_fcn is not None:
            x={'0': x0, 'press': event.xdata, 'new': event.xdata, 'press_new': event.xdata}
            y={'0': y0, 'press': event.ydata, 'new': event.ydata, 'press_new': event.ydata}
            self.press_fcn(x, y)
        
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
        
        x={'0': x0, 'press': xpress, 'new': xnew, 'press_new': event.xdata}
        y={'0': y0, 'press': ypress, 'new': ynew, 'press_new': event.ydata}
        
        self.update_fcn(x, y)
        self._draw_items_artist()

    def _on_release(self, event):
        'on release we reset the press data'
        if Draggable.lock is not self:
            return

        self.press = None
        Draggable.lock = None
        
        if self.release_fcn is not None:
            self.release_fcn()
        
        # reconnect _draw_event
        parent = self.parent
        draw_event_signal = lambda event: parent._draw_event(event)
        parent._draw_event_signal = parent.canvas.mpl_connect('draw_event', draw_event_signal)

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.parent.canvas.mpl_disconnect(self.cidpress)
        self.parent.canvas.mpl_disconnect(self.cidrelease)
        self.parent.canvas.mpl_disconnect(self.cidmotion)