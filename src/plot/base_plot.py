# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

from qtpy.QtWidgets import QMenu, QAction
from qtpy import QtCore, QtGui

from copy import deepcopy

from matplotlib import figure as mplfigure
from matplotlib.backend_bases import key_press_handler

# This should make plotting backend qt binding indifferent
if QtCore.qVersion().split('.')[0] == '5':
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
elif QtCore.qVersion().split('.')[0] == '4':
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

import matplotlib as mpl
#import matplotlib.style as mplstyle
#mplstyle.use('fast')
#mpl.use("module://mplcairo.qt") # This implements mplcairo, faster/more accurate. Issues with other OSes?
import numpy as np

from plot.custom_mplscale import *
from plot.custom_mpl_ticker_formatter import *
from timeit import default_timer as timer

class Base_Plot(QtCore.QObject):
    def __init__(self, parent, widget, mpl_layout):
        super().__init__(parent)
        self.parent = parent

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
        self.background_data = self.canvas.copy_from_bbox(ax.bbox)
    
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
    
    def update_xylim(self, axes, xlim=[], ylim=[], force_redraw=True):
        data = self._get_data(axes)         

        # on creation, there is no data, don't update
        if np.shape(data['x'])[0] < 2 or np.shape(data['y'])[0] < 2:   
            return
        
        for (axis, lim) in zip(['x', 'y'], [xlim, ylim]):
            # Set Limits
            if len(lim) == 0:
                eval('self.set_' + axis + 'lim(axes, data["' + axis + '"])')
            else:
                eval('axes.set_' + axis + 'lim(lim)')
            
            # If bisymlog, also update scaling, C
            if eval('axes.get_' + axis + 'scale()') == 'bisymlog':
                self._set_scale(axis, 'bisymlog', axes)
            
            ''' # TODO: Do this some day, probably need to create 
                        annotation during canvas creation
            # Move exponent 
            exp_loc = {'x': (.89, .01), 'y': (.01, .96)}
            eval(f'axes.get_{axis}axis().get_offset_text().set_visible(False)')
            ax_max = eval(f'max(axes.get_{axis}ticks())')
            oom = np.floor(np.log10(ax_max)).astype(int)
            axes.annotate(fr'$\times10^{oom}$', xy=exp_loc[axis], 
                          xycoords='axes fraction')
            '''
        
        if force_redraw:
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
        
        elif 'weight_unc_fcn' in axes.item:
            data['x'] = axes.item['weight_unc_fcn'].get_xdata()
            data['y'] = axes.item['weight_unc_fcn'].get_ydata()
        
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
        def RoundToSigFigs(x, p):
            x = np.asarray(x)
            x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
            mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
            return np.round(x * mags) / mags
    
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
            # default string to evaluate 
            str = 'axes.set_{0:s}scale("{1:s}")'.format(coord, 'bisymlog')
            
            data = self._get_data(axes)[coord]
            if len(data) != 0:
                finite_data = np.array(data)[np.isfinite(data)] # ignore nan and inf
                min_data = finite_data.min()  
                max_data = finite_data.max()
                
                if min_data != max_data:
                    # if zero is within total range, find largest pos or neg range
                    if np.sign(max_data) != np.sign(min_data):  
                        pos_data = finite_data[finite_data>=0]
                        pos_range = pos_data.max() - pos_data.min()
                        neg_data = finite_data[finite_data<=0]
                        neg_range = neg_data.max() - neg_data.min()
                        C = np.max([pos_range, neg_range])
                    else:
                        C = np.abs(max_data-min_data)
                    C /= 1E3                  # scaling factor TODO: debating between 100, 500 and 1000
                    C = RoundToSigFigs(C, 1)  # round to 1 significant figure
                    str = 'axes.set_{0:s}scale("{1:s}", C={2:e})'.format(coord, 'bisymlog', C)
        
        eval(str)
        if type == 'linear' and coord == 'x':
            formatter = MathTextSciSIFormatter(useOffset=False, useMathText=True)
            axes.xaxis.set_major_formatter(formatter)
            
        elif type == 'linear' and coord == 'y':
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
        self.canvas.restore_region(self.background_data)           
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
    
    def set_background(self):
        self.canvas.mpl_disconnect(self._draw_event_signal)
        self.canvas.draw() # for when shock changes. Without signal disconnect, infinite loop
        self._draw_event_signal = self.canvas.mpl_connect('draw_event', self._draw_event)
        self.background_data = self.canvas.copy_from_bbox(self.fig.bbox)
    
    def _draw_event(self, event=None):   # After redraw (new/resizing window), obtain new background
        self._animate_items(True)
        self.set_background()
        self._draw_items_artist()
        #self.canvas.draw_idle()
    
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
            if not self.toolbar.mode:
                self._popup_menu(event)
            # if self.toolbar._active is 'ZOOM':  # if zoom is on, turn off
                # self.toolbar.press_zoom(event)  # cancels current zooom
                # self.toolbar.zoom()             # turns zoom off
            elif event.dblclick:                  # if double right click, go to default view
                self.toolbar.home()

    def key_press(self, event):
        if event.key == 'escape':
            if self.toolbar.mode == 'zoom rect':  # if zoom is on, turn off
                self.toolbar.zoom()               # turns zoom off
            elif self.toolbar.mode == 'pan/zoom':
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
        
        pos = self.parent.mapFromGlobal(QtGui.QCursor().pos())
        
        popup_menu = QMenu(self.parent)
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
                    
        popup_menu.exec_(self.parent.mapToGlobal(pos))    
    
    def _setAutoScale(self, choice, event, axes):
        if choice == len(self.autoScale):
            for n in range(len(self.autoScale)):
                self.autoScale[n] = event
        else:
            self.autoScale[choice] = event
        
        if event:   # if something toggled true, update limits
            self.update_xylim(axes)


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
        