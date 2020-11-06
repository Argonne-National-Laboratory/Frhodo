# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import matplotlib as mpl
import numpy as np

from plot.base_plot import Base_Plot
from plot.draggable import DraggableLegend


class Plot(Base_Plot):
    def __init__(self, parent, widget, mpl_layout):
        super().__init__(parent, widget, mpl_layout)

    def create_canvas(self):
        self.ax = []
        self.ax.append(self.fig.add_subplot(1,1,1))
        self.ax.append(self.ax[0].twinx())
        self.ax[0].set_zorder(1)
        self.ax[1].set_zorder(0)    # put secondary axis behind primary
        
        max_lines = self.parent.sim_explorer.max_history + 1
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


class Matplotlib_Click_Passthrough(object):
    def __init__(self, artists):
        if not isinstance(artists, list):
            artists = [artists]
        self.artists = artists
        artists[0].figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        for artist in self.artists:
            artist.pick(event)