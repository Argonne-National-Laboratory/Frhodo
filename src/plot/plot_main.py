# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.
        
from plot import raw_signal_plot, signal_plot, sim_explorer_plot, optimization_plot, plot_widget


class All_Plots:    # container to hold all plots
    def __init__(self, parent):
        self.raw_sig = raw_signal_plot.Plot(parent, parent.raw_signal_plot_widget, parent.mpl_raw_signal)
        self.signal = signal_plot.Plot(parent, parent.signal_plot_widget, parent.mpl_signal)
        self.sim_explorer = sim_explorer_plot.Plot(parent, parent.sim_explorer_plot_widget, parent.mpl_sim_explorer)
        self.opt = optimization_plot.Plot(parent, parent.opt_plot_widget, parent.mpl_opt)
        
        self.observable_widget = plot_widget.Observable_Widgets(parent)