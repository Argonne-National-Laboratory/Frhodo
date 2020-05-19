import numpy as np
from scipy import stats
from tabulate import tabulate
from copy import deepcopy
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backend_bases import key_press_handler
from matplotlib import scale as mplscale
from matplotlib import figure as mplfigure
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT
from qtpy.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot
from qtpy.QtWidgets import QMenu, QAction
from qtpy import QtCore, QtGui

colormap = ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
            "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
            "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
            "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
            "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
            "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
            "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
            "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
            "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
            "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
            "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
            "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
            "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
            "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
            "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
            "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",
            "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D",
            "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176",
            "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5",
            "#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4",
            "#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01",
            "#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966",
            "#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0",
            "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C",
            "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868",
            "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183",
            "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
            "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F",
            "#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E",
            "#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F",
            "#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00",
            "#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66",
            "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25",
            "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B", "#1E2324", "#DEC9B2", "#9D4948",
            "#85ABB4", "#342142", "#D09685", "#A4ACAC", "#00FFFF", "#AE9C86", "#742A33", "#0E72C5",
            "#AFD8EC", "#C064B9", "#91028C", "#FEEDBF", "#FFB789", "#9CB8E4", "#AFFFD1", "#2A364C",
            "#4F4A43", "#647095", "#34BBFF", "#807781", "#920003", "#B3A5A7", "#018615", "#F1FFC8",
            "#976F5C", "#FF3BC1", "#FF5F6B", "#077D84", "#F56D93", "#5771DA", "#4E1E2A", "#830055",
            "#02D346", "#BE452D", "#00905E", "#BE0028", "#6E96E3", "#007699", "#FEC96D", "#9C6A7D",
            "#3FA1B8", "#893DE3", "#79B4D6", "#7FD4D9", "#6751BB", "#B28D2D", "#E27A05", "#DD9CB8",
            "#AABC7A", "#980034", "#561A02", "#8F7F00", "#635000", "#CD7DAE", "#8A5E2D", "#FFB3E1",
            "#6B6466", "#C6D300", "#0100E2", "#88EC69", "#8FCCBE", "#21001C", "#511F4D", "#E3F6E3",
            "#FF8EB1", "#6B4F29", "#A37F46", "#6A5950", "#1F2A1A", "#04784D", "#101835", "#E6E0D0",
            "#FF74FE", "#00A45F", "#8F5DF8", "#4B0059", "#412F23", "#D8939E", "#DB9D72", "#604143",
            "#B5BACE", "#989EB7", "#D2C4DB", "#A587AF", "#77D796", "#7F8C94", "#FF9B03", "#555196",
            "#31DDAE", "#74B671", "#802647", "#2A373F", "#014A68", "#696628", "#4C7B6D", "#002C27",
            "#7A4522", "#3B5859", "#E5D381", "#FFF3FF", "#679FA0", "#261300", "#2C5742", "#9131AF",
            "#AF5D88", "#C7706A", "#61AB1F", "#8CF2D4", "#C5D9B8", "#9FFFFB", "#BF45CC", "#493941",
            "#863B60", "#B90076", "#003177", "#C582D2", "#C1B394", "#602B70", "#887868", "#BABFB0",
            "#030012", "#D1ACFE", "#7FDEFE", "#4B5C71", "#A3A097", "#E66D53", "#637B5D", "#92BEA5",
            "#00F8B3", "#BEDDFF", "#3DB5A7", "#DD3248", "#B6E4DE", "#427745", "#598C5A", "#B94C59",
            "#8181D5", "#94888B", "#FED6BD", "#536D31", "#6EFF92", "#E4E8FF", "#20E200", "#FFD0F2",
            "#4C83A1", "#BD7322", "#915C4E", "#8C4787", "#025117", "#A2AA45", "#2D1B21", "#A9DDB0",
            "#FF4F78", "#528500", "#009A2E", "#17FCE4", "#71555A", "#525D82", "#00195A", "#967874",
            "#555558", "#0B212C", "#1E202B", "#EFBFC4", "#6F9755", "#6F7586", "#501D1D", "#372D00",
            "#741D16", "#5EB393", "#B5B400", "#DD4A38", "#363DFF", "#AD6552", "#6635AF", "#836BBA",
            "#98AA7F", "#464836", "#322C3E", "#7CB9BA", "#5B6965", "#707D3D", "#7A001D", "#6E4636",
            "#443A38", "#AE81FF", "#489079", "#897334", "#009087", "#DA713C", "#361618", "#FF6F01",
            "#006679", "#370E77", "#4B3A83", "#C9E2E6", "#C44170", "#FF4526", "#73BE54", "#C4DF72",
            "#ADFF60", "#00447D", "#DCCEC9", "#BD9479", "#656E5B", "#EC5200", "#FF6EC2", "#7A617E",
            "#DDAEA2", "#77837F", "#A53327", "#608EFF", "#B599D7", "#A50149", "#4E0025", "#C9B1A9",
            "#03919A", "#1B2A25", "#E500F1", "#982E0B", "#B67180", "#E05859", "#006039", "#578F9B",
            "#305230", "#CE934C", "#B3C2BE", "#C0BAC0", "#B506D3", "#170C10", "#4C534F", "#224451",
            "#3E4141", "#78726D", "#B6602B", "#200441", "#DDB588", "#497200", "#C5AAB6", "#033C61",
            "#71B2F5", "#A9E088", "#4979B0", "#A2C3DF", "#784149", "#2D2B17", "#3E0E2F", "#57344C",
            "#0091BE", "#E451D1", "#4B4B6A", "#5C011A", "#7C8060", "#FF9491", "#4C325D", "#005C8B",
            "#E5FDA4", "#68D1B6", "#032641", "#140023", "#8683A9", "#CFFF00", "#A72C3E", "#34475A",
            "#B1BB9A", "#B4A04F", "#8D918E", "#A168A6", "#813D3A", "#425218", "#DA8386", "#776133",
            "#563930", "#8498AE", "#90C1D3", "#B5666B", "#9B585E", "#856465", "#AD7C90", "#E2BC00",
            "#E3AAE0", "#B2C2FE", "#FD0039", "#009B75", "#FFF46D", "#E87EAC", "#DFE3E6", "#848590",
            "#AA9297", "#83A193", "#577977", "#3E7158", "#C64289", "#EA0072", "#C4A8CB", "#55C899",
            "#E78FCF", "#004547", "#F6E2E3", "#966716", "#378FDB", "#435E6A", "#DA0004", "#1B000F",
            "#5B9C8F", "#6E2B52", "#011115", "#E3E8C4", "#AE3B85", "#EA1CA9", "#FF9E6B", "#457D8B",
            "#92678B", "#00CDBB", "#9CCC04", "#002E38", "#96C57F", "#CFF6B4", "#492818", "#766E52",
            "#20370E", "#E3D19F", "#2E3C30", "#B2EACE", "#F3BDA4", "#A24E3D", "#976FD9", "#8C9FA8",
            "#7C2B73", "#4E5F37", "#5D5462", "#90956F", "#6AA776", "#DBCBF6", "#DA71FF", "#987C95",
            "#52323C", "#BB3C42", "#584D39", "#4FC15F", "#A2B9C1", "#79DB21", "#1D5958", "#BD744E",
            "#160B00", "#20221A", "#6B8295", "#00E0E4", "#102401", "#1B782A", "#DAA9B5", "#B0415D",
            "#859253", "#97A094", "#06E3C4", "#47688C", "#7C6755", "#075C00", "#7560D5", "#7D9F00",
            "#C36D96", "#4D913E", "#5F4276", "#FCE4C8", "#303052", "#4F381B", "#E5A532", "#706690",
            "#AA9A92", "#237363", "#73013E", "#FF9079", "#A79A74", "#029BDB", "#FF0169", "#C7D2E7",
            "#CA8869", "#80FFCD", "#BB1F69", "#90B0AB", "#7D74A9", "#FCC7DB", "#99375B", "#00AB4D",
            "#ABAED1", "#BE9D91", "#E6E5A7", "#332C22", "#DD587B", "#F5FFF7", "#5D3033", "#6D3800",
            "#FF0020", "#B57BB3", "#D7FFE6", "#C535A9", "#260009", "#6A8781", "#A8ABB4", "#D45262",
            "#794B61", "#4621B2", "#8DA4DB", "#C7C890", "#6FE9AD", "#A243A7", "#B2B081", "#181B00",
            "#286154", "#4CA43B", "#6A9573", "#A8441D", "#5C727B", "#738671", "#D0CFCB", "#897B77",
            "#1F3F22", "#4145A7", "#DA9894", "#A1757A", "#63243C", "#ADAAFF", "#00CDE2", "#DDBC62",
            "#698EB1", "#208462", "#00B7E0", "#614A44", "#9BBB57", "#7A5C54", "#857A50", "#766B7E",
            "#014833", "#FF8347", "#7A8EBA", "#274740", "#946444", "#EBD8E6", "#646241", "#373917",
            "#6AD450", "#81817B", "#D499E3", "#979440", "#011A12", "#526554", "#B5885C", "#A499A5",
            "#03AD89", "#B3008B", "#E3C4B5", "#96531F", "#867175", "#74569E", "#617D9F", "#E70452",
            "#067EAF", "#A697B6", "#B787A8", "#9CFF93", "#311D19", "#3A9459", "#6E746E", "#B0C5AE",
            "#84EDF7", "#ED3488", "#754C78", "#384644", "#C7847B", "#00B6C5", "#7FA670", "#C1AF9E",
            "#2A7FFF", "#72A58C", "#FFC07F", "#9DEBDD", "#D97C8E", "#7E7C93", "#62E674", "#B5639E",
            "#FFA861", "#C2A580", "#8D9C83", "#B70546", "#372B2E", "#0098FF", "#985975", "#20204C",
            "#FF6C60", "#445083", "#8502AA", "#72361F", "#9676A3", "#484449", "#CED6C2", "#3B164A",
            "#CCA763", "#2C7F77", "#02227B", "#A37E6F", "#CDE6DC", "#CDFFFB", "#BE811A", "#F77183",
            "#EDE6E2", "#CDC6B4", "#FFE09E", "#3A7271", "#FF7B59", "#4E4E01", "#4AC684", "#8BC891",
            "#BC8A96", "#CF6353", "#DCDE5C", "#5EAADD", "#F6A0AD", "#E269AA", "#A3DAE4", "#436E83",
            "#002E17", "#ECFBFF", "#A1C2B6", "#50003F", "#71695B", "#67C4BB", "#536EFF", "#5D5A48",
            "#890039", "#969381", "#371521", "#5E4665", "#AA62C3", "#8D6F81", "#2C6135", "#410601",
            "#564620", "#E69034", "#6DA6BD", "#E58E56", "#E3A68B", "#48B176", "#D27D67", "#B5B268",
            "#7F8427", "#FF84E6", "#435740", "#EAE408", "#F4F5FF", "#325800", "#4B6BA5", "#ADCEFF",
            "#9B8ACC", "#885138", "#5875C1", "#7E7311", "#FEA5CA", "#9F8B5B", "#A55B54", "#89006A",
            "#AF756F", "#2A2000", "#576E4A", "#7F9EFF", "#7499A1", "#FFB550", "#00011E", "#D1511C",
            "#688151", "#BC908A", "#78C8EB", "#8502FF", "#483D30", "#C42221", "#5EA7FF", "#785715",
            "#0CEA91", "#FFFAED", "#B3AF9D", "#3E3D52", "#5A9BC2", "#9C2F90", "#8D5700", "#ADD79C",
            "#00768B", "#337D00", "#C59700", "#3156DC", "#944575", "#ECFFDC", "#D24CB2", "#97703C",
            "#4C257F", "#9E0366", "#88FFEC", "#B56481", "#396D2B", "#56735F", "#988376", "#9BB195",
            "#A9795C", "#E4C5D3", "#9F4F67", "#1E2B39", "#664327", "#AFCE78", "#322EDF", "#86B487",
            "#C23000", "#ABE86B", "#96656D", "#250E35", "#A60019", "#0080CF", "#CAEFFF", "#323F61",
            "#A449DC", "#6A9D3B", "#FF5AE4", "#636A01", "#D16CDA", "#736060", "#FFBAAD", "#D369B4",
            "#FFDED6", "#6C6D74", "#927D5E", "#845D70", "#5B62C1", "#2F4A36", "#E45F35", "#FF3B53",
            "#AC84DD", "#762988", "#70EC98", "#408543", "#2C3533", "#2E182D", "#323925", "#19181B",
            "#2F2E2C", "#023C32", "#9B9EE2", "#58AFAD", "#5C424D", "#7AC5A6", "#685D75", "#B9BCBD",
            "#834357", "#1A7B42", "#2E57AA", "#E55199", "#316E47", "#CD00C5", "#6A004D", "#7FBBEC",
            "#F35691", "#D7C54A", "#62ACB7", "#CBA1BC", "#A28A9A", "#6C3F3B", "#FFE47D", "#DCBAE3",
            "#5F816D", "#3A404A", "#7DBF32", "#E6ECDC", "#852C19", "#285366", "#B8CB9C", "#0E0D00",
            "#4B5D56", "#6B543F", "#E27172", "#0568EC", "#2EB500", "#D21656", "#EFAFFF", "#682021",
            "#2D2011", "#DA4CFF", "#70968E", "#FF7B7D", "#4A1930", "#E8C282", "#E7DBBC", "#A68486",
            "#1F263C", "#36574E", "#52CE79", "#ADAAA9", "#8A9F45", "#6542D2", "#00FB8C", "#5D697B",
            "#CCD27F", "#94A5A1", "#790229", "#E383E6", "#7EA4C1", "#4E4452", "#4B2C00", "#620B70",
            "#314C1E", "#874AA6", "#E30091", "#66460A", "#EB9A8B", "#EAC3A3", "#98EAB3", "#AB9180",
            "#B8552F", "#1A2B2F", "#94DDC5", "#9D8C76", "#9C8333", "#94A9C9", "#392935", "#8C675E",
            "#CCE93A", "#917100", "#01400B", "#449896", "#1CA370", "#E08DA7", "#8B4A4E", "#667776",
            "#4692AD", "#67BDA8", "#69255C", "#D3BFFF", "#4A5132", "#7E9285", "#77733C", "#E7A0CC",
            "#51A288", "#2C656A", "#4D5C5E", "#C9403A", "#DDD7F3", "#005844", "#B4A200", "#488F69",
            "#858182", "#D4E9B9", "#3D7397", "#CAE8CE", "#D60034", "#AA6746", "#9E5585", "#BA6200"]

for i in range(4):    # really lazy way to reorder this list rather than edit it
    colormap.append(colormap.pop(1))

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
        else:
            for name in ['weight_l', 'density']:
                if name in axes.item:
                    if isinstance(axes.item[name], list):
                        for coord in ['x', 'y']:
                            range = np.array([])
                            for item in axes.item[name]:
                                if coord == 'x':    
                                    coordData = item.get_xdata()
                                elif coord == 'y':
                                    coordData = item.get_ydata()
                                
                                coordData = np.array(coordData)[np.isfinite(coordData)]
                                if coordData.size == 0:
                                    continue
                                    
                                data_min = coordData.min()
                                data_max = coordData.max()
                                
                                if range.size == 0:
                                    range = np.array([data_min, data_max])
                                else:
                                    if data_min < range[0]:
                                        range[0] = data_min
                                    if data_max > range[1]:
                                        range[1] = data_max
                                     
                            data[coord] = range
                    else:
                        data['x'] = axes.item[name].get_xdata()
                        data['y'] = axes.item[name].get_ydata()
        
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
        
        self.fig.subplots_adjust(left=0.06, bottom=0.065, right=0.98,
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
            self.update_sim(parent.SIM.t_lab, parent.SIM.drhodz)
                
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
    
    def update_sim(self, t, drhodz, rxnChanged=False):
        time_offset = parent.display_shock['time_offset']
        exp_data = parent.display_shock['exp_data']
        
        self.ax[0].item['sim_info_text'].set_text(self.info_table_text())
        
        if len(self.ax[1].item['history_data']) > 0:
            self.update_history()
        
        self.ax[1].item['sim_data'].set_xdata(t + time_offset)
        self.ax[1].item['sim_data'].set_ydata(drhodz)
        
        if exp_data.size == 0: # if exp data doesn't exist rescale
            self.set_xlim(self.ax[1], [np.round(np.min(t))-1, np.round(np.max(t))+1])
            if np.count_nonzero(drhodz):    # only update ylim if not all values are zero
                self.set_ylim(self.ax[1], drhodz)
        
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
            self.fig.subplots_adjust(left=0.06, bottom=0.065, right=0.94,
                                     top=0.97, hspace=0, wspace=0.12)
        else:
            self.ax[1].get_yaxis().set_visible(False)
            self.fig.subplots_adjust(left=0.06, bottom=0.065, right=0.98,
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
        self.canvas.mpl_connect("motion_notify_event", self.hover)
        self.threadpool = parent.threadpool
        
    def create_canvas(self):
        self.ax = []
        
        ## Set left (QQ) plot ##
        self.ax.append(self.fig.add_subplot(1,2,1))
        self.ax[0].item = {}
        self.ax[0].item['data'] = self.ax[0].scatter([],[], color='0', facecolors='0',
            linewidth=0.5, alpha = 0.85)
        self.ax[0].item['ref_line'] = self.ax[0].add_line(mpl.lines.Line2D([],[], c = '#800000', zorder=1E100))
        
        self.ax[0].text(.5,1.03,'QQ-Plot of Residuals', fontsize='large',
            horizontalalignment='center', verticalalignment='top', transform=self.ax[0].transAxes)
        
        self.fig.subplots_adjust(left=0.06, bottom=0.065, right=0.98,
                        top=0.965, hspace=0, wspace=0.12)
        
        ## Set right (density) plot ##
        self.ax.append(self.fig.add_subplot(1,2,2))
        self.ax[1].item = {}
        
        self.ax[1].item['density'] = [self.ax[1].add_line(mpl.lines.Line2D([],[], c=colormap[0]))]
        self.ax[1].item['shade'] = []
        for i in range(2):
            self.add_density_line()
            shaded = self.ax[1].fill_between([0, 0], 0, 0, color=colormap[i+1], alpha=0.20, zorder=i+1)
            self.ax[1].item['shade'].append(shaded)
        
        self.ax[1].item['outlier_l'] = []
        for i in range(2):
            self.ax[1].item['outlier_l'].append(self.ax[1].axvline(x=np.nan, ls='--', c = '#BF0000'))
        
        self.ax[1].text(.5,1.03,'Density Plot of Residuals', fontsize='large',
            horizontalalignment='center', verticalalignment='top', transform=self.ax[1].transAxes) 
        
        self.ax[1].item['annot'] = self.ax[1].annotate("", xy=(0,0), xytext=(-20,20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        self.ax[1].item['annot'].set_visible(False)
        
        # Create canvas from Base
        super().create_canvas()
     
    def add_density_line(self):
        i = len(self.ax[1].item['density'])
        line = self.ax[1].add_line(mpl.lines.Line2D([],[], color=colormap[i+1], alpha=0.5, zorder=i+1))
        self.ax[1].item['density'].append(line)
     
    def update_annot(self, line, ind):
        annot = self.ax[1].item['annot']
        x,y = line.get_data()
        annot.xy = (x[ind['ind'][0]], y[ind['ind'][0]])
        # text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                               # " ".join([names[n] for n in ind["ind"]]))
        text = 'cool'
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(self, event):
        annot = self.ax[1].item['annot']
        vis = annot.get_visible()
        if event.inaxes == self.ax[1]:
            for line in self.ax[1].item['density']:
                contains, ind = line.contains(event)
                if contains:
                    self.update_annot(line, ind)
                    annot.set_visible(True)
                    self.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        self.canvas.draw_idle()

    def update(self, data, update_lim=True):
        def shape_data(x, y): return np.transpose(np.vstack((x, y)))
        
        shocks2run = data['shocks2run']
        resid = data['resid']
        weights = data['weights']
        num_shocks = len(shocks2run)
        
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
        # clear shades
        for shade in self.ax[1].item['shade'][::-1]:
            shade.remove()
            
        self.ax[1].item['shade'] = []
        self.clear_opt()    # clear density plots
        
        # kernel density estimates
        allResid = np.concatenate(resid, axis=0)

        mu, std = stats.norm.fit(allResid)
        x_grid = np.linspace(mu-4*std, mu+4*std, 200)
        
        gaussian_fit = stats.norm.pdf(x_grid, mu, std)
        self.ax[1].item['density'][0].set_xdata(x_grid)
        self.ax[1].item['density'][0].set_ydata(gaussian_fit)
        
        worker = Update_Densities(self, self.add_density_line, x_grid, data)
        self.threadpool.start(worker)
        # self.threadpool.waitForDone()
        
        if update_lim:
            self.update_xylim(self.ax[0])
            self.update_xylim(self.ax[1])
    
        # print('{:0.1f} us'.format((timer() - start)*1E3))

    def clear_opt(self):
        # for line in self.ax[0].item['data']:    # clear lines in QQ plot
            # line.set_xdata([])
            # line.set_ydata([])
        
        for line in self.ax[1].item['density'][1:]: # clear lines in density plot
            line.set_xdata([])
            line.set_ydata([])
 
class WorkerSignals(QObject):
    result = pyqtSignal(int)
 
class Update_Densities(QRunnable):
    def __init__(self, parent, add_density_line, x_grid, data):
        super().__init__()
        
        self.signals = WorkerSignals()
        self.ax = parent.ax[1]
        self.add_density_line = add_density_line
        self.x_grid = x_grid
        self.data = data
      
    def update_densities(self):   # run in it's own thread for gui responsiveness
        def calc_density(x_grid, data, dim=1):
            stdev = np.std(data)
            A = np.min([np.std(data), stats.iqr(data)/1.34])/stdev  # bandwidth is multiplied by std of sample
            bw = 0.9*A*len(data)**(-1./(dim+4))
            density = stats.gaussian_kde(data, bw_method=bw)(x_grid)

            return density
        
        x_grid = self.x_grid
        data = self.data
        ax = self.ax
        
        shocks2run = data['shocks2run']
        resid = data['resid']
        weights = data['weights']
        num_shocks = len(shocks2run)
        
        for i in range(num_shocks):
            if len(ax.item['density'])-2 < i:   # add line if fewer than experiments
                self.add_density_line()
            
            density = calc_density(x_grid, resid[i])
            ax.item['density'][i+1].set_xdata(x_grid)
            ax.item['density'][i+1].set_ydata(density)
                   
            zorder = ax.item['density'][i+1].zorder
            color = ax.item['density'][i+1]._color
            shade = ax.fill_between(x_grid, 0, density, alpha=0.01, zorder=zorder, color=color)
            ax.item['shade'].append(shade)
    
    @pyqtSlot()
    def run(self):
        try:
            self.update_densities()
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            # self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit([])  # Return the result of the processing
        finally:
            pass
        
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