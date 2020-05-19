# This file is part of Frhodo. Copyright © 2020, UChicago Argonne, LLC
# and licensed under BSD-3-Clause. See License.txt in the top-level 
# directory for license and copyright information.

import matplotlib as mpl


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
        if self.dr_obj.axes is None or self.parent.toolbar.mode: return # don't drag if something else is selected
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
  