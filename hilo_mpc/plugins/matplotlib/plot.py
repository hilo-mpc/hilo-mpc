#   
#   This file is part of HILO-MPC
#
#   HILO-MPC is a toolbox for easy, flexible and fast development of machine-learning-supported
#   optimal control and estimation problems
#
#   Copyright (c) 2021 Johannes Pohlodek, Bruno Morabito, Rolf Findeisen
#                      All rights reserved
#
#   HILO-MPC is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   HILO-MPC is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with HILO-MPC. If not, see <http://www.gnu.org/licenses/>.
#

# TODO: Typing hints
# NOTE: Adapted from pandas plotting functionality

from abc import ABCMeta, abstractmethod
from math import ceil
import warnings

import matplotlib.cm as cm
import matplotlib.colors
import numpy as np

from ...util.util import is_list_like, random_state


_LINE_KWARGS = {
    'line_alpha': 'alpha',
    'line_cap': 'solid_capstyle',
    'line_color': 'color',
    'line_dash': 'linestyle',
    'line_join': 'solid_joinstyle',
    'line_width': 'linewidth'
}
_STEP_KWARGS = {
    'step_mode': 'where'
}
_SCATTER_KWARGS = {
    'marker': 'marker',
    'marker_color': 'c',
    'marker_size': 's'
}
_FILL_BETWEEN_KWARGS = {
    'line_color': 'edgecolor',
    'line_width': 'linewidth',
    'fill_color': 'color',
    'fill_alpha': 'alpha'
}


class BasePlot(metaclass=ABCMeta):
    """"""
    _layout_type = 'grid'
    _pop_attributes = [
        'tight_layout',
        'dpi',
        'background_fill_color'
    ]
    _attr_defaults = {
        'tight_layout': True,
        'dpi': None,
        'background_fill_color': None
    }

    def __init__(
            self,
            data,
            subplots=False,
            figsize=None,
            legend=False,
            legend_title=None,
            ax=None,
            title=None,
            xlabel=None,
            ylabel=None,
            colormap=None,
            layout=None,
            fill_between=None,
            interactive=False,
            anim=False,
            **kwargs
    ):
        """Constructor method"""
        self.data = data

        self.subplots = subplots

        self.figsize = figsize
        self.layout = layout
        self.fill_between = fill_between
        self.interactive = interactive
        self.anim = anim

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.legend = legend
        self.legend_title = legend_title
        self.legend_handles = []
        self.legend_labels = []

        for attr in self._pop_attributes:
            value = kwargs.pop(attr, self._attr_defaults.get(attr))
            setattr(self, attr, value)

        self.ax = ax

        if 'cmap' in kwargs and colormap:
            raise TypeError("Only specify one of 'cmap' and 'colormap'")
        elif 'cmap' in kwargs:
            self.colormap = kwargs.pop('cmap')
        else:
            self.colormap = colormap

        self.kwargs = kwargs

    def _add_legend_handle(self, handle, label):
        """

        :param handle:
        :param label:
        :return:
        """
        if label is not None:
            self.legend_handles.append(handle)
            self.legend_labels.append(label)

    def _decorate_subplots(self):
        """

        :return:
        """
        if is_list_like(self.xlabel):
            if len(self.xlabel) != self.n_plots:
                msg = f"The length of 'xlabel' must equal the number of columns if using 'xlabel' of type " \
                      f"'list'.\nlength of xlabel = {len(self.xlabel)}\nnumber of columns = {self.n_plots}"
                raise ValueError(msg)

        if is_list_like(self.ylabel):
            if len(self.ylabel) != self.n_plots:
                msg = f"The length of 'ylabel' must equal the number of columns if using 'ylabel' of type " \
                      f"'list'.\nlength of ylabel = {len(self.ylabel)}\nnumber of columns = {self.n_plots}"
                raise ValueError(msg)

        for k, ax in enumerate(self.axes):
            if ax.get_visible():
                if is_list_like(self.xlabel):
                    if self.xlabel[k] is not None:
                        ax.set_xlabel(self.xlabel[k])
                else:
                    if self.xlabel is not None:
                        ax.set_xlabel(self.xlabel)

                if is_list_like(self.ylabel):
                    if self.ylabel[k] is not None:
                        ax.set_ylabel(self.ylabel[k])
                else:
                    if self.ylabel is not None:
                        ax.set_ylabel(self.ylabel)

                if self.interactive:
                    _update_slider(self.fig, ax, self.sliders[k], self.data[k])

        if self.title:
            if self.subplots:
                if is_list_like(self.title):
                    if len(self.title) != self.n_plots:
                        msg = f"The length of 'title' must equal the number of columns if using 'title' of type " \
                              f"'list' and 'subplots=True'.\nlength of title = {len(self.title)}\n" \
                              f"number of columns = {self.n_plots}"
                        raise ValueError(msg)

                    for ax, title in zip(self.axes, self.title):
                        ax.set_title(title)
                else:
                    self.fig.suptitle(self.title)
            else:
                if is_list_like(self.title):
                    msg = "Using 'title' of type 'list' is not supported unless 'subplots=True' is passed."
                    raise ValueError(msg)
                self.axes[0].set_title(self.title)

    def _get_ax(self, k):
        """

        :param k:
        :return:
        """
        if self.subplots:
            ax = self.axes[k]
        else:
            ax = self.axes[0]

        ax.get_yaxis().set_visible(True)
        return ax

    @staticmethod
    def _get_ax_legend_handle(ax):
        """

        :param ax:
        :return:
        """
        legend = ax.get_legend()

        handle, _ = ax.get_legend_handles_labels()
        other_ax = getattr(ax, 'left_ax', None) or getattr(ax, 'right_ax', None)
        other_legend = None
        if other_ax is not None:
            other_legend = other_ax.get_legend()
        if legend is not None and other_legend is not None:
            legend = other_legend
            ax = other_ax
        return ax, legend, handle

    def _get_colors(self, num_colors=None, color_kwargs='color'):
        """

        :param num_colors:
        :param color_kwargs:
        :return:
        """
        if num_colors is None:
            num_colors = self.n_plots

        return _get_standard_colors(num_colors=num_colors, colormap=self.colormap, color=self.kwargs.get(color_kwargs))

    def _iter_data(self):
        """

        :return:
        """
        for sub, plots in self.data.items():
            index = plots['x']
            if 'line_color' in self.kwargs:
                colors = self._get_colors(num_colors=len(plots['y']), color_kwargs='line_color')
            elif 'marker_color' in self.kwargs:
                colors = self._get_colors(num_colors=len(plots['y']), color_kwargs='marker_color')
            else:
                colors = self._get_colors(num_colors=len(plots['y']))
            keys = list(plots['y'])
            for key, values in plots['y'].items():
                if isinstance(colors, dict):
                    color = colors[key]
                else:
                    color = colors[keys.index(key)]
                yield sub, index[key], key, values, color

    def _make_fill_between(self):
        """

        :return:
        """
        if not self.subplots:
            if is_list_like(self.fill_between):
                fill_between = self.fill_between[0]
            else:
                fill_between = self.fill_between
            if fill_between is not None:
                ax = self._get_ax(0)
                args = self._process_fill_between(fill_between)
                ax.fill_between(*args, **fill_between)
        else:
            if is_list_like(self.fill_between):
                if len(self.fill_between) != self.n_plots:
                    msg = f"The length of 'fill_between' must equal the number of columns if using 'fill_between' of " \
                          f"type 'list'.\nlength of fill_between = {len(self.fill_between)}\nnumber of columns = " \
                          f"{self.n_plots}"
                    raise ValueError(msg)
            for k in range(self.n_plots):
                if is_list_like(self.fill_between):
                    fill_between = self.fill_between[k]
                elif self.fill_between is not None:
                    fill_between = self.fill_between.copy()
                else:
                    fill_between = None
                if fill_between is not None:
                    ax = self._get_ax(k)
                    args = self._process_fill_between(fill_between)
                    ax.fill_between(*args, **fill_between)

    def _make_legend(self):
        """

        :return:
        """
        ax, legend, handle = self._get_ax_legend_handle(self.axes[0])

        handles = []
        labels = []
        title = ""

        if not self.subplots:
            if legend is not None:
                title = legend.get_title().get_text()
                handles.extend(handle)
                labels = [k.get_text() for k in legend.get_texts()]

            if self.legend:
                if self.legend == 'reverse':
                    self.legend_handles = reversed(self.legend_handles)
                    self.legend_labels = reversed(self.legend_labels)

                handles += self.legend_handles
                labels += self.legend_labels

                if self.legend_title is not None:
                    title = self.legend_title

            if len(handles) > 0:
                ax.legend(handles, labels, loc='best', title=title)

        elif self.subplots and self.legend:
            for k, ax in enumerate(self.axes):
                if ax.get_visible() and (self.legend or (is_list_like(self.legend) and self.legend[k])):
                    ax.legend(loc='best')

    @abstractmethod
    def _make_plot(self):
        """

        :return:
        """
        pass

    @classmethod
    def _plot(cls, ax, x, y, style=None, **kwargs):
        """

        :param ax:
        :param x:
        :param y:
        :param style:
        :param kwargs:
        :return:
        """
        if style is not None:
            args = (x, y, style)
        else:
            args = (x, y)
        return ax.plot(*args, **kwargs)

    @staticmethod
    def _process_fill_between(kwargs):
        """

        :param kwargs:
        :return:
        """
        x = kwargs.pop('x', None)
        if x is None:
            raise KeyError("No base for 'fill_between' option supplied")
        if x.ndim > 1:
            if 1 in x.shape:
                x = x.flatten()
            else:
                raise ValueError("Base argument for the 'fill_between' option needs to be a flat array")
        lb = kwargs.pop('lb', None)
        if lb is None:
            raise KeyError("No lower bound for 'fill_between' option supplied")
        if lb.ndim > 1:
            if 1 in lb.shape:
                lb = lb.flatten()
            else:
                raise ValueError("Lower bound argument for the 'fill_between' option needs to be a flat array")
        ub = kwargs.pop('ub', None)
        if ub is None:
            raise KeyError("No upper bound for 'fill_between' option supplied")
        if ub.ndim > 1:
            if 1 in ub.shape:
                ub = ub.flatten()
            else:
                raise ValueError("Upper bound argument for the 'fill_between' option needs to be a flat array")

        if x.size == lb.size == ub.size:
            kwargs['y2'] = lb
            for kw in _FILL_BETWEEN_KWARGS:
                if kw in kwargs:
                    kwargs[_FILL_BETWEEN_KWARGS[kw]] = kwargs.pop(kw)
            return x, ub
        else:
            raise ValueError(f"Dimension mismatch between base, lower and upper bound for the 'fill_between' option."
                             f"\nbase: {x.size}\nlower: {lb.size}\nupper: {ub.size}")

    def _setup_subplots(self):
        """

        :return:
        """
        fig_kwargs = {}
        subplot_kwargs = {}
        if self.figsize is not None:
            if isinstance(self.figsize, (list, tuple, set)) and len(self.figsize) <= 2:
                if len(self.figsize) == 1:
                    width = height = self.figsize[0]
                else:
                    width = self.figsize[0]
                    height = self.figsize[1]
                fig_kwargs['figsize'] = (width, height)

            if self.dpi is not None:
                fig_kwargs['dpi'] = self.dpi

        if self.background_fill_color is not None:
            subplot_kwargs['facecolor'] = self.background_fill_color
        fig_kwargs['tight_layout'] = self.tight_layout

        if self.subplots:
            fig_kwargs = {}
            if self.interactive:
                fig_kwargs['interactive'] = True
                fig_kwargs['x_data'] = [data['x'] for data in self.data.values()]
                fig_kwargs['tight_layout'] = False
            fig, axes, sliders = _subplots(n_axes=self.n_plots, layout=self.layout, layout_type=self._layout_type,
                                           **fig_kwargs)
        else:
            if self.ax is None:
                fig = self.plt.figure(**fig_kwargs)
                axes = fig.add_subplot(111, **subplot_kwargs)
            else:
                fig = self.ax.get_figure()
                if self.figsize is not None:
                    fig.set_size_inches(self.figsize)
                axes = self.ax
            sliders = []
            if self.interactive:
                sliders.append(_slider(axes, self.data[0]['x']))

        axes = _flatten(axes)

        self.fig = fig
        self.axes = axes
        self.sliders = sliders

    @property
    def n_plots(self):
        """

        :return:
        """
        return len(self.data)

    @property
    def plt(self):
        """

        :return:
        """
        import matplotlib.pyplot as plt

        return plt

    @property
    def result(self):
        """

        :return:
        """
        if self.subplots:
            if self.layout is not None:
                axes = self.axes.reshape(*self.layout)
            else:
                axes = self.axes
        else:
            axes = self.axes[0]

        if self.anim:
            from matplotlib.animation import FuncAnimation

            return self.plt, self.fig, axes, FuncAnimation
        else:
            return axes

    def draw(self):
        """

        :return:
        """
        if not self.anim:
            plt = self.plt
            plt.draw_if_interactive()
            plt.show()

    def generate(self):
        """

        :return:
        """
        self._setup_subplots()
        self._make_plot()
        self._make_fill_between()
        self._make_legend()
        self._decorate_subplots()


class MultiPlot(BasePlot):
    """"""
    def __init__(self, data, kind, **kwargs):
        """Constructor method"""
        super().__init__(data, **kwargs)

        self.kind = kind
        if len(self.kind) != self.n_plots:
            msg = f"The length of 'kind' must equal the number of columns if using 'kind' of type " \
                  f"'list'.\nlength of kind = {len(self.kind)}\nnumber of columns = {self.n_plots}"
            raise ValueError(msg)

        self._organize_kwargs()

    def _organize_kwargs(self):
        """

        :return:
        """
        supplied_kinds = set()
        for kind in self.kind:
            if is_list_like(kind):
                supplied_kinds.update(kind)
            else:
                supplied_kinds.add(kind)

        self._line_kwargs = {}
        self._step_kwargs = {}
        self._scatter_kwargs = {}

        for kw, val in self.kwargs.items():
            if kw in _LINE_KWARGS:
                if 'line' in supplied_kinds:
                    self._line_kwargs[_LINE_KWARGS[kw]] = val
                if 'dashed' in supplied_kinds:
                    self._line_kwargs[_LINE_KWARGS[kw]] = val
                if 'dotted' in supplied_kinds:
                    self._line_kwargs[_LINE_KWARGS[kw]] = val
                if 'dotdash' in supplied_kinds:
                    self._line_kwargs[_LINE_KWARGS[kw]] = val
                if 'dashdot' in supplied_kinds:
                    self._line_kwargs[_LINE_KWARGS[kw]] = val
                if 'step' in supplied_kinds:
                    self._step_kwargs[_LINE_KWARGS[kw]] = val
            if kw in _STEP_KWARGS:
                if 'step' in supplied_kinds:
                    self._step_kwargs[_STEP_KWARGS[kw]] = val
            if kw in _SCATTER_KWARGS:
                if 'scatter' in supplied_kinds:
                    self._scatter_kwargs[_SCATTER_KWARGS[kw]] = val
            if kw == 'x_range':
                self._line_kwargs[kw] = val
                self._step_kwargs[kw] = val
                self._scatter_kwargs[kw] = val

    def _get_kwargs(self, kind):
        """

        :param kind:
        :return:
        """
        if kind in ['line', 'dashed', 'dotted', 'dotdash', 'dashdot']:
            return self._line_kwargs.copy()
        elif kind == 'step':
            return self._step_kwargs.copy()
        elif kind == 'scatter':
            return self._scatter_kwargs.copy()
        else:
            return self.kwargs.copy()

    def _iter_data(self):
        """

        :return:
        """
        for sub, plots in self.data.items():
            index = plots['x']
            colors = self._get_colors(num_colors=len(plots['y']))
            kind = self.kind[sub]
            zorder = len(kind) * [None]
            # NOTE: Scatter has zorder=1 by default, so it will be drawn below lines. Setting zorder of scatter to 2
            #  and zorder of lines to 1 to be able to see scatters in plots with dense lines.
            if 'line' in kind and 'scatter' in kind:
                for key, value in enumerate(kind):
                    if value == 'line':
                        zorder[key] = 1
                    elif value == 'scatter':
                        zorder[key] = 2
            for k, (key, values) in enumerate(plots['y'].items()):
                yield sub, k, index[key], key, values, colors[k], kind[k], zorder[k]

    def _make_plot(self):
        """

        :return:
        """
        it = self._iter_data()

        for (subplot, k, x, label, y, colors, kind, zorder) in it:
            ax = self._get_ax(subplot)
            kwargs = self._get_kwargs(kind)

            if kind == 'scatter':
                if is_list_like(kwargs['marker']):
                    kwargs['marker'] = kwargs['marker'][subplot][k]
                if is_list_like(kwargs['s']):
                    kwargs['s'] = kwargs['s'][subplot][k]
            kwargs['color'] = colors
            kwargs['legend'] = False
            kwargs['ax'] = ax
            if zorder is not None:
                kwargs['zorder'] = zorder

            data = {
                0: {
                    'x': {label: x},
                    'y': {label: y}
                }
            }

            plot_obj = MultiPlot.return_plot_class(kind)(data, **kwargs)
            plot_obj.generate()

    @staticmethod
    def return_plot_class(arg):
        """

        :param arg:
        :return:
        """
        return {
            'line': LinePlot,
            'dashed': DashedPlot,
            'dotted': DottedPlot,
            'dashdot': DashDotPlot,
            'step': StepPlot,
            'scatter': ScatterPlot
        }[arg]


class LinePlot(BasePlot):
    """"""
    def __init__(self, data, **kwargs):
        """Constructor method"""
        super().__init__(data, **kwargs)

    def _make_plot(self):
        """

        :return:
        """
        plot = self._plot
        it = self._iter_data()

        for (k, x, label, y, color) in it:
            ax = self._get_ax(k)
            kwargs = self.kwargs.copy()

            x_range = kwargs.pop('x_range', None)

            kwargs['color'] = color
            kwargs['label'] = label

            new_lines = plot(ax, x, y, **kwargs)
            self._add_legend_handle(new_lines[0], label)

            if x_range is None:
                lines = _get_all_lines(ax)
                left, right = _get_x_lim(lines)
            else:
                left, right = x_range
            ax.set_xlim(left, right)


class DashedPlot(LinePlot):
    """"""
    def __init__(self, data, **kwargs):
        """Constructor method"""
        super().__init__(data, **kwargs)

        self.kwargs['linestyle'] = 'dashed'


class DottedPlot(LinePlot):
    """"""
    def __init__(self, data, **kwargs):
        """Constructor method"""
        super().__init__(data, **kwargs)

        self.kwargs['linestyle'] = 'dotted'


class DashDotPlot(LinePlot):
    """"""
    def __init__(self, data, **kwargs):
        """Constructor method"""
        super().__init__(data, **kwargs)

        self.kwargs['linestyle'] = 'dashdot'


class StepPlot(BasePlot):
    """"""
    def __init__(self, data, **kwargs):
        """Constructor method"""
        super().__init__(data, **kwargs)

    def _make_plot(self):
        """

        :return:
        """
        plot = self._plot
        it = self._iter_data()

        for (k, x, label, y, color) in it:
            ax = self._get_ax(k)
            kwargs = self.kwargs.copy()

            x_range = kwargs.pop('x_range', None)

            kwargs['color'] = color
            kwargs['label'] = label

            new_lines = plot(ax, x, y, **kwargs)
            self._add_legend_handle(new_lines[0], label)

            if x_range is None:
                lines = _get_all_lines(ax)
                left, right = _get_x_lim(lines)
            else:
                left, right = x_range
            ax.set_xlim(left, right)

    @classmethod
    def _plot(cls, ax, x, y, style=None, **kwargs):
        """

        :param ax:
        :param x:
        :param y:
        :param style:
        :param kwargs:
        :return:
        """
        if style is not None:
            args = (x, y, style)
        else:
            args = (x, y)
        return ax.step(*args, **kwargs)


class ScatterPlot(BasePlot):
    """"""
    def __init__(self, data, marker='x', **kwargs):
        """Constructor method"""
        super().__init__(data, **kwargs)

        self.marker = marker
        if is_list_like(self.marker) and len(self.marker) != self.n_plots:
            msg = f"The length of 'marker' must equal the number of columns if using 'marker' of type " \
                  f"'list'.\nlength of marker = {len(self.marker)}\nnumber of columns = {self.n_plots}"
            raise ValueError(msg)

    def _make_plot(self):
        """

        :return:
        """
        plot = self._plot
        it = self._iter_data()

        for (k, x, label, y, color) in it:
            ax = self._get_ax(k)
            kwargs = self.kwargs.copy()

            x_range = kwargs.pop('x_range', None)

            if is_list_like(self.marker):
                marker = self.marker[k]
            else:
                marker = self.marker
            kwargs['marker'] = marker
            marker_size = kwargs.get('size')
            if marker_size is None:
                marker_size = kwargs.pop('marker_size', None)
            if marker_size is not None:
                if is_list_like(marker_size):
                    kwargs['s'] = marker_size[k]
                else:
                    kwargs['s'] = marker_size

            # TODO: If we additionally supply the marker_color ('c') keyword this will lead to an error during plotting
            kwargs['color'] = color
            kwargs['label'] = label

            new_lines = plot(ax, x, y, **kwargs)
            self._add_legend_handle(new_lines, label)

            if x_range is None:
                lines = _get_all_lines(ax)
                bboxes = _get_all_bboxes(ax)
                left, right = _get_x_lim(lines, bboxes=bboxes)
            else:
                left, right = x_range
            ax.set_xlim(left, right)

    @classmethod
    def _plot(cls, ax, x, y, style=None, **kwargs):
        """

        :param ax:
        :param x:
        :param y:
        :param style:
        :param kwargs:
        :return:
        """
        if style is not None:
            args = (x, y, style)
        else:
            args = (x, y)
        return ax.scatter(*args, **kwargs)


def _flatten(axes):
    """

    :param axes:
    :return:
    """
    if not is_list_like(axes):
        return np.array([axes])
    elif isinstance(axes, np.ndarray):
        return axes.ravel()
    return np.array(axes)


def _get_all_lines(ax):
    """

    :param ax:
    :return:
    """
    lines = ax.get_lines()

    if hasattr(ax, 'right_ax'):
        lines += ax.right_ax.get_lines()

    if hasattr(ax, 'left_ax'):
        lines += ax.left_ax.get_lines()

    return lines


def _get_all_bboxes(ax):
    """

    :param ax:
    :return:
    """
    bboxes = []
    for collection in ax.collections:
        bboxes.append(collection.axes.dataLim)

    return bboxes


def _get_layout(n_plots, layout=None, layout_type='grid'):
    """

    :param n_plots:
    :param layout:
    :param layout_type:
    :return:
    """
    if layout is not None:
        if not isinstance(layout, (list, tuple)) or len(layout) != 2:
            raise ValueError("Layout must be a tuple of (rows, columns)")

        n_rows, n_cols = layout

        def ceil_(x):
            """

            :param x:
            :return:
            """
            return int(ceil(x))

        if n_rows == -1 and n_cols > 0:
            layout = n_rows, n_cols = (ceil_(float(n_plots) / n_cols), n_cols)
        elif n_cols == -1 and n_rows > 0:
            layout = n_rows, n_cols = (n_rows, ceil_(float(n_plots) / n_rows))
        elif n_cols <= 0 and n_rows <= 0:
            msg = "At least one dimension of layout must be positive"
            raise ValueError(msg)

        if n_rows * n_cols < n_plots:
            raise ValueError(f"Layout of {n_rows}x{n_cols} must be larger than required size {n_plots}")

        return layout

    if layout_type == 'single':
        return 1, 1
    if layout_type == 'horizontal':
        return 1, n_plots
    if layout_type == 'vertical':
        return n_plots, 1

    layouts = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2)}
    try:
        return layouts[n_plots]
    except KeyError:
        k = 1
        while k ** 2 < n_plots:
            k += 1

        if (k - 1) * k >= n_plots:
            return (k - 1), k
        else:
            return k, k


def _get_standard_colors(num_colors=None, colormap=None, color_type='default', color=None):
    """

    :param num_colors:
    :param colormap:
    :param color_type:
    :param color:
    :return:
    """
    import matplotlib.pyplot as plt

    if color is None and colormap is not None:
        if isinstance(colormap, str):
            cmap = colormap
            colormap = cm.get_cmap(colormap)
            if colormap is None:
                raise ValueError(f"Colormap {cmap} is not recognized")
        colors = [colormap[k] for k in range(num_colors)]
    elif color is not None:
        if colormap is not None:
            warnings.warn("'color' and 'colormap' cannot be used simultaneously. Using 'color'...")
        colors = (list(color) if is_list_like(color) and not isinstance(color, dict) else color)
    else:
        if color_type == 'default':
            try:
                colors = [c['color'] for c in list(plt.rcParams['axes.prop_cycle'])]
            except KeyError:
                colors = list(plt.rcParams.get('axes.color_cycle', list('bgrcmyk')))
            if isinstance(colors, str):
                colors = list(colors)

            colors = colors[:num_colors]
        elif color_type == 'random':
            def random_color(column):
                """

                :param column:
                :return:
                """
                rs = random_state(column)
                return rs.rand(3).tolist()

            colors = [random_color(num) for num in range(num_colors)]
        else:
            raise ValueError("'color_type' must be either 'default' or 'random'")

    if isinstance(colors, str):
        conv = matplotlib.colors.ColorConverter()

        def _maybe_valid_colors(colors):
            """

            :param colors:
            :return:
            """
            try:
                [conv.to_rgba(c) for c in colors]
                return True
            except ValueError:
                return False

        maybe_single_color = _maybe_valid_colors([colors])
        maybe_color_cycle = _maybe_valid_colors(list(colors))
        if maybe_single_color and maybe_color_cycle and len(colors) > 1:
            hex_color = [c['color'] for c in list(plt.rcParams['axes.prop_cycle'])]
            colors = [hex_color[int(colors[1])]]
        elif maybe_color_cycle:
            colors = [colors]
        else:
            colors = [colors]

    if len(colors) < num_colors:
        try:
            multiple = num_colors // len(colors) - 1
        except ZeroDivisionError:
            raise ValueError("Invalid color argument: ''")
        mod = num_colors % len(colors)

        colors += multiple * colors
        colors += colors[:mod]

    return colors


def _get_x_lim(lines, bboxes=None):
    """

    :param lines:
    :param bboxes:
    :return:
    """
    left, right = np.inf, -np.inf
    for line in lines:
        x = line.get_xdata(orig=False)
        left = min(np.nanmin(x), left)
        right = max(np.nanmax(x), right)
    if bboxes is not None:
        for bbox in bboxes:
            left = min(bbox.xmin, left)
            right = max(bbox.xmax, right)
    return left, right


def _subplots(n_axes=None, squeeze=True, subplot_kwargs=None, layout=None, layout_type='grid', **kwargs):
    """

    :param n_axes:
    :param squeeze:
    :param subplot_kwargs:
    :param layout:
    :param layout_type:
    :param kwargs:
    :return:
    """
    import matplotlib.pyplot as plt

    if subplot_kwargs is None:
        subplot_kwargs = {}

    if not kwargs:
        x_data = None
    else:
        interactive = kwargs.pop('interactive', False)
        if interactive:
            x_data = kwargs.pop('x_data', None)
            if x_data is None:
                raise KeyError("Argument 'x_data' is missing. If interactive flag is set to True, 'x_data' needs to be"
                               " supplied as well")
        else:
            x_data = None

    fig = plt.figure(**kwargs)
    sliders = []
    if x_data is not None:
        plt.subplots_adjust(left=.05, bottom=.1, right=.95, top=.98, wspace=.2, hspace=.2)

    n_rows, n_cols = _get_layout(n_axes, layout=layout, layout_type=layout_type)
    n_plots = n_rows * n_cols

    ax_arr = np.empty(n_plots, dtype=object)

    ax0 = fig.add_subplot(n_rows, n_cols, 1, **subplot_kwargs)
    if x_data is not None:
        sliders.append(_slider(ax0, x_data[0]))

    ax_arr[0] = ax0

    for k in range(1, n_plots):
        kwds = subplot_kwargs.copy()
        if k >= n_axes:
            kwds['sharex'] = None
            kwds['sharey'] = None
        ax = fig.add_subplot(n_rows, n_cols, k + 1, **kwds)
        if x_data is not None and k < n_axes:
            sliders.append(_slider(ax, x_data[k]))
        ax_arr[k] = ax

    if n_axes != n_plots:
        for ax in ax_arr[n_axes:]:
            ax.set_visible(False)

    if squeeze:
        if n_plots == 1:
            axes = ax_arr[0]
        else:
            axes = ax_arr.reshape(n_rows, n_cols).squeeze()
    else:
        axes = ax_arr.reshape(n_rows, n_cols)

    return fig, axes, sliders


def _slider(ax, data, epsilon=1e-8):
    """

    :param ax:
    :param data:
    :param epsilon:
    :return:
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    def x_aligned_axes(y_distance, width, **kwargs):
        """

        :param y_distance:
        :param width:
        :param kwargs:
        :return:
        """
        return plt.axes([ax.get_position().x0, ax.get_position().y0 - y_distance, ax.get_position().width, width],
                        **kwargs)

    ax_slider = x_aligned_axes(.05, .02)

    expected = next(iter(data.values()))
    all_equal = all(
        (value == expected).all() if value.size == expected.size else value == expected for value in data.values())
    if all_equal:
        x_data = expected
    else:
        index = np.argmax(value.size for value in data.values())
        x_data = list(data.values())[index]
    n_data = len(x_data)
    diff = np.diff(x_data)
    unique_diff = np.all(np.abs(diff - diff[0]) < epsilon)

    if unique_diff:
        slider = Slider(ax_slider, '', x_data[0], x_data[-1], valinit=x_data[-1], valstep=diff[0])
    else:
        slider = Slider(ax_slider, '', 0, n_data - 1, valinit=n_data - 1, valstep=1)

    return slider


def _update_slider(fig, ax, slider, data):
    """

    :param fig:
    :param ax:
    :param slider:
    :param data:
    :return:
    """
    x_data = data['x']
    y_data = data['y']

    keys = y_data.keys()
    lines = ax.get_lines()

    def f(val):
        """

        :param val:
        :return:
        """
        args = []
        for key in keys:
            mask = x_data[key] > val
            arg = y_data[key].copy()
            arg[mask] = np.nan
            args.append(arg)
        return args

    def update(val):
        """

        :param val:
        :return:
        """
        args = f(val)
        for k, arg in enumerate(args):
            lines[k].set_ydata(arg)
        fig.canvas.draw_idle()

    slider.on_changed(update)


__all__ = [
    'MultiPlot',
    'LinePlot',
    'StepPlot',
    'DashedPlot',
    'DottedPlot',
    'DashDotPlot',
    'ScatterPlot'
]
