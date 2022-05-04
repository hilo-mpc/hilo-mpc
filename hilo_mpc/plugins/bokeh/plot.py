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

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from math import ceil
import pathlib
import warnings

import bokeh.palettes as cm
import numpy as np

from ...util.util import is_list_like, random_state


_LINE_KWARGS = ['line_alpha', 'line_cap', 'line_color', 'line_dash', 'line_dash_offset', 'line_join', 'line_width']
_STEP_KWARGS = ['step_mode']
_SCATTER_KWARGS = ['marker', 'marker_color', 'marker_size']


class BasePlot(metaclass=ABCMeta):
    """"""
    # TODO: Support for format string
    _layout_type = 'grid'
    _pop_attributes = [
        'axis_label_text_font_size',
        'major_label_text_font_size',
        'background_fill_color',
        'logy',
        'logx',
        'loglog'
    ]
    _attr_defaults = {
        'axis_label_text_font_size': None,
        'major_label_text_font_size': None,
        'background_fill_color': None,
        'logy': False,
        'logx': False,
        'loglog': False
    }

    def __init__(
            self,
            data,
            subplots=False,
            figsize=None,
            legend=False,
            legend_title=None,
            ax=None,
            sources=None,
            title=None,
            xlabel=None,
            ylabel=None,
            colormap=None,
            layout=None,
            fill_between=None,
            interactive=False,
            server=False,
            output_file=None,
            output_notebook=False,
            browser=None,
            return_html=False,
            **kwargs
    ):
        """Constructor method"""
        self.data = data
        if sources is None:
            sources = []
        self.sources = sources

        self.subplots = subplots

        self.figsize = figsize
        self.layout = layout
        self.fill_between = fill_between
        self.interactive = interactive
        self.server = server

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.legend = legend
        self.legend_title = legend_title

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

        self.output_file = output_file
        self.output_notebook = output_notebook
        self.browser = browser
        self.return_html = return_html

        self.kwargs = kwargs

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

        for k in range(self.n_plots):
            ax = self._get_glyph(k)
            if ax.visible:
                if is_list_like(self.xlabel):
                    if self.xlabel[k] is not None:
                        ax.xaxis.axis_label = self.xlabel[k]
                else:
                    if self.xlabel is not None:
                        ax.xaxis.axis_label = self.xlabel

                if is_list_like(self.ylabel):
                    if self.ylabel[k] is not None:
                        ax.yaxis.axis_label = self.ylabel[k]
                else:
                    if self.ylabel is not None:
                        ax.yaxis.axis_label = self.ylabel

                if self.axis_label_text_font_size is not None:
                    ax.yaxis.axis_label_text_font_size = self.axis_label_text_font_size
                    ax.xaxis.axis_label_text_font_size = self.axis_label_text_font_size

                if self.major_label_text_font_size is not None:
                    ax.yaxis.major_label_text_font_size = self.major_label_text_font_size
                    ax.xaxis.major_label_text_font_size = self.major_label_text_font_size

            if self.interactive:
                slider = self._get_glyph(k, get_interactive=True)
                _update_slider(slider, self.sources[k])

        if self.title is not None:
            if self.subplots:
                if is_list_like(self.title):
                    if len(self.title) != self.n_plots:
                        msg = f"The length of 'title' must equal the number of columns if using 'title' of type " \
                              f"'list' and 'subplots=True'.\nlength of title = {len(self.title)}\nnumber of columns =" \
                              f" {self.n_plots}"
                        raise ValueError(msg)

                    for k in range(self.n_plots):
                        ax = self._get_glyph(k)
                        ax.title.text = self.title[k]
                else:
                    warnings.warn("An overall title for all subplots is currently not supported")
            else:
                if is_list_like(self.title):
                    msg = "Using 'title' of type 'list' is not supported unless 'subplots=True' is passed."
                    raise ValueError(msg)
                if self.interactive:
                    # NOTE: At the moment the first entry will be the Figure and the second one the Slider
                    self.axes.children[0].title.text = self.title
                else:
                    self.axes.title.text = self.title

    def _get_colors(self, num_colors=None, color_kwargs='color', subplot=None):
        """

        :param num_colors:
        :param color_kwargs:
        :param subplot:
        :return:
        """
        if num_colors is None:
            num_colors = self.n_plots

        if subplot is None:
            color = self.kwargs.get(color_kwargs)
        else:
            color = self.kwargs.get(color_kwargs)[subplot]

        return _get_standard_colors(num_colors=num_colors, colormap=self.colormap, color=color)

    def _get_glyph(self, k, get_interactive=False):
        """

        :param k:
        :param get_interactive:
        :return:
        """
        if self.subplots:
            ax = self.axes.children
            if len(ax) == 2 and self.n_plots != 2:
                ax = ax[1].children[k][0]
            elif len(ax) >= self.n_plots:
                ax = ax[k]
            else:
                raise ValueError("Number of desired plots is not equal to the number of created plots")
        else:
            ax = self.axes

        if self.interactive:
            if not get_interactive:
                ax = ax.children[0]
            else:
                ax = ax.children[1]
                return ax

        ax.yaxis.visible = True
        return ax

    def _iter_data(self):
        """

        :return:
        """
        for sub, plots in self.data.items():
            if len(self.sources) < self.n_plots:
                source_data = {}
                for val in plots['x'].values():
                    if not source_data:
                        source_data['x'] = val
                    else:
                        if val != source_data['x']:
                            raise ValueError
                source_data.update({'y' + str(k): val for k, val in enumerate(plots['y'].values())})
                source = self.source(source_data)
                self.sources.append(source)
            elif len(self.sources) == self.n_plots:
                source = self.sources[sub]
            else:
                raise RuntimeError("length of ColumnDataSource list > number of plots")
            if 'line_color' in self.kwargs:
                colors = self._get_colors(num_colors=len(plots['y']), color_kwargs='line_color')
            elif 'marker_color' in self.kwargs:
                colors = self._get_colors(num_colors=len(plots['y']), color_kwargs='marker_color')
            else:
                colors = self._get_colors(num_colors=len(plots['y']))
            keys = list(plots['y'])
            for k, key in enumerate(keys):
                if isinstance(colors, dict):
                    color = colors[key]
                else:
                    color = colors[keys.index(key)]
                yield sub, k, key, source, color

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
                ax = self._get_glyph(0)
                self._process_fill_between(fill_between)
                ax.patch(**fill_between)
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
                    ax = self._get_glyph(k)
                    self._process_fill_between(fill_between)
                    ax.patch(**fill_between)

    def _make_legend(self):
        """

        :return:
        """
        if not self.subplots:
            if is_list_like(self.legend):
                legend = self.legend[0]
            else:
                legend = self.legend
            if not legend:
                self.axes.legend.visible = False
            else:
                if self.legend_title is not None:
                    if is_list_like(self.legend_title):
                        msg = "Using 'legend_title' of type 'list' is not supported unless 'subplots=True' is passed."
                        raise ValueError(msg)
                    self.axes.legend.title = self.legend_title
        else:
            if is_list_like(self.legend):
                if len(self.legend) != self.n_plots:
                    msg = f"The length of 'legend' must equal the number of columns if using 'legend' of type " \
                          f"'list'.\nlength of legend = {len(self.legend)}\nnumber of columns = {self.n_plots}"
                    raise ValueError(msg)
            if is_list_like(self.legend_title):
                if len(self.legend_title) != self.n_plots:
                    msg = f"The length of 'legend_title' must equal the number of columns if using 'legend_title'" \
                          f" of type 'list' and 'legend=True'.\nlength of legend_title = {len(self.legend_title)}" \
                          f"\nnumber of columns = {self.n_plots}"
                    raise ValueError(msg)
            for k in range(self.n_plots):
                ax = self._get_glyph(k)
                if not self.legend or (is_list_like(self.legend) and not self.legend[k]):
                    ax.legend.visible = False
                else:
                    if is_list_like(self.legend_title):
                        ax.legend.title = self.legend_title[k]
                    else:
                        ax.legend.title = self.legend_title

    @abstractmethod
    def _make_plot(self):
        """

        :return:
        """
        pass

    def _process_fill_between(self, kwargs):
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
            x_lb = x
            x_ub = x_lb[::-1]
            source = self.source({'x': np.hstack((x_lb, x_ub)), 'y': np.hstack((lb, ub[::-1]))})
            kwargs['x'] = 'x'
            kwargs['y'] = 'y'
            kwargs['source'] = source
            if 'label' in kwargs:
                kwargs['legend_label'] = kwargs.pop('label')
        else:
            raise ValueError(f"Dimension mismatch between base, lower and upper bound for the 'fill_between' option."
                             f"\nbase: {x.size}\nlower: {lb.size}\nupper: {ub.size}")

    def _setup_subplots(self):
        """

        :return:
        """
        fig_kwargs = {}
        if self.figsize is not None:
            if self.background_fill_color is not None:
                fig_kwargs['background_fill_color'] = self.background_fill_color

            if isinstance(self.figsize, (list, tuple, set)) and len(self.figsize) <= 2:
                if len(self.figsize) == 1:
                    width = height = self.figsize[0]
                else:
                    width = self.figsize[0]
                    height = self.figsize[1]
                fig_kwargs['plot_width'] = width
                fig_kwargs['plot_height'] = height

        # TODO: Initialize slider here already and set properties later
        if self.subplots:
            # TODO: Add support for self.ax
            if self.interactive:
                fig_kwargs['interactive'] = True
                fig_kwargs['x_data'] = [data['x'] for data in self.data.values()]
            fig = _subplots(n_axes=self.n_plots, subplot_kwargs=fig_kwargs, layout=self.layout,
                            layout_type=self._layout_type)
        else:
            if self.ax is None:
                fig = self.plt.figure(**fig_kwargs)
            else:
                fig = self.ax
            if self.interactive:
                fig = self.column(fig, _slider(self.data[0]['x']))

        self.axes = fig

    @property
    def column(self):
        """

        :return:
        """
        from bokeh.layouts import column

        return column

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
        import bokeh.plotting as plt

        return plt

    @property
    def result(self):
        """

        :return:
        """
        if self.server:
            from bokeh.models import Button
            from bokeh.server.server import Server
            # from bokeh.themes import Theme

            # path = pathlib.Path(__file__)
            # file = str(path.parent.absolute()) + '/theme.yaml'

            return self.column(Button(), self.axes), self.sources, Server, None  # Theme(filename=file)
        elif self.return_html:
            from bokeh.resources import CDN
            from bokeh.embed import file_html

            return file_html(self.axes, CDN)
        else:
            if self.output_file is not None:
                return self.output_file
            else:
                from bokeh.io.util import detect_current_filename

                return detect_current_filename()

    @property
    def source(self):
        """

        :return:
        """
        from bokeh.models import ColumnDataSource

        return ColumnDataSource

    def draw(self):
        """

        :return:
        """
        if not self.server:
            plt = self.plt
            if self.output_notebook:
                plt.output_notebook()
            else:
                if self.output_file is not None:
                    path = pathlib.Path(self.output_file).parent.absolute()
                    if not path.is_dir():
                        path.mkdir(parents=True, exist_ok=True)
                    plt.output_file(self.output_file)
            plt.show(self.axes, browser=self.browser)

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
                    self._line_kwargs[kw] = val
                if 'dashed' in supplied_kinds:
                    self._line_kwargs[kw] = val
                if 'dotted' in supplied_kinds:
                    self._line_kwargs[kw] = val
                if 'dotdash' in supplied_kinds:
                    self._line_kwargs[kw] = val
                if 'dashdot' in supplied_kinds:
                    self._line_kwargs[kw] = val
                if 'step' in supplied_kinds:
                    self._step_kwargs[kw] = val
            if kw in _STEP_KWARGS:
                if 'step' in supplied_kinds:
                    if kw == 'step_mode':
                        self._step_kwargs['mode'] = val
                    else:
                        self._step_kwargs[kw] = val
            if kw in _SCATTER_KWARGS:
                if 'scatter' in supplied_kinds:
                    if kw == 'marker_size':
                        self._scatter_kwargs['size'] = val
                    else:
                        self._scatter_kwargs[kw] = val
            if kw == 'x_range':
                self._line_kwargs[kw] = val
                self._step_kwargs[kw] = val
                self._scatter_kwargs[kw] = val

    def _get_kwargs(self, kind):
        """

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
            if len(self.sources) < self.n_plots:
                sources = []
                if len(plots['x']) == len(plots['y']) == len(self.kind[sub]):
                    for key, val in plots['x'].items():
                        source_data = {}
                        source_data['x'] = val
                        source_data['y0'] = plots['y'][key]
                        source = self.source(source_data)
                        sources.append(source)
                elif len(plots['x']) == 1 and len(plots['y']) == len(self.kind[sub]):
                    raise NotImplementedError
                elif len(plots['x']) == len(plots['y']) and len(self.kind[sub]) == 1:
                    raise NotImplementedError
                elif len(plots['x']) == len(self.kind[sub]) == 1:
                    raise NotImplementedError
                else:
                    raise ValueError("Dimension mismatch between the number of plot elements for both axes and the "
                                     "supplied plot types.")
                self.sources.append(sources)
            elif len(self.sources) == self.n_plots:
                sources = self.sources[sub]
            else:
                raise RuntimeError("length of ColumnDataSource list > number of plots")

            colors = self._get_colors(num_colors=len(plots['y']))
            keys = list(plots['y'])  # TODO: Check if len(plots['y']) and len(self.kind[sub]) are the same. The code
            # TODO: needs to be adjusted if this is not the case.
            for k, kind in enumerate(self.kind[sub]):
                yield sub, k, [keys[k]], sources[k], colors[k], kind

    def _make_plot(self):
        """

        :return:
        """
        it = self._iter_data()

        for (subplot, k, labels, source, colors, kind) in it:
            ax = self._get_glyph(subplot)
            kwargs = self._get_kwargs(kind)

            if kind == 'scatter':
                if is_list_like(kwargs['marker']):
                    kwargs['marker'] = kwargs['marker'][subplot][k]
                if is_list_like(kwargs['size']):
                    kwargs['size'] = kwargs['size'][subplot][k]
            # NOTE: Going with list(color) for now since processing string representation is not yet implemented
            kwargs['color'] = colors
            kwargs['legend'] = True
            kwargs['ax'] = ax
            kwargs['sources'] = [source]

            data = {
                0: {
                    'x': [],
                    'y': {label: [] for label in labels}
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
            'dotdash': DotDashPlot,
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
        it = self._iter_data()

        for (subplot, k, label, source, color) in it:
            ax = self._get_glyph(subplot)
            kwargs = self.kwargs.copy()

            x_range = kwargs.pop('x_range', None)

            kwargs['line_color'] = color
            kwargs['legend_label'] = label
            kwargs['source'] = source

            ax.line(x='x', y='y' + str(k), **kwargs)

            if x_range is None:
                left, right = _get_x_lim(ax.renderers)
            else:
                left, right = x_range
            ax.x_range.start = left
            ax.x_range.end = right


class DashedPlot(LinePlot):
    """"""
    def __init__(self, data, **kwargs):
        """Constructor method"""
        super().__init__(data, **kwargs)

        self.kwargs['line_dash'] = 'dashed'


class DottedPlot(LinePlot):
    """"""
    def __init__(self, data, **kwargs):
        """Constructor method"""
        super().__init__(data, **kwargs)

        self.kwargs['line_dash'] = 'dotted'


class DotDashPlot(LinePlot):
    """"""
    def __init__(self, data, **kwargs):
        """Constructor method"""
        super().__init__(data, **kwargs)

        self.kwargs['line_dash'] = 'dotdash'


class DashDotPlot(LinePlot):
    """"""
    def __init__(self, data, **kwargs):
        """Constructor method"""
        super().__init__(data, **kwargs)

        self.kwargs['line_dash'] = 'dashdot'


class StepPlot(BasePlot):
    """"""
    def __init__(self, data, **kwargs):
        """Constructor method"""
        super().__init__(data, **kwargs)

    def _make_plot(self):
        """

        :return:
        """
        it = self._iter_data()

        for (subplot, k, label, source, color) in it:
            ax = self._get_glyph(subplot)
            kwargs = self.kwargs.copy()

            x_range = kwargs.pop('x_range', None)

            kwargs['line_color'] = color
            kwargs['legend_label'] = label
            kwargs['source'] = source

            ax.step(x='x', y='y' + str(k), **kwargs)

            if x_range is None:
                left, right = _get_x_lim(ax.renderers)
            else:
                left, right = x_range
            ax.x_range.start = left
            ax.x_range.end = right


class ScatterPlot(BasePlot):
    """"""
    def __init__(self, data, marker='x', **kwargs):
        """Constructor class"""
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
        it = self._iter_data()

        for (subplot, k, label, source, color) in it:
            ax = self._get_glyph(subplot)
            kwargs = self.kwargs.copy()

            x_range = kwargs.pop('x_range', None)

            kwargs['color'] = color
            kwargs['legend_label'] = label
            kwargs['source'] = source

            # NOTE: k should actually always be zero at the moment, since we only put one plottable in
            #  ColumnDataSource to better handle different dimension
            if is_list_like(self.marker):
                marker = self.marker[subplot]
                if is_list_like(marker):
                    marker = marker[k]
            else:
                marker = self.marker
            marker_size = kwargs.get('size')
            if marker_size is None:
                marker_size = kwargs.pop('marker_size', None)
            if marker_size is not None:
                if is_list_like(marker_size):
                    marker_size = marker_size[subplot]
                    if is_list_like(marker_size):
                        kwargs['size'] = marker_size[k]
                    else:
                        kwargs['size'] = marker_size

            self._plot(ax, marker)(x='x', y='y' + str(k), **kwargs)

            if x_range is None:
                left, right = _get_x_lim(ax.renderers)
            else:
                left, right = x_range
            ax.x_range.start = left
            ax.x_range.end = right

    @staticmethod
    def _plot(ax, marker):
        """

        :param ax:
        :param marker:
        :return:
        """
        return {
            '*': ax.asterisk,
            'o': ax.circle,
            'o+': ax.circle_cross,
            'o.': ax.circle_dot,
            'ox': ax.circle_x,
            'oy': ax.circle_y,
            '+': ax.cross,
            '-': ax.dash,
            'd': ax.diamond,
            'd+': ax.diamond_cross,
            'd.': ax.diamond_dot,
            '.': ax.dot,
            'h': ax.hex,
            'h.': ax.hex_dot,
            'it': ax.inverted_triangle,
            'pl': ax.plus,
            'sq': ax.square,
            'sq+': ax.square_cross,
            'sq.': ax.square_dot,
            'sqp': ax.square_pin,
            'sqx': ax.square_x,
            's': ax.star,
            's.': ax.star_dot,
            'tr': ax.triangle,
            'tr.': ax.triangle_dot,
            'trp': ax.triangle_pin,
            'x': ax.x,
            'y': ax.y
        }[marker]


def _convert_color(c):
    """

    :param c:
    :return:
    """
    if c not in 'bgrcmykw':
        return c
    return {
        'b': 'blue',
        'g': 'green',
        'r': 'red',
        'c': 'cyan',
        'm': 'magenta',
        'y': 'yellow',
        'k': 'black',
        'w': 'white'
    }[c]


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
    if color is None and colormap is not None:
        if isinstance(colormap, str):
            cmap = colormap
            if 'Category' in colormap and not colormap.isdigit():
                # TODO: Maybe we could combine all Category20's if we have '<= 60' colors?
                if num_colors <= 10:
                    colormap += '10'
                else:
                    colormap += '20'
            if colormap in cm.all_palettes:
                colormap = cm.all_palettes[colormap]
            else:
                raise ValueError(f"Colormap {cmap} is not recognized")
        last = list(colormap)[-1]
        colors = [colormap[last][k] for k in range(num_colors)]
    elif color is not None:
        if colormap is not None:
            warnings.warn("'color' and 'colormap' cannot be used simultaneously. Using 'color'...")
        colors = (list(color) if is_list_like(color) and not isinstance(color, dict) else color)
    else:
        if color_type == 'default':
            if num_colors <= 10:
                colors = cm.Category10_10
            else:
                colors = cm.Category20_20

            colors = colors[:num_colors]
        elif color_type == 'random':
            def random_color(column):
                """

                :param column:
                :return:
                """
                rs = random_state(column)
                return rs.rand(3).to_list()

            colors = [random_color(k) for k in range(num_colors)]
        else:
            raise ValueError("'color_type' must be either 'default' or 'random'")

    if isinstance(colors, str):
        from bokeh.colors import named

        def _maybe_valid_colors(colors):
            """

            :param colors:
            :return:
            """
            try:
                [getattr(named, _convert_color(c)) for c in colors]
                return True
            except AttributeError:
                return False

        maybe_single_color = _maybe_valid_colors([colors])
        maybe_color_cycle = _maybe_valid_colors(list(colors))
        if maybe_single_color and maybe_color_cycle and len(colors) > 1:
            # NOTE: Some HEX stuff
            raise NotImplementedError
        elif maybe_color_cycle:
            if len(colors) > 1:
                colors = [_convert_color(c) for c in colors]
            else:
                colors = [_convert_color(colors)]
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


def _get_x_lim(lines):
    """

    :param lines:
    :return:
    """
    left, right = np.inf, -np.inf
    for line in lines:
        x = line.data_source.data['x']
        left = min(np.nanmin(x), left)
        right = max(np.nanmax(x), right)
    return left, right


def _subplots(n_axes=None, subplot_kwargs=None, layout=None, layout_type='grid', **kwargs):
    """

    :param n_axes:
    :param subplot_kwargs:
    :param layout:
    :param layout_type:
    :param kwargs:
    :return:
    """
    from bokeh.plotting import figure
    from bokeh.layouts import column, row, gridplot

    if subplot_kwargs is None:
        subplot_kwargs = {}
        x_data = None
    else:
        interactive = subplot_kwargs.pop('interactive', False)
        if interactive:
            x_data = subplot_kwargs.pop('x_data', None)
            if x_data is None:
                raise KeyError("Argument 'x_data' is missing. If interactive flag is set to True, 'x_data' needs to be"
                               " supplied as well")
        else:
            x_data = None

    n_rows, n_cols = _get_layout(n_axes, layout=layout, layout_type=layout_type)
    n_plots = n_rows * n_cols

    fig = figure(**subplot_kwargs)
    if x_data is not None:
        fig = column(fig, _slider(x_data[0]))

    if n_rows == 1 and n_cols > 1:
        figs = [fig]
        for k in range(1, n_plots):
            if k < n_axes:
                fig = figure(**subplot_kwargs)
                if x_data is not None:  # Means, that interactive flag was set to True
                    fig = column(fig, _slider(x_data[k]))
                figs.append(fig)
        return row(*figs, **kwargs)
    elif n_cols == 1 and n_rows > 1:
        figs = [fig]
        for k in range(1, n_plots):
            if k < n_axes:
                fig = figure(**subplot_kwargs)
                if x_data is not None:  # Means, that interactive flag was set to True
                    fig = column(fig, _slider(x_data[k]))
                figs.append(fig)
        return column(*figs, **kwargs)
    elif n_rows > 1 and n_cols > 1:
        grid = []
        for i in range(n_rows):
            row = []
            for j in range(n_cols):
                if i + j == 0:
                    row.append(fig)
                    continue
                if i * n_cols + j < n_axes:
                    fig = figure(**subplot_kwargs)
                    if x_data is not None:  # Means, that interactive flag was set to True
                        fig = column(fig, _slider(x_data[i * n_cols + j]))
                    row.append(fig)
                else:
                    row.append(None)
            grid.append(row)
        return gridplot(grid, **kwargs)
    else:
        return fig


def _slider(data: dict[str, np.ndarray], epsilon: float = 1e-8):
    """

    :param data:
    :param epsilon:
    :return:
    """
    from bokeh.models import Slider

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
        slider = Slider(start=x_data[0], end=x_data[-1], value=x_data[-1], step=diff[0])
    else:
        slider = Slider(start=0, end=n_data - 1, value=n_data - 1, step=1)

    return slider


def _update_slider(slider, sources):
    """

    :param slider:
    :param sources:
    :return:
    """
    from bokeh.models import ColumnDataSource, CustomJS

    full_sources = [ColumnDataSource(source.data) for source in sources]

    n_y = len(sources)
    n_data = [source.data['x'].size for source in sources]

    callback = CustomJS(args=dict(source=sources, full_source=full_sources, n_y=n_y, n_data=n_data), code="""
        var val = cb_obj.value;
        for (var i=0; i<n_y; i++) {
            var data = source[i].data;
            var full_data = full_source[i].data;
            var x = data['x'];
            var y = data['y0'];
            var y_full = full_data['y0'];
            for (var j=0; j<n_data[i]; j++) {
                if (x[j]>val) {
                    y[j] = Number.NaN;
                } else {
                    y[j] = y_full[j];
                }
            }
            source[i].change.emit();
        }
    """)

    slider.js_on_change('value', callback)


__all__ = [
    'MultiPlot',
    'LinePlot',
    'StepPlot',
    'DashedPlot',
    'DottedPlot',
    'DotDashPlot',
    'DashDotPlot',
    'ScatterPlot'
]
