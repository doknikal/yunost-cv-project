"""
charts.py — три графика аналитики на plotly (как в лекции 4 курса: go.Figure, go.Bar, go.Scatter).

Фигура строится в Python из уже посчитанных по SQLite чисел и уходит в браузер как JSON;
там её рисует локальный plotly.js (без интернета). Сами числа считает AnalyticsService.
"""
import json
from typing import Dict, List

import plotly.graph_objects as go

from .decorators import timed

SOURCE_LABELS = {'scan': 'Сканы', 'photo': 'Фотографии'}
SOURCE_COLORS = {'scan': '#d33852', 'photo': '#6586ad'}
BAR_COLOR = '#b5142a'


class PlotlyCharts:
    """Адаптер: словарь с данными графика -> фигура plotly в виде JSON."""

    def __init__(self, labels: Dict[str, str]):
        self.labels = labels                     # подписи признаков: raggedness -> «Рваность правого края»

    @staticmethod
    def _layout(figure: go.Figure, x_title: str, y_title: str) -> dict:
        figure.update_layout(
            xaxis_title=x_title, yaxis_title=y_title,
            margin=dict(t=30, b=50, l=60, r=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', y=1.08, x=0), hoverlabel=dict(namelength=-1),
            separators=', ',                     # русская запись чисел: 0,25 и 1 000
        )
        figure.update_xaxes(gridcolor='rgba(128,128,128,0.2)', zeroline=False)
        figure.update_yaxes(gridcolor='rgba(128,128,128,0.2)', zeroline=False)
        return json.loads(figure.to_json())

    @timed
    def histogram(self, data: dict) -> dict:
        bars: List[dict] = data['bars']
        figure = go.Figure(go.Bar(
            x=[(b['left'] + b['right']) / 2 for b in bars],
            y=[b['count'] for b in bars],
            width=[b['right'] - b['left'] for b in bars],
            customdata=[[b['left'], b['right']] for b in bars],
            hovertemplate='%{customdata[0]:.4g} — %{customdata[1]:.4g}<br>страниц: %{y}<extra></extra>',
            marker=dict(color=BAR_COLOR, opacity=0.75, line=dict(color=BAR_COLOR, width=1)),
            name='Страниц',
        ))
        figure.update_layout(bargap=0, showlegend=False)
        return self._layout(figure, self.labels[data['metric']], 'Число страниц')

    @timed
    def trend(self, data: dict) -> dict:
        word = 'среднее' if data['aggregation'] == 'mean' else 'медиана'
        figure = go.Figure()
        for series in data['series']:
            points = series['points']
            figure.add_trace(go.Scatter(
                x=[p['year'] for p in points], y=[p['value'] for p in points],
                customdata=[[p['pages'], p['issues']] for p in points],
                mode='lines+markers', name=SOURCE_LABELS[series['source']],
                line=dict(color=SOURCE_COLORS[series['source']], width=2), marker=dict(size=9),
                hovertemplate=f'%{{x}}: {word} %{{y:.4g}}<br>страниц: %{{customdata[0]}}, '
                              'номеров: %{customdata[1]}<extra>%{fullData.name}</extra>',
            ))
        figure.update_xaxes(dtick=1, tickformat='d')
        return self._layout(figure, 'Год', self.labels[data['metric']])

    @timed
    def scatter(self, data: dict) -> dict:
        figure = go.Figure()
        for source in ('scan', 'photo'):
            points = [p for p in data['points'] if p['source'] == source]
            if not points:
                continue
            figure.add_trace(go.Scatter(
                x=[p['x'] for p in points], y=[p['y'] for p in points],
                text=[p['file'] for p in points], customdata=[p['year'] for p in points],
                mode='markers', name=SOURCE_LABELS[source],
                marker=dict(color=SOURCE_COLORS[source], size=8, opacity=0.8,
                            line=dict(color='white', width=1)),
                hovertemplate='%{text} (%{customdata})<br>x: %{x:.4g}<br>y: %{y:.4g}'
                              '<extra>%{fullData.name}</extra>',
            ))
        return self._layout(figure, self.labels[data['x_metric']], self.labels[data['y_metric']])
