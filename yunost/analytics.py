"""
analytics.py — три настраиваемые визуализации корпуса. Все наблюдения берутся из SQLite.

Связи классов (термины из лекции 2 курса):
  * агрегация (has-a): AnalyticsService получает готовый CorpusRepository снаружи;
  * композиция (part-of): построитель графиков PlotlyCharts сервис создаёт сам, и он живёт
    ровно столько же, сколько сервис.
"""
from collections import defaultdict
from statistics import mean, median
from typing import Literal

import numpy as np

from .charts import PlotlyCharts
from .decorators import timed
from .storage import METRICS, CorpusFilter, CorpusRepository

Metric = Literal['ink_density', 'text_density', 'illo_area_ratio', 'n_illustrations',
                 'whitespace_ratio', 'n_columns', 'line_count', 'raggedness', 'edge_density']


class AnalyticsService:
    """Считает гистограмму, динамику и рассеяние по выборке из базы и строит к ним графики."""

    def __init__(self, repository: CorpusRepository):
        self.repository = repository             # агрегация: база создаётся и живёт отдельно
        self.charts = PlotlyCharts(METRICS)      # композиция: часть сервиса

    def options(self) -> dict:
        return dict(self.repository.summary(),
                    metrics=[dict(key=key, label=label) for key, label in METRICS.items()],
                    uploads_stored=False)

    @staticmethod
    def _counts(rows: list, usable: list) -> dict:
        return dict(storage='sqlite', selected_pages=len(rows), used_pages=len(usable),
                    missing_pages=len(rows) - len(usable))

    @timed
    def histogram(self, metric: Metric, bins: int, selection: CorpusFilter) -> dict:
        if not 2 <= bins <= 60:
            raise ValueError('Число интервалов: 2–60')
        rows = self.repository.select([metric], selection)
        usable = [r for r in rows if r[metric] is not None]
        bars = []
        if usable:
            counts, edges = np.histogram([r[metric] for r in usable], bins=bins)
            bars = [dict(left=float(edges[i]), right=float(edges[i + 1]), count=int(count))
                    for i, count in enumerate(counts)]
        result = dict(self._counts(rows, usable), metric=metric, bins=bins, bars=bars)
        result['figure'] = self.charts.histogram(result)
        return result

    @timed
    def trend(self, metric: Metric, aggregation: str, selection: CorpusFilter) -> dict:
        if aggregation not in {'mean', 'median'}:
            raise ValueError('Агрегация: mean или median')
        rows = self.repository.select([metric], selection)
        usable = [r for r in rows if r[metric] is not None]
        groups = defaultdict(list)
        for row in usable:
            groups[row['source'], row['year']].append(row)
        aggregate = mean if aggregation == 'mean' else median
        series = []
        for source in sorted({r['source'] for r in usable}):
            points = [dict(year=year, value=aggregate([r[metric] for r in group]),
                           pages=len(group), issues=len({r['issue'] for r in group}))
                      for (src, year), group in sorted(groups.items()) if src == source]
            series.append(dict(source=source, points=points))
        result = dict(self._counts(rows, usable), metric=metric, aggregation=aggregation, series=series)
        result['figure'] = self.charts.trend(result)
        return result

    @timed
    def scatter(self, x_metric: Metric, y_metric: Metric, selection: CorpusFilter) -> dict:
        rows = self.repository.select([x_metric, y_metric], selection)
        usable = [r for r in rows if r[x_metric] is not None and r[y_metric] is not None]
        points = [dict(file=r['file'], issue=r['issue'], year=r['year'], source=r['source'],
                       x=r[x_metric], y=r[y_metric]) for r in usable]
        result = dict(self._counts(rows, usable), x_metric=x_metric, y_metric=y_metric, points=points)
        result['figure'] = self.charts.scatter(result)
        return result
