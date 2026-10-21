"""Three parameterized visualizations; all observations come from SQLite queries."""
from collections import defaultdict
from statistics import mean, median
from typing import Literal

import numpy as np

from .decorators import timed
from .storage import METRICS

Metric = Literal['ink_density', 'text_density', 'illo_area_ratio', 'n_illustrations',
                 'whitespace_ratio', 'n_columns', 'line_count', 'raggedness', 'edge_density']


class AnalyticsService:
    def __init__(self, repository):
        self.repository = repository

    def options(self):
        return dict(self.repository.summary(),
                    metrics=[dict(key=key, label=label) for key, label in METRICS.items()],
                    uploads_stored=False)

    @staticmethod
    def _counts(rows, usable):
        return dict(storage='sqlite', selected_pages=len(rows), used_pages=len(usable),
                    missing_pages=len(rows) - len(usable))

    @timed
    def histogram(self, metric, bins, selection):
        if not 2 <= bins <= 60:
            raise ValueError('Число интервалов: 2–60')
        rows = self.repository.select([metric], selection)
        usable = [r for r in rows if r[metric] is not None]
        bars = []
        if usable:
            counts, edges = np.histogram([r[metric] for r in usable], bins=bins)
            bars = [dict(left=float(edges[i]), right=float(edges[i + 1]), count=int(count))
                    for i, count in enumerate(counts)]
        return dict(self._counts(rows, usable), metric=metric, bins=bins, bars=bars)

    @timed
    def trend(self, metric, aggregation, selection):
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
        return dict(self._counts(rows, usable), metric=metric, aggregation=aggregation, series=series)

    @timed
    def scatter(self, x_metric, y_metric, selection):
        rows = self.repository.select([x_metric, y_metric], selection)
        usable = [r for r in rows if r[x_metric] is not None and r[y_metric] is not None]
        points = [dict(file=r['file'], issue=r['issue'], year=r['year'], source=r['source'],
                       x=r[x_metric], y=r[y_metric]) for r in usable]
        return dict(self._counts(rows, usable), x_metric=x_metric, y_metric=y_metric, points=points)
