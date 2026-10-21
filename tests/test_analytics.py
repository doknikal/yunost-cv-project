"""Coursework: actual SQL persistence, repeatable imports and adjustable charts."""
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence
import csv
import hashlib
import io
from pathlib import Path
import sqlite3
from statistics import mean, median

from fastapi.testclient import TestClient
import numpy as np
import pytest

import server
from yunost.analytics import AnalyticsService
from yunost.layout import LayoutBlock, LayoutWord
from yunost.storage import APPLICATION_ID, METRICS, CorpusFilter, CorpusRepository
from yunost.text import TextPageResult

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/manifest.csv'
FEATURES = ROOT / 'data/features.csv'


def rewrite(source, target, change):
    rows = list(csv.DictReader(io.StringIO(source.read_text())))
    fields = list(rows[0])
    change(rows)
    with target.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    return target


@pytest.fixture
def repository(tmp_path):
    repo = CorpusRepository(tmp_path / 'corpus.sqlite3')
    assert repo.import_csv(MANIFEST, FEATURES) == 111
    return repo


def test_csv_round_trip_is_exact_and_upsert_is_idempotent(repository):
    before = repository.select(METRICS)
    expected = {r['file']: r for r in csv.DictReader(io.StringIO(FEATURES.read_text()))}
    assert len(before) == 111
    for row in before:
        original = expected[row['file']]
        for field in ('issue', 'source'):
            assert row[field] == original[field]
        for field in ('year', 'month', 'page'):
            assert row[field] == int(original[field])
        for field in METRICS:
            assert row[field] == (float(original[field]) if original[field] else None)
    assert sum(r['raggedness'] is None for r in before) == 1
    assert repository.import_csv(MANIFEST, FEATURES) == 111
    assert repository.select(METRICS) == before
    assert repository.summary() == {
        'storage': 'sqlite', 'pages': 111, 'issues': 15,
        'years': [1957, 1962, 1963, 1965, 1967],
        'sources': [{'source': 'photo', 'pages': 36}, {'source': 'scan', 'pages': 75}]}
    with sqlite3.connect(repository.path) as db:
        assert db.execute('PRAGMA application_id').fetchone()[0] == APPLICATION_ID
        assert db.execute('PRAGMA user_version').fetchone()[0] == 1
        assert dict(db.execute('SELECT * FROM metadata')) == {
            'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            'features_sha256': hashlib.sha256(FEATURES.read_bytes()).hexdigest()}
        assert not list(db.execute('PRAGMA foreign_key_check'))
        assert 'orig_path' not in [r[1] for r in db.execute('PRAGMA table_info(pages)')]


def test_upsert_updates_and_reads_survive_reopening(repository, tmp_path):
    features = rewrite(FEATURES, tmp_path / 'updated.csv', lambda rows: rows[0].update(line_count='999'))
    repository.import_csv(MANIFEST, features)
    reopened = CorpusRepository(repository.path)
    assert reopened.summary()['pages'] == 111
    assert reopened.select(['line_count'])[0]['line_count'] == 999


def test_after_seed_no_csv_is_needed(repository, monkeypatch):
    def forbidden(*args):
        raise AssertionError('CSV was reopened instead of querying SQLite')
    monkeypatch.setattr(CorpusRepository, '_csv', forbidden)
    repository.ensure_seeded('/missing/manifest.csv', '/missing/features.csv')
    service = AnalyticsService(CorpusRepository(repository.path))
    assert service.options()['pages'] == 111
    assert service.histogram('line_count', 8, CorpusFilter())['used_pages'] == 111
    assert service.trend('line_count', 'mean', CorpusFilter())['used_pages'] == 111
    assert service.scatter('line_count', 'raggedness', CorpusFilter())['used_pages'] == 110


@pytest.mark.parametrize('mutation', [
    lambda r: r.append(r[0].copy()),
    lambda r: r.pop(),
    lambda r: r[0].update(year='1958'),
    lambda r: r[0].update(line_count='1.5'),
    lambda r: r[0].update(line_count='-1'),
    lambda r: r[0].update(raggedness='nan'),
    lambda r: r[0].update(raggedness='inf'),
    lambda r: r[0].update(raggedness='not-a-number'),
])
def test_bad_import_does_not_change_existing_rows(repository, tmp_path, mutation):
    before = repository.select(METRICS)
    path = tmp_path / 'bad.csv'
    rewrite(FEATURES, path, mutation)
    with pytest.raises(ValueError):
        repository.import_csv(MANIFEST, path)
    assert repository.select(METRICS) == before


def test_missing_metric_column_is_reported(repository, tmp_path):
    path = tmp_path / 'missing.csv'
    lines = FEATURES.read_text().splitlines()
    index = lines[0].split(',').index('raggedness')
    path.write_text('\n'.join(','.join(v for i, v in enumerate(line.split(',')) if i != index) for line in lines))
    with pytest.raises(ValueError, match='отсутствует признак'):
        repository.import_csv(MANIFEST, path)


def test_sql_failure_rolls_back_entire_import(repository, tmp_path):
    before = repository.select(METRICS)
    with sqlite3.connect(repository.path) as db:
        db.execute("CREATE TRIGGER reject_features BEFORE UPDATE ON features "
                   "WHEN NEW.line_count=999 BEGIN SELECT RAISE(ABORT, 'fixture failure'); END")
    path = rewrite(FEATURES, tmp_path / 'updated.csv', lambda rows: rows[-1].update(line_count='999'))
    with pytest.raises(sqlite3.IntegrityError, match='fixture failure'):
        repository.import_csv(MANIFEST, path)
    assert repository.select(METRICS) == before


@pytest.mark.parametrize('statement', [
    'CREATE TABLE private_data (value TEXT)', 'CREATE VIEW private_data AS SELECT 1',
    'PRAGMA application_id = 123', 'PRAGMA user_version = 2'])
def test_unrelated_database_is_not_modified(tmp_path, statement):
    path = tmp_path / 'foreign.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute(statement)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='не база'):
        CorpusRepository(path)
    assert path.read_bytes() == before


def test_future_schema_is_not_modified(repository):
    with sqlite3.connect(repository.path) as db:
        db.execute('PRAGMA user_version = 999')
    before = repository.path.read_bytes()
    with pytest.raises(ValueError, match='версия'):
        CorpusRepository(repository.path)
    assert repository.path.read_bytes() == before


def test_parallel_first_use_imports_once_without_duplicates(tmp_path):
    path = tmp_path / 'concurrent.sqlite3'
    def first_use(_):
        repo = CorpusRepository(path)
        repo.ensure_seeded(MANIFEST, FEATURES)
        return len(repo.select(['line_count']))
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert list(pool.map(first_use, range(12))) == [111] * 12
    assert CorpusRepository(path).summary()['issues'] == 15


@pytest.mark.parametrize('kwargs', [{'year_min': 2000, 'year_max': 1900}, {'year_min': 1799},
                                   {'year_max': 2201}, {'source': "scan' OR 1=1 --"}])
def test_filters_are_validated(kwargs):
    with pytest.raises(ValueError):
        CorpusFilter(**kwargs)


def test_sql_identifiers_are_allowlisted(repository):
    with pytest.raises(ValueError):
        repository.select(['line_count; DROP TABLE pages; --'])
    with pytest.raises(ValueError):
        repository.select([])
    assert repository.summary()['pages'] == 111


@pytest.mark.parametrize('metric', list(METRICS))
def test_all_charts_match_sql_and_independent_aggregations(repository, metric):
    service = AnalyticsService(repository)
    rows = repository.select([metric])
    values = [r[metric] for r in rows if r[metric] is not None]
    histogram = service.histogram(metric, 7, CorpusFilter())
    counts, edges = np.histogram(values, bins=7)
    assert [b['count'] for b in histogram['bars']] == counts.tolist()
    assert [b['left'] for b in histogram['bars']] == edges[:-1].tolist()
    assert sum(b['count'] for b in histogram['bars']) == len(values)
    assert histogram['missing_pages'] == 111 - len(values)
    for mode, function in [('mean', mean), ('median', median)]:
        trend = service.trend(metric, mode, CorpusFilter())
        assert sum(p['pages'] for s in trend['series'] for p in s['points']) == len(values)
        for series in trend['series']:
            for point in series['points']:
                group = [r for r in rows if r['source'] == series['source'] and
                         r['year'] == point['year'] and r[metric] is not None]
                assert point['value'] == function(r[metric] for r in group)
                assert point['issues'] == len({r['issue'] for r in group})
    scatter = service.scatter(metric, metric, CorpusFilter())
    assert len(scatter['points']) == len(values)
    assert all(p['x'] == p['y'] for p in scatter['points'])


def test_source_year_filters_and_empty_selection(repository):
    service = AnalyticsService(repository)
    photo = CorpusFilter(1962, 1965, 'photo')
    assert len(repository.select(['line_count'], photo)) == 36
    assert service.histogram('line_count', 2, photo)['used_pages'] == 36
    points = service.scatter('line_count', 'raggedness', photo)['points']
    assert all(p['source'] == 'photo' and 1962 <= p['year'] <= 1965 for p in points)
    empty = CorpusFilter(1957, 1957, 'photo')
    assert service.histogram('line_count', 2, empty)['bars'] == []
    assert service.trend('line_count', 'median', empty)['series'] == []
    assert service.scatter('line_count', 'raggedness', empty)['points'] == []


@pytest.fixture
def client(repository, monkeypatch):
    service = AnalyticsService(repository)
    monkeypatch.setattr(server, 'get_analytics', lambda: service)
    return TestClient(server.app)


def test_analytics_page_assets_and_api(client):
    for path in ['/analytics', '/static/analytics.js', '/static/analytics.css', '/vendor/plotly.min.js']:
        assert client.get(path).status_code == 200
    options = client.get('/api/analytics/options').json()
    assert options['pages'] == 111 and options['uploads_stored'] is False
    assert len(options['metrics']) == 9
    for kind in ('histogram', 'trend', 'scatter'):
        result = client.get('/api/analytics/' + kind)
        assert result.status_code == 200
        assert result.json()['used_pages'] == 110
        assert result.json()['missing_pages'] == 1
    assert client.get('/api/analytics/histogram?metric=line_count&bins=23&source=photo').json()['used_pages'] == 36
    assert client.get('/api/analytics/trend?aggregation=mean&source=scan').json()['series'][0]['source'] == 'scan'
    empty = client.get('/api/analytics/scatter?year_min=1958&year_max=1960').json()
    assert empty['selected_pages'] == 0


@pytest.mark.parametrize('path', [
    'histogram?bins=1', 'histogram?bins=61', 'histogram?bins=2.5', 'histogram?metric=unknown',
    'trend?aggregation=sum', 'scatter?x_metric=unknown', 'scatter?y_metric=unknown',
    'histogram?source=other', 'trend?year_min=2000&year_max=1900', 'scatter?year_min=1799',
    'trend?year_max=2201', 'histogram?year_min=abc'])
def test_api_rejects_invalid_parameters(client, path):
    assert client.get('/api/analytics/' + path).status_code == 422


def test_database_failure_does_not_disable_ocr_landing(tmp_path, monkeypatch):
    path = tmp_path / 'foreign.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE private_data (value TEXT)')
    monkeypatch.setattr(server, '_analytics', None)
    monkeypatch.setenv('YUNOST_DB_PATH', str(path))
    client = TestClient(server.app)
    assert client.get('/api/analytics/options').status_code == 503
    assert client.get('/').status_code == 200
    assert client.post('/api/ocr', files={'file': ('empty.png', b'')}).status_code == 400


def test_three_special_methods_and_live_response():
    def block(text, rejected=''):
        result = LayoutBlock([LayoutWord(text, 10, 10, 80, 12, 95, (1, 1, 1))])
        result.rejected = rejected
        return result
    one, debris, two = block('Проза'), block('мусор', 'fixture'), block('Стихи')
    result = TextPageResult([one, debris, two], np.full((200, 200), 255, np.uint8))
    assert isinstance(result, Sequence)
    for name in ('__len__', '__iter__', '__getitem__'):
        assert getattr(TextPageResult, name) is not getattr(Sequence, name)
    assert len(result) == 2
    assert list(result) == [one, two]
    assert result[0] is one and result[-1] is two
    assert result[:] == [one, two]
    with pytest.raises(IndexError):
        _ = result[2]
    response = server.page_response(result)
    assert response['text_blocks'] == 2 and response['other_blocks'] == 1
    assert response['blocks'] == [one.to_dict(), two.to_dict()]
    assert response['first_block'] == one.to_dict()
    empty = TextPageResult([debris], result.gray)
    assert len(empty) == 0 and list(empty) == [] and empty[:] == []
    assert server.page_response(empty)['first_block'] is None


def test_plotly_figures_match_numbers(repository):
    """Графики строятся на plotly из тех же чисел, что и таблицы."""
    service = AnalyticsService(repository)
    histogram = service.histogram('line_count', 9, CorpusFilter())
    [bars] = histogram['figure']['data']
    assert bars['type'] == 'bar' and len(bars['y']) == 9 and sum(bars['y']) == histogram['used_pages']
    trend = service.trend('raggedness', 'median', CorpusFilter())
    assert [t['name'] for t in trend['figure']['data']] == ['Фотографии', 'Сканы']
    assert all(t['mode'] == 'lines+markers' for t in trend['figure']['data'])
    scatter = service.scatter('n_columns', 'raggedness', CorpusFilter(source='photo'))
    assert sum(len(t['x']) for t in scatter['figure']['data']) == scatter['used_pages'] == 35  # одна фото-страница без рваности
    assert scatter['figure']['layout']['xaxis']['title']['text'] == 'Число колонок'
    empty = service.histogram('line_count', 4, CorpusFilter(1958, 1960))
    assert empty['figure']['data'][0]['y'] == []
