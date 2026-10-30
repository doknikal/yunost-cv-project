"""SQLite storage for the research corpus, never for users' OCR uploads."""
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
import io
import math
import os
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
APPLICATION_ID = 0x59554E4F  # YUNO: do not silently modify an unrelated database.
SCHEMA_VERSION = 1
METRICS = {
    'ink_density': 'Плотность чернил',
    'text_density': 'Плотность текста',
    'illo_area_ratio': 'Доля площади иллюстраций',
    'n_illustrations': 'Число иллюстраций',
    'whitespace_ratio': 'Доля полей',
    'n_columns': 'Число колонок',
    'line_count': 'Число строк',
    'raggedness': 'Рваность правого края',
    'edge_density': 'Плотность контуров',
}
INTEGER_METRICS = {'n_illustrations', 'n_columns', 'line_count'}


def default_database():
    return Path(os.environ.get('YUNOST_DB_PATH', str(ROOT / 'data/corpus.sqlite3'))).expanduser()


@dataclass(frozen=True)
class CorpusFilter:
    year_min: int = 1800
    year_max: int = 2200
    source: str = 'all'

    def __post_init__(self):
        if not 1800 <= self.year_min <= self.year_max <= 2200:
            raise ValueError('Начальный год должен быть не больше конечного (1800–2200)')
        if self.source not in {'all', 'scan', 'photo'}:
            raise ValueError('Источник: all, scan или photo')


class CorpusRepository:
    """One short-lived connection per operation: safe for FastAPI worker threads."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(str(self.path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys = ON')
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            application = db.execute('PRAGMA application_id').fetchone()[0]
            version = db.execute('PRAGMA user_version').fetchone()[0]
            objects = db.execute("SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall()
            if application != APPLICATION_ID and (application != 0 or objects or version):
                raise ValueError('Это не база корпуса «Юности». Существующие данные не изменены.')
            if version not in (0, SCHEMA_VERSION):
                raise ValueError('Неподдерживаемая версия базы; существующие данные не изменены.')
            db.execute('CREATE TABLE IF NOT EXISTS issues ('
                       'id TEXT PRIMARY KEY, year INTEGER NOT NULL, month INTEGER NOT NULL '
                       'CHECK(month BETWEEN 1 AND 12))')
            db.execute('CREATE TABLE IF NOT EXISTS pages ('
                       'file TEXT PRIMARY KEY, issue_id TEXT NOT NULL REFERENCES issues(id), '
                       'page INTEGER NOT NULL CHECK(page > 0), '
                       "source TEXT NOT NULL CHECK(source IN ('scan', 'photo')), "
                       'UNIQUE(issue_id, page))')
            columns = ', '.join(f'{name} {"INTEGER" if name in INTEGER_METRICS else "REAL"}'
                                for name in METRICS)
            db.execute('CREATE TABLE IF NOT EXISTS features ('
                       'page_file TEXT PRIMARY KEY REFERENCES pages(file), ' + columns + ')')
            db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            db.execute(f'PRAGMA application_id = {APPLICATION_ID}')
            db.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')

    @staticmethod
    def _csv(path):
        content = Path(path).read_bytes()
        rows = list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'), newline='')))
        required = {'file', 'issue', 'year', 'month', 'page', 'source'}
        if any(not required.issubset(row) or any(row[k] is None for k in required) for row in rows):
            raise ValueError('В CSV отсутствуют обязательные метаданные')
        if not rows or any(not r.get('file') for r in rows):
            raise ValueError('В CSV нет страниц или имен файлов')
        if len({r['file'] for r in rows}) != len(rows):
            raise ValueError('Повторяющиеся страницы в CSV')
        return {r['file']: r for r in rows}, hashlib.sha256(content).hexdigest()

    def import_csv(self, manifest_path, features_path, only_if_empty=False):
        """Validate both files first, then atomically upsert; blank metrics become NULL."""
        manifest, manifest_hash = self._csv(manifest_path)
        features, features_hash = self._csv(features_path)
        if manifest.keys() != features.keys():
            raise ValueError('Наборы страниц в manifest.csv и features.csv различаются')
        issues, pages, values = {}, [], []
        for name, row in manifest.items():
            year, month, page = (int(row[k]) for k in ('year', 'month', 'page'))
            source, issue = row['source'], row['issue']
            if (not 1800 <= year <= 2200 or not 1 <= month <= 12 or page < 1
                    or source not in {'scan', 'photo'} or issue != f'{year:04d}-{month:02d}'
                    or '/' in name or '\\' in name):
                raise ValueError('Некорректные метаданные страницы: ' + name)
            if any(row[k] != features[name][k] for k in ('issue', 'year', 'month', 'page', 'source')):
                raise ValueError('Метаданные CSV не совпадают: ' + name)
            numbers = []
            for metric in METRICS:
                if features[name].get(metric) is None:
                    raise ValueError('В CSV отсутствует признак ' + metric)
                raw = features[name][metric].strip()
                number = float(raw) if raw else None
                if number is not None:
                    if not math.isfinite(number) or number < 0:
                        raise ValueError('Некорректное значение ' + metric)
                    if metric in INTEGER_METRICS:
                        if not number.is_integer():
                            raise ValueError('Ожидалось целое значение ' + metric)
                        number = int(number)
                numbers.append(number)
            issues[issue] = (issue, year, month)
            pages.append((name, issue, page, source))
            values.append((name, *numbers))
        if len({(p[1], p[2]) for p in pages}) != len(pages):
            raise ValueError('Повторяющийся номер страницы внутри выпуска')
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            if only_if_empty and db.execute('SELECT COUNT(*) FROM pages').fetchone()[0]:
                return 0
            db.executemany('INSERT INTO issues VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE '
                           'SET year=excluded.year, month=excluded.month', issues.values())
            db.executemany('INSERT INTO pages VALUES (?, ?, ?, ?) ON CONFLICT(file) DO UPDATE '
                           'SET issue_id=excluded.issue_id, page=excluded.page, source=excluded.source', pages)
            names = ', '.join(METRICS)
            placeholders = ', '.join('?' for _ in range(len(METRICS) + 1))
            updates = ', '.join(f'{m}=excluded.{m}' for m in METRICS)
            db.executemany(f'INSERT INTO features (page_file, {names}) VALUES ({placeholders}) '
                           f'ON CONFLICT(page_file) DO UPDATE SET {updates}', values)
            for key, digest in [('manifest_sha256', manifest_hash), ('features_sha256', features_hash)]:
                db.execute('INSERT INTO metadata VALUES (?, ?) ON CONFLICT(key) DO UPDATE '
                           'SET value=excluded.value', (key, digest))
        return len(pages)

    def ensure_seeded(self, manifest_path, features_path):
        # Once imported, normal operation does not open either CSV again.
        if self.summary()['pages'] == 0:
            self.import_csv(manifest_path, features_path, only_if_empty=True)
        with self._connection() as db:
            missing = db.execute('SELECT COUNT(*) FROM pages p LEFT JOIN features f '
                                 'ON f.page_file=p.file WHERE f.page_file IS NULL').fetchone()[0]
        if missing:
            raise ValueError('В базе есть страницы без признаков. Повторите импорт корпуса.')

    def summary(self):
        with self._connection() as db:
            pages = db.execute('SELECT COUNT(*) FROM pages').fetchone()[0]
            issues = db.execute('SELECT COUNT(*) FROM issues').fetchone()[0]
            years = [r[0] for r in db.execute('SELECT DISTINCT year FROM issues ORDER BY year')]
            sources = [dict(r) for r in db.execute('SELECT source, COUNT(*) AS pages '
                                                   'FROM pages GROUP BY source ORDER BY source')]
        return dict(storage='sqlite', pages=pages, issues=issues, years=years, sources=sources)

    def select(self, metrics, selection=CorpusFilter()):
        metrics = list(dict.fromkeys(metrics))
        if not metrics or any(m not in METRICS for m in metrics):
            raise ValueError('Неизвестный признак')
        columns = ', '.join('f.' + m for m in metrics)  # identifiers only from the allowlist
        sql = ('SELECT p.file, p.issue_id AS issue, i.year, i.month, p.page, p.source, ' + columns +
               ' FROM pages p JOIN issues i ON i.id=p.issue_id JOIN features f ON f.page_file=p.file '
               'WHERE i.year BETWEEN ? AND ?')
        parameters = [selection.year_min, selection.year_max]
        if selection.source != 'all':
            sql += ' AND p.source=?'
            parameters.append(selection.source)
        sql += ' ORDER BY i.year, i.month, p.page, p.file'
        with self._connection() as db:
            return [dict(row) for row in db.execute(sql, parameters)]
