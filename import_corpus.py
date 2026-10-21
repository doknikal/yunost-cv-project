"""Explicit, repeatable CSV -> SQLite import, independent of the OCR engine."""
import argparse
from pathlib import Path

from yunost.storage import CorpusRepository, ROOT, default_database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=default_database())
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/manifest.csv')
    parser.add_argument('--features', type=Path, default=ROOT / 'data/features.csv')
    args = parser.parse_args()
    repository = CorpusRepository(args.database)
    imported = repository.import_csv(args.manifest, args.features)
    print(f'Импортировано/обновлено страниц: {imported}; база: {repository.path}')
    print(repository.summary())


if __name__ == '__main__':
    main()
