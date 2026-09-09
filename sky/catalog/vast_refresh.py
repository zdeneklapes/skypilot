"""Refresh the locally managed Vast catalog when credentials are available."""

import csv
import os
from pathlib import Path
import tempfile
import time

import filelock

from sky import sky_logging
from sky.catalog import common as catalog_common
from sky.catalog.data_fetchers import fetch_vast

CATALOG_FILENAME = 'vast/vms.csv'
DEFAULT_MAX_AGE_SECONDS = 20 * 60
_CREDENTIAL_PATH = '~/.config/vastai/vast_api_key'
_REQUIRED_COLUMNS = {
    'InstanceType',
    'AcceleratorName',
    'AcceleratorCount',
    'vCPUs',
    'MemoryGiB',
    'GpuInfo',
    'Price',
    'SpotPrice',
    'Region',
}

logger = sky_logging.init_logger(__name__)


def has_credentials() -> bool:
    """Return whether the Vast credential file permits a local refresh."""
    return Path(os.path.expanduser(_CREDENTIAL_PATH)).is_file()


def validate_catalog(path: Path) -> None:
    """Validate the CSV columns and ensure at least one usable GPU row."""
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        missing_columns = _REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing_columns:
            missing = ', '.join(sorted(missing_columns))
            raise ValueError(
                f'Vast catalog is missing required columns: {missing}')

        for row in reader:
            try:
                accelerator_count = float(row['AcceleratorCount'])
            except (TypeError, ValueError):
                continue
            if (row.get('AcceleratorName') and accelerator_count > 0 and
                    row.get('GpuInfo')):
                return
    raise ValueError('Vast catalog does not contain usable GPU entries')


def _count_catalog_records(path: Path) -> int:
    """Return the number of data rows in one generated Vast catalog CSV."""
    with path.open(encoding='utf-8', newline='') as stream:
        return sum(1 for _ in csv.DictReader(stream))


def catalog_is_fresh(target: Path) -> bool:
    """Return whether a recent, validated local catalog can be reused."""
    max_age_seconds = int(
        os.environ.get('VAST_CATALOG_MAX_AGE_SECONDS', DEFAULT_MAX_AGE_SECONDS))
    if max_age_seconds <= 0 or not target.is_file():
        return False
    age_seconds = max(0.0, time.time() - target.stat().st_mtime)
    if age_seconds > max_age_seconds:
        return False
    try:
        validate_catalog(target)
    except Exception:  # pylint: disable=broad-except
        return False
    logger.info(
        'Vast catalog refresh skipped: path=%s age_seconds=%.0f records=%d '
        '(catalog is fresh).', target, age_seconds,
        _count_catalog_records(target))
    return True


def refresh_catalog(force: bool = False) -> bool:
    """Fetch, validate, and atomically install the current Vast catalog.

    A refresh is intentionally disabled unless the Vast credential file is
    available. If a provider call fails, a previously validated CSV remains in
    place and continues to serve catalog queries.

    Args:
        force: Refresh even when the current catalog is still within its
            configured maximum age.
    """
    if not has_credentials():
        logger.info('Vast catalog refresh skipped: Vast credentials are '
                    'unavailable.')
        return False

    target = Path(catalog_common.get_catalog_path(CATALOG_FILENAME))
    target.parent.mkdir(parents=True, exist_ok=True)
    with filelock.FileLock(str(target) + '.refresh.lock'):
        if not force and catalog_is_fresh(target):
            return True

        file_descriptor, staged_name = tempfile.mkstemp(prefix='.vast-vms-',
                                                        suffix='.csv',
                                                        dir=target.parent)
        os.close(file_descriptor)
        staged = Path(staged_name)
        try:
            records_before = (_count_catalog_records(target)
                              if target.is_file() else 0)
            fetched_records = fetch_vast.fetch_vast_catalog()
            fetch_vast.save_catalog(fetched_records, str(staged))
            validate_catalog(staged)
            records_after = _count_catalog_records(staged)
            records_fetched = len(fetched_records)
            if target.is_file() and staged.read_bytes() == target.read_bytes():
                logger.info(
                    'Vast catalog fetched but CSV is unchanged: path=%s '
                    'records_before=%d records_fetched=%d records_after=%d.',
                    target, records_before, records_fetched, records_after)
                return True
            os.replace(staged, target)
            logger.info(
                'Vast catalog CSV updated: path=%s records_before=%d '
                'records_fetched=%d records_after=%d records_delta=%+d.',
                target, records_before, records_fetched, records_after,
                records_after - records_before)
            return True
        except Exception:  # pylint: disable=broad-except
            if target.is_file():
                try:
                    validate_catalog(target)
                except Exception:  # pylint: disable=broad-except
                    pass
                else:
                    logger.warning(
                        'Vast catalog refresh failed; using the validated '
                        'existing CSV: path=%s records=%d.', target,
                        _count_catalog_records(target))
                    return True
            raise RuntimeError(
                'Vast catalog refresh failed and no valid existing catalog '
                'is available') from None
        finally:
            staged.unlink(missing_ok=True)
