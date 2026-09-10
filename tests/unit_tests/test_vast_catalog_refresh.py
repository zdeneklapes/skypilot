"""Tests for the locally refreshed Vast catalog."""

import ast
import csv
import importlib
import logging
import os
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from sky.catalog import common
from sky.catalog import vast_catalog
from sky.catalog import vast_refresh
from sky.catalog.data_fetchers import fetch_vast
from sky.utils import annotations

_CATALOG_FIELDS = [
    'InstanceType',
    'AcceleratorName',
    'AcceleratorCount',
    'vCPUs',
    'MemoryGiB',
    'GpuInfo',
    'Price',
    'SpotPrice',
    'Region',
    'HostingType',
]


def _write_catalog(path: Path,
                   *,
                   include_hosting_type: bool = True,
                   row_count: int = 1,
                   instance_type: str = 'vastv2-1x-A100-81920-4-8192') -> None:
    fields = _CATALOG_FIELDS if include_hosting_type else _CATALOG_FIELDS[:-1]
    row = {
        'InstanceType': instance_type,
        'AcceleratorName': 'A100',
        'AcceleratorCount': '1',
        'vCPUs': '4',
        'MemoryGiB': '8',
        'GpuInfo': "{'Gpus': [{'MemoryInfo': {'SizeInMiB': 81920}}]}",
        'Price': '0.8',
        'SpotPrice': '0.8',
        'Region': 'any',
        'HostingType': '1',
    }
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(row_count):
            row_for_index = dict(row)
            if index > 0:
                row_for_index['Region'] = f'City {index}, FR, EU'
            writer.writerow({field: row_for_index[field] for field in fields})


def _sdk_offer(**overrides):
    """Return one Vast SDK 1.5.0-shaped offer for catalog tests."""
    offer = {
        'gpu_name': 'A100',
        'num_gpus': 1,
        'cpu_cores': 4,
        'cpu_ram': 8192,
        'gpu_total_ram': 81920,
        'dph_total': .8,
        'min_bid': .4,
        'geolocation': 'Prague, CZ, EU',
        'hosting_type': 1,
    }
    offer.update(overrides)
    return offer


@pytest.fixture(autouse=True)
def clear_request_catalog_cache():
    """Isolate request-scoped catalog snapshots between refresh tests."""
    annotations.clear_request_level_cache()
    yield
    annotations.clear_request_level_cache()


def test_catalog_loader_uses_local_common_catalog(monkeypatch):
    """Vast catalog reads the local common cache instead of hosted CSV text."""
    calls = []

    def read_catalog(filename: str):
        calls.append(filename)
        return pd.DataFrame([{
            'InstanceType': '1x-A100-4-8192',
            'AcceleratorName': 'A100',
            'AcceleratorCount': 1,
            'vCPUs': 4,
            'MemoryGiB': 8,
            'GpuInfo': 'gpu-info',
            'Price': .8,
            'SpotPrice': .8,
            'Region': 'any',
        }])

    monkeypatch.setattr(common, 'read_catalog', read_catalog)
    importlib.reload(vast_catalog)

    assert calls == ['vast/vms.csv']
    assert vast_catalog._catalog_df().iloc[0]['AcceleratorName'] == 'A100'
    monkeypatch.undo()
    importlib.reload(vast_catalog)


def test_catalog_loader_rejects_missing_required_columns(monkeypatch):
    """A malformed local catalog cannot silently enter resource selection."""
    monkeypatch.setattr(vast_catalog, '_df',
                        pd.DataFrame([{
                            'InstanceType': 'example'
                        }]))

    with pytest.raises(common.CatalogFetchError,
                       match='missing required columns'):
        vast_catalog._catalog_df()


def test_datacenter_filter_fails_closed_without_hosting_type():
    """Datacenter-only requests must not admit unknown hosting types."""
    df = pd.DataFrame([{field: '1' for field in _CATALOG_FIELDS[:-1]}])

    assert vast_catalog._apply_datacenter_filter(df, datacenter_only=True).empty


def test_fetch_vast_catalog_and_save_catalog_are_reusable(
        monkeypatch, tmp_path):
    """Catalog refresh keeps raw CPU/RAM and disables SDK hidden defaults."""
    offer = _sdk_offer(geolocation='any')
    client = mock.Mock(spec=['search_offers'])
    client.search_offers.return_value = [offer, offer]
    monkeypatch.setattr(fetch_vast.vast, 'vast', lambda: client)

    catalog_path = tmp_path / 'vms.csv'
    fetch_vast.save_catalog(fetch_vast.fetch_vast_catalog(), str(catalog_path))

    vast_refresh.validate_catalog(catalog_path)
    search_kwargs = client.search_offers.call_args.kwargs
    assert search_kwargs['query'] == (
        'verified=true rentable=true rented=false external=false '
        'georegion=true inet_down>=100 disk_space>=80')
    assert search_kwargs['no_default'] is True
    assert search_kwargs['type'] == 'on-demand'
    assert search_kwargs['order'] == 'dph_total'
    assert search_kwargs['storage'] == 80


def test_fetch_vast_catalog_accepts_nullable_sdk_optional_fields(monkeypatch):
    """Missing nullable SDK fields get conservative catalog defaults."""
    offer = _sdk_offer(min_bid=None, geolocation=None, hosting_type=None)
    del offer['hosting_type']
    client = mock.Mock(spec=['search_offers'])
    client.search_offers.return_value = [offer]
    monkeypatch.setattr(fetch_vast.vast, 'vast', lambda: client)

    rows = fetch_vast.fetch_vast_catalog()

    assert len(rows) == 1
    assert rows[0]['Price'] == '0.80'
    assert rows[0]['SpotPrice'] == '0.80'
    assert rows[0]['Region'] == 'any'
    assert rows[0]['HostingType'] == 0


def test_fetch_vast_catalog_skips_only_malformed_offers(monkeypatch, caplog):
    """Malformed SDK offers are counted without discarding valid peers."""
    offers = [
        _sdk_offer(),
        _sdk_offer(cpu_ram=None),
        _sdk_offer(dph_total='not-a-price'),
    ]
    client = mock.Mock(spec=['search_offers'])
    client.search_offers.return_value = offers
    monkeypatch.setattr(fetch_vast.vast, 'vast', lambda: client)

    with caplog.at_level(logging.WARNING, logger=fetch_vast.__name__):
        rows = fetch_vast.fetch_vast_catalog()

    assert len(rows) == 1
    assert 'rejected=2' in caplog.text
    assert 'invalid_shape=1' in caplog.text
    assert 'invalid_price=1' in caplog.text


def test_fetch_vast_catalog_fails_when_every_offer_is_malformed(monkeypatch):
    """A refresh fails when provider output has no usable catalog row."""
    client = mock.Mock(spec=['search_offers'])
    client.search_offers.return_value = [_sdk_offer(cpu_cores=None)]
    monkeypatch.setattr(fetch_vast.vast, 'vast', lambda: client)

    with pytest.raises(ValueError, match='no usable offers.*invalid_shape=1'):
        fetch_vast.fetch_vast_catalog()


def test_catalog_instance_type_uses_shared_concrete_offer_builder(monkeypatch):
    """Catalog and live admission encode concrete Vast shapes identically."""
    builder = mock.Mock(return_value='vastv2-shared')
    monkeypatch.setattr(fetch_vast.vast, 'build_instance_type_from_offer',
                        builder)
    offer = {
        'gpu_name': 'A100',
        'num_gpus': 1,
        'cpu_cores': 16,
        'cpu_ram': 49152,
    }

    assert fetch_vast.create_instance_type(offer, 81920) == 'vastv2-shared'
    builder.assert_called_once_with({**offer, 'gpu_ram': 81920})


def test_fetch_vast_catalog_keeps_countries_distinct(monkeypatch):
    """Asian and European country rows never collapse into continent buckets."""
    shared_offer = _sdk_offer()
    offers = [{
        **shared_offer, 'geolocation': region
    } for region in ('Jiangsu, CN, AS', 'Japan, JP, AS', 'France, FR, EU')]
    client = type('Client', (),
                  {'search_offers': lambda _self, **_kwargs: offers})()
    monkeypatch.setattr(fetch_vast.vast, 'vast', lambda: client)

    rows = fetch_vast.fetch_vast_catalog()

    assert {row['Region'] for row in rows} == {
        'Jiangsu, CN, AS',
        'Japan, JP, AS',
        'France, FR, EU',
    }


def test_fetch_vast_catalog_preserves_per_gpu_memory_identity(monkeypatch):
    """A100 40GB and 80GB offers must be distinct durable catalog resources."""
    shared_offer = _sdk_offer(gpu_name='A100 SXM4', cpu_cores=32, cpu_ram=65536)
    offers = [{
        **shared_offer,
        'num_gpus': 1,
        'gpu_total_ram': 40960,
        'geolocation': 'Georgia, US, NA',
    }, {
        **shared_offer,
        'num_gpus': 1,
        'gpu_total_ram': 81920,
        'geolocation': 'Prague, CZ, EU',
    }, {
        **shared_offer,
        'num_gpus': 2,
        'gpu_total_ram': 163840,
        'geolocation': 'Prague, CZ, EU',
    }]
    client = type('Client', (),
                  {'search_offers': lambda _self, **_kwargs: offers})()
    monkeypatch.setattr(fetch_vast.vast, 'vast', lambda: client)

    rows = fetch_vast.fetch_vast_catalog()

    rows_by_instance_type = {row['InstanceType']: row for row in rows}
    assert set(rows_by_instance_type) == {
        'vastv2-1x-A100_SXM4-40960-32-65536',
        'vastv2-1x-A100_SXM4-81920-32-65536',
        'vastv2-2x-A100_SXM4-81920-32-65536',
    }
    assert rows_by_instance_type['vastv2-1x-A100_SXM4-81920-32-65536'][
        'AcceleratorName'] == ('A100-80GB')
    gpu_info = ast.literal_eval(
        rows_by_instance_type['vastv2-2x-A100_SXM4-81920-32-65536']['GpuInfo'])
    assert gpu_info['Gpus'][0]['MemoryInfo']['SizeInMiB'] == 81920
    assert gpu_info['TotalGpuMemoryInMiB'] == 163840


def test_refresh_catalog_replaces_validated_staged_file(monkeypatch, tmp_path):
    """A successful Vast refresh atomically installs only validated output."""
    catalog_path = tmp_path / 'vast' / 'vms.csv'
    monkeypatch.setattr(vast_refresh.catalog_common, 'get_catalog_path',
                        lambda _name: str(catalog_path))
    monkeypatch.setattr(vast_refresh, 'has_credentials', lambda: True)
    monkeypatch.setattr(fetch_vast, 'fetch_vast_catalog', lambda: [{}])
    monkeypatch.setattr(fetch_vast, 'save_catalog',
                        lambda _rows, output: _write_catalog(Path(output)))

    assert vast_refresh.refresh_catalog()
    vast_refresh.validate_catalog(catalog_path)


def test_recent_legacy_catalog_is_regenerated(monkeypatch, tmp_path):
    """A fresh legacy-only cache must be regenerated into v2 identities."""
    catalog_path = tmp_path / 'vast' / 'vms.csv'
    catalog_path.parent.mkdir()
    _write_catalog(catalog_path, instance_type='1x-A100-4-8192')
    monkeypatch.setattr(vast_refresh.catalog_common, 'get_catalog_path',
                        lambda _name: str(catalog_path))
    monkeypatch.setattr(vast_refresh, 'has_credentials', lambda: True)
    fetch_catalog = mock.Mock(return_value=[{}])
    monkeypatch.setattr(fetch_vast, 'fetch_vast_catalog', fetch_catalog)
    monkeypatch.setattr(fetch_vast, 'save_catalog',
                        lambda _rows, output: _write_catalog(Path(output)))

    with pytest.raises(ValueError, match='supported vastv2'):
        vast_refresh.validate_catalog(catalog_path)
    assert not vast_refresh.catalog_is_fresh(catalog_path)
    assert vast_refresh.refresh_catalog()
    fetch_catalog.assert_called_once_with()
    vast_refresh.validate_catalog(catalog_path)


def test_refresh_catalog_logs_fetched_unchanged_record_count(
        monkeypatch, tmp_path, caplog):
    """A fetched-but-identical response logs that no CSV update occurred."""
    catalog_path = tmp_path / 'vast' / 'vms.csv'
    catalog_path.parent.mkdir()
    _write_catalog(catalog_path)
    os.utime(catalog_path, ns=(1, 1))
    original_mtime = catalog_path.stat().st_mtime_ns
    monkeypatch.setattr(vast_refresh.catalog_common, 'get_catalog_path',
                        lambda _name: str(catalog_path))
    monkeypatch.setattr(vast_refresh, 'has_credentials', lambda: True)
    monkeypatch.setattr(fetch_vast, 'fetch_vast_catalog', lambda: [{}])
    monkeypatch.setattr(fetch_vast, 'save_catalog',
                        lambda _rows, output: _write_catalog(Path(output)))
    replace_catalog = mock.Mock(wraps=vast_refresh.os.replace)
    monkeypatch.setattr(vast_refresh.os, 'replace', replace_catalog)

    refresh_logger = logging.getLogger('sky.catalog.vast_refresh')
    refresh_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger=refresh_logger.name):
            assert vast_refresh.refresh_catalog(force=True)
    finally:
        refresh_logger.removeHandler(caplog.handler)

    assert 'Vast catalog fetched but CSV is unchanged' in caplog.text
    assert 'records_before=1' in caplog.text
    assert 'records_fetched=1' in caplog.text
    replace_catalog.assert_called_once()
    assert catalog_path.stat().st_mtime_ns > original_mtime
    assert vast_refresh.catalog_is_fresh(catalog_path)


def test_refresh_catalog_logs_updated_record_counts(monkeypatch, tmp_path,
                                                    caplog):
    """A changed provider response logs the old and new CSV record counts."""
    catalog_path = tmp_path / 'vast' / 'vms.csv'
    catalog_path.parent.mkdir()
    _write_catalog(catalog_path)
    monkeypatch.setattr(vast_refresh.catalog_common, 'get_catalog_path',
                        lambda _name: str(catalog_path))
    monkeypatch.setattr(vast_refresh, 'has_credentials', lambda: True)
    monkeypatch.setattr(fetch_vast, 'fetch_vast_catalog', lambda: [{}, {}])
    monkeypatch.setattr(
        fetch_vast, 'save_catalog',
        lambda _rows, output: _write_catalog(Path(output), row_count=2))

    refresh_logger = logging.getLogger('sky.catalog.vast_refresh')
    refresh_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger=refresh_logger.name):
            assert vast_refresh.refresh_catalog(force=True)
    finally:
        refresh_logger.removeHandler(caplog.handler)

    assert 'Vast catalog CSV updated' in caplog.text
    assert 'records_before=1' in caplog.text
    assert 'records_fetched=2' in caplog.text
    assert 'records_after=2' in caplog.text


def test_refresh_catalog_keeps_valid_file_on_fetch_failure(
        monkeypatch, tmp_path, caplog):
    """Refresh preserves old data and logs a redacted provider cause."""
    catalog_path = tmp_path / 'vast' / 'vms.csv'
    catalog_path.parent.mkdir()
    _write_catalog(catalog_path)
    original = catalog_path.read_text(encoding='utf-8')
    os.utime(catalog_path, ns=(1, 1))
    credential_path = tmp_path / 'vast_api_key'
    credential_path.write_text('credential-secret', encoding='utf-8')
    monkeypatch.setattr(vast_refresh.catalog_common, 'get_catalog_path',
                        lambda _name: str(catalog_path))
    monkeypatch.setattr(vast_refresh, '_CREDENTIAL_PATH', str(credential_path))
    monkeypatch.setattr(
        fetch_vast, 'fetch_vast_catalog', lambda:
        (_ for _ in ()).throw(RuntimeError('SDK offline: credential-secret')))

    refresh_logger = logging.getLogger(vast_refresh.__name__)
    refresh_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=vast_refresh.__name__):
            assert vast_refresh.refresh_catalog()
    finally:
        refresh_logger.removeHandler(caplog.handler)

    assert catalog_path.read_text(encoding='utf-8') == original
    assert 'RuntimeError' in caplog.text
    assert 'SDK offline' in caplog.text
    assert '<redacted>' in caplog.text
    assert 'credential-secret' not in caplog.text


def test_forced_refresh_reports_failure_with_valid_stale_catalog(
        monkeypatch, tmp_path):
    """A forced refresh preserves a sanitized provider failure as its cause."""
    catalog_path = tmp_path / 'vast' / 'vms.csv'
    catalog_path.parent.mkdir()
    _write_catalog(catalog_path)
    original = catalog_path.read_text(encoding='utf-8')
    monkeypatch.setattr(vast_refresh.catalog_common, 'get_catalog_path',
                        lambda _name: str(catalog_path))
    monkeypatch.setattr(vast_refresh, 'has_credentials', lambda: True)
    monkeypatch.setattr(fetch_vast, 'fetch_vast_catalog', lambda:
                        (_ for _ in ()).throw(RuntimeError('offline')))

    with pytest.raises(RuntimeError, match='RuntimeError: offline') as exc_info:
        vast_refresh.refresh_catalog(force=True)
    assert catalog_path.read_text(encoding='utf-8') == original
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == 'offline'


def test_refresh_catalog_skips_without_vast_credential_file(monkeypatch):
    """Refresh remains disabled unless the Vast credential file is present."""
    monkeypatch.setattr(vast_refresh, 'has_credentials', lambda: False)
    monkeypatch.setattr(fetch_vast, 'fetch_vast_catalog',
                        lambda: pytest.fail('refresh must not fetch'))

    assert not vast_refresh.refresh_catalog()


def test_refresh_catalog_force_bypasses_fresh_catalog(monkeypatch, tmp_path):
    """A feasibility retry refreshes a valid catalog inside its age window."""
    catalog_path = tmp_path / 'vast' / 'vms.csv'
    catalog_path.parent.mkdir()
    _write_catalog(catalog_path)
    monkeypatch.setattr(vast_refresh.catalog_common, 'get_catalog_path',
                        lambda _name: str(catalog_path))
    monkeypatch.setattr(vast_refresh, 'has_credentials', lambda: True)
    fetch_catalog = mock.Mock(return_value=[{}])
    monkeypatch.setattr(fetch_vast, 'fetch_vast_catalog', fetch_catalog)
    monkeypatch.setattr(fetch_vast, 'save_catalog',
                        lambda _rows, output: _write_catalog(Path(output)))

    assert vast_refresh.refresh_catalog(force=True)
    fetch_catalog.assert_called_once_with()
