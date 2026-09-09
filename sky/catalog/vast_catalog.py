""" Vast | Catalog

This module loads the service catalog file and can be used to
query instance types and pricing information for Vast.ai.
"""

import ast
import math
import typing
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from sky import sky_logging
from sky.adaptors import vast as vast_adaptor
from sky.catalog import common
from sky.utils import annotations
from sky.utils import resources_utils
from sky.utils import ux_utils

if typing.TYPE_CHECKING:
    from sky.clouds import cloud

_REQUIRED_CATALOG_COLUMNS = {
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

_df = common.read_catalog('vast/vms.csv')
logger = sky_logging.init_logger(__name__)


@annotations.lru_cache(scope='request', maxsize=1)
def _catalog_df() -> pd.DataFrame:
    """Return validated stable Vast instance-type metadata from local cache.

    Vast marketplace offers are selected only during provisioning. They must
    not become catalog identities because offer IDs can disappear at any time.
    Local catalog refreshes replace the CSV atomically; all catalog queries
    within one request share the same stable metadata snapshot.
    """
    try:
        catalog_df = _df
        missing_columns = _REQUIRED_CATALOG_COLUMNS.difference(
            catalog_df.columns)
        if missing_columns:
            missing = ', '.join(sorted(missing_columns))
            raise common.CatalogFetchError(
                'Local Vast catalog is missing required columns: '
                f'{missing}.')

        accelerator_count = pd.to_numeric(catalog_df['AcceleratorCount'],
                                          errors='coerce')
        usable_gpu_rows = (catalog_df['AcceleratorName'].notna() &
                           accelerator_count.gt(0) &
                           catalog_df['GpuInfo'].notna())
        if usable_gpu_rows.any():
            return typing.cast(pd.DataFrame, catalog_df)
        raise common.CatalogFetchError(
            'Local Vast catalog contains no usable GPU rows.')
    except common.CatalogFetchError:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        raise common.CatalogFetchError(
            'Local Vast catalog is not valid CSV.') from exc


def reload_catalog() -> None:
    """Reload only Vast metadata after an atomic local catalog refresh."""
    global _df
    _df = common.read_catalog('vast/vms.csv')
    _catalog_df.cache_clear()


def _normalize_accelerator_alias(name: str) -> str:
    """Normalize only harmless Vast accelerator spelling differences."""
    return str(name).casefold().replace(' ', '').replace('_', '')


def _matching_accelerator_rows(catalog_df: pd.DataFrame,
                               acc_name: str) -> pd.DataFrame:
    """Resolve exact names first, then one unique normalized name."""
    accelerator_names = catalog_df['AcceleratorName'].astype(str)
    exact_rows = catalog_df[accelerator_names == acc_name]
    if not exact_rows.empty:
        return exact_rows
    normalized_name = _normalize_accelerator_alias(acc_name)
    normalized_matches = accelerator_names.map(
        _normalize_accelerator_alias) == normalized_name
    matching_names = set(accelerator_names[normalized_matches])
    if len(matching_names) != 1:
        return catalog_df.iloc[0:0]
    return catalog_df[normalized_matches]


def get_canonical_accelerator_name(instance_type: str,
                                   requested_name: Optional[str] = None) -> str:
    """Return the catalog's canonical name for a raw Vast GPU identity."""
    rows = _catalog_df()
    rows = rows[rows['InstanceType'] == instance_type]
    canonical_names = set(rows['AcceleratorName'].dropna().astype(str))
    if len(canonical_names) == 1:
        return canonical_names.pop()
    if requested_name is not None:
        matching_rows = _matching_accelerator_rows(_catalog_df(),
                                                   requested_name)
        matching_names = set(
            matching_rows['AcceleratorName'].dropna().astype(str))
        if len(matching_names) == 1:
            return matching_names.pop()
        return requested_name
    return vast_adaptor.get_offer_requirements(instance_type,
                                               region=None,
                                               disk_size=1,
                                               datacenter_only=False,
                                               reliable_hosts=False,
                                               network_tier='standard').gpu_name


def _apply_datacenter_filter(df: pd.DataFrame,
                             datacenter_only: bool) -> pd.DataFrame:
    """Filter dataframe by hosting_type if datacenter_only is True.

    hosting_type: 0 = Consumer hosted, 1 = Datacenter hosted
    """
    if not datacenter_only:
        return df
    if 'HostingType' not in df.columns:
        return df.iloc[0:0]
    hosting_type = pd.to_numeric(df['HostingType'], errors='coerce')
    return df[hosting_type.ge(1)]


def _planning_catalog_df() -> pd.DataFrame:
    """Return only v2 rows supported by new Vast placement paths."""
    catalog_df = _catalog_df()
    supported = catalog_df['InstanceType'].astype(str).str.startswith('vastv2-')
    return catalog_df[supported]


def instance_type_exists(instance_type: str) -> bool:
    return common.instance_type_exists_impl(_catalog_df(), instance_type)


def _get_missing_v2_instance_type_requirements(
        catalog_df: pd.DataFrame,
        instance_type: str) -> Optional[vast_adaptor.VastOfferRequirements]:
    """Parse an absent stable v2 identity without making it launchable."""
    if (not instance_type.startswith('vastv2-') or
            common.instance_type_exists_impl(catalog_df, instance_type)):
        return None
    requirements = vast_adaptor.get_offer_requirements(
        instance_type,
        region=None,
        disk_size=1,
        datacenter_only=False,
        reliable_hosts=False,
        network_tier='standard',
    )
    logger.debug('Using embedded metadata for stale Vast v2 instance type %s',
                 instance_type)
    return requirements


def _get_missing_legacy_instance_type_metadata(
        catalog_df: pd.DataFrame,
        instance_type: str) -> Optional[vast_adaptor.VastInstanceTypeMetadata]:
    """Parse an absent legacy identity for status metadata only."""
    if (instance_type.startswith('vastv2-') or
            common.instance_type_exists_impl(catalog_df, instance_type)):
        return None
    metadata = vast_adaptor.get_instance_type_metadata(instance_type)
    logger.debug(
        'Using embedded status metadata for absent legacy Vast '
        'instance type %s', instance_type)
    return metadata


def validate_region_zone(
        region: Optional[str],
        zone: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Vast does not support zones.')
    return common.validate_region_zone_impl('vast', _catalog_df(), region, zone)


def get_hourly_cost(instance_type: str,
                    use_spot: bool = False,
                    region: Optional[str] = None,
                    zone: Optional[str] = None) -> float:
    """Returns the cost, or the cheapest cost among all zones for spot."""
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Vast does not support zones.')
    try:
        return common.get_hourly_cost_impl(_catalog_df(), instance_type,
                                           use_spot, region, zone)
    except ValueError:
        if region is None:
            raise
        # A live offer may exist in a country absent from the refreshed
        # metadata. Use the global catalog price as an estimate in that case.
        return common.get_hourly_cost_impl(_catalog_df(), instance_type,
                                           use_spot, None, zone)


def get_vcpus_mem_from_instance_type(
        instance_type: str) -> Tuple[Optional[float], Optional[float]]:
    catalog_df = _catalog_df()
    legacy_metadata = _get_missing_legacy_instance_type_metadata(
        catalog_df, instance_type)
    if legacy_metadata is not None:
        return legacy_metadata.cpu_cores, legacy_metadata.cpu_ram_mib / 1024
    requirements = _get_missing_v2_instance_type_requirements(
        catalog_df, instance_type)
    if requirements is not None:
        cpu_ram_mib = requirements.cpu_ram_mib
        assert cpu_ram_mib is not None
        return requirements.cpu_cores, cpu_ram_mib / 1024
    return common.get_vcpus_mem_from_instance_type_impl(catalog_df,
                                                        instance_type)


def get_default_instance_type(cpus: Optional[str] = None,
                              memory: Optional[str] = None,
                              disk_tier: Optional[
                                  resources_utils.DiskTier] = None,
                              local_disk: Optional[str] = None,
                              region: Optional[str] = None,
                              zone: Optional[str] = None,
                              use_spot: bool = False,
                              max_hourly_cost: Optional[float] = None,
                              datacenter_only: bool = False) -> Optional[str]:
    del disk_tier, local_disk
    # NOTE: After expanding catalog to multiple entries, you may
    # want to specify a default instance type or family.
    df = _apply_datacenter_filter(_planning_catalog_df(), datacenter_only)
    return common.get_instance_type_for_cpus_mem_impl(df, cpus, memory, region,
                                                      zone, use_spot,
                                                      max_hourly_cost)


def _representative_instance_types(rows: pd.DataFrame) -> List[str]:
    """Return one stable type for every raw GPU/count/VRAM identity."""
    identities: Dict[Tuple[str, int, int], str] = {}
    for instance_type in rows['InstanceType']:
        requirements = vast_adaptor.get_offer_requirements(
            str(instance_type),
            region=None,
            disk_size=1,
            datacenter_only=False,
            reliable_hosts=False,
            network_tier='standard')
        key = (requirements.gpu_name.casefold(), requirements.num_gpus,
               requirements.gpu_ram_mib)
        identities.setdefault(key, str(instance_type))
    return list(identities.values())


def get_default_instance_types(zone: Optional[str] = None) -> List[str]:
    """Return every supported GPU identity for live default placement."""
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Vast does not support zones.')
    return _representative_instance_types(_planning_catalog_df())


def get_accelerators_from_instance_type(
        instance_type: str) -> Optional[Dict[str, Union[int, float]]]:
    catalog_df = _catalog_df()
    legacy_metadata = _get_missing_legacy_instance_type_metadata(
        catalog_df, instance_type)
    if legacy_metadata is not None:
        return {legacy_metadata.gpu_name: legacy_metadata.num_gpus}
    requirements = _get_missing_v2_instance_type_requirements(
        catalog_df, instance_type)
    if requirements is not None:
        return {requirements.gpu_name: requirements.num_gpus}
    return common.get_accelerators_from_instance_type_impl(
        catalog_df, instance_type)


def get_legacy_per_gpu_vram_mib(instance_type: str, num_gpus: int) -> int:
    """Resolve a legacy type only when all catalog rows prove one VRAM value."""
    vram_values = set()
    catalog_df = _catalog_df()
    rows = catalog_df[catalog_df['InstanceType'] == instance_type]
    for gpu_info in rows['GpuInfo']:
        try:
            parsed_gpu_info = (ast.literal_eval(gpu_info) if isinstance(
                gpu_info, str) else gpu_info)
        except (TypeError, ValueError, SyntaxError):
            continue
        try:
            total_vram_mib = float(parsed_gpu_info['TotalGpuMemoryInMiB'])
        except (KeyError, TypeError, ValueError):
            if num_gpus != 1:
                continue
            try:
                total_vram_mib = float(
                    parsed_gpu_info['Gpus'][0]['MemoryInfo']['SizeInMiB'])
            except (KeyError, TypeError, ValueError):
                continue
        if math.isfinite(total_vram_mib) and total_vram_mib > 0:
            vram_values.add(round(total_vram_mib / num_gpus))
    if len(vram_values) != 1:
        raise ValueError(
            'Legacy Vast instance type is ambiguous across GPU VRAM values; '
            'refresh or reselect the resource.')
    return vram_values.pop()


def get_instance_type_for_accelerator(
        acc_name: str,
        acc_count: int,
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
        use_spot: bool = False,
        local_disk: Optional[str] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
        max_hourly_cost: Optional[float] = None,
        datacenter_only: bool = False) -> Tuple[Optional[List[str]], List[str]]:
    """Return representative catalog types for a raw GPU identity.

    CPU, RAM, locality, host policy, market, and price are checked live.
    """
    del (cpus, memory, use_spot, local_disk, region, max_hourly_cost,
         datacenter_only)
    if zone is not None:
        with ux_utils.print_exception_no_traceback():
            raise ValueError('Vast does not support zones.')
    catalog_df = _planning_catalog_df()
    rows = _matching_accelerator_rows(catalog_df, acc_name)
    accelerator_count = pd.to_numeric(rows['AcceleratorCount'], errors='coerce')
    rows = rows[(accelerator_count - float(acc_count)).abs() <= 0.01]
    if rows.empty:
        _, fuzzy_candidates = common.get_instance_type_for_accelerator_impl(
            df=catalog_df,
            acc_name=acc_name,
            acc_count=acc_count,
        )
        return None, fuzzy_candidates

    # CPU, RAM, locality, host policy, and live price are marketplace state.
    return _representative_instance_types(rows), []


def get_region_zones_for_instance_type(instance_type: str,
                                       use_spot: bool) -> List['cloud.Region']:
    catalog_df = _catalog_df()
    df = catalog_df[catalog_df['InstanceType'] == instance_type]
    return common.get_region_zones(df, use_spot)


# TODO: this differs from the fluffy catalog version
def list_accelerators(
        gpus_only: bool,
        name_filter: Optional[str],
        region_filter: Optional[str],
        quantity_filter: Optional[int],
        case_sensitive: bool = True,
        all_regions: bool = False,
        require_price: bool = True) -> Dict[str, List[common.InstanceTypeInfo]]:
    """Returns all instance types in Vast offering GPUs."""
    del require_price  # Unused.
    return common.list_accelerators_impl('Vast', _catalog_df(), gpus_only,
                                         name_filter, region_filter,
                                         quantity_filter, case_sensitive,
                                         all_regions)
