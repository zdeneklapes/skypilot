"""A script that generates the Vast Cloud catalog. """

#
# Due to the design of the sdk, pylint has a false
# positive for the functions.
#
# pylint: disable=assignment-from-no-return
import argparse
import collections
import csv
import json
import math
import os
from typing import Any, Dict, List, Set, Tuple

from sky import sky_logging
from sky.adaptors import vast

_MAPPED_KEYS = (
    ('gpu_name', 'InstanceType'),
    ('gpu_name', 'AcceleratorName'),
    ('num_gpus', 'AcceleratorCount'),
    ('cpu_cores', 'vCPUs'),
    ('cpu_ram', 'MemoryGiB'),
    ('gpu_name', 'GpuInfo'),
    ('dph_total', 'Price'),
    ('min_bid', 'SpotPrice'),
    ('geolocation', 'Region'),
    ('hosting_type', 'HostingType'),
)

_CATALOG_DISK_SIZE_GIB = 80
_CATALOG_QUERY = (
    'verified=true rentable=true rented=false external=false georegion=true '
    f'inet_down>=100 disk_space>={_CATALOG_DISK_SIZE_GIB}')

logger = sky_logging.init_logger(__name__)


class _OfferNormalizationError(ValueError):
    """A sanitized reason why one provider offer cannot become metadata."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def create_instance_type(obj: Dict[str, Any], per_gpu_vram_mib: int) -> str:
    """Return a stable Vast type that preserves per-device VRAM."""
    return vast.build_instance_type_from_offer({
        **obj,
        'gpu_ram': per_gpu_vram_mib,
    })


def get_per_gpu_vram_mib(offer: Dict[str, Any]) -> int:
    """Return validated per-device VRAM from Vast's total-memory field."""
    try:
        gpu_count = int(offer['num_gpus'])
        total_vram_mib = float(offer['gpu_total_ram'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Vast offer is missing GPU memory metadata.') from exc
    if (gpu_count <= 0 or not math.isfinite(total_vram_mib) or
            total_vram_mib <= 0):
        raise ValueError('Vast offer has invalid GPU memory metadata.')
    return round(total_vram_mib / gpu_count)


def _nonnegative_finite_number(value: Any) -> float:
    """Return one finite non-negative provider number."""
    if isinstance(value, bool):
        raise ValueError
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError
    return number


def _normalize_optional_region(value: Any) -> str:
    """Normalize an absent provider locality to the global catalog region."""
    if value is None:
        return 'any'
    if not isinstance(value, str):
        raise _OfferNormalizationError('invalid_region')
    return value.strip() or 'any'


def _normalize_optional_hosting_type(offer: Dict[str, Any]) -> int:
    """Treat absent SDK hosting metadata as consumer-hosted."""
    value = offer.get('hosting_type')
    if value is None:
        datacenter = offer.get('datacenter')
        if datacenter in (True, 1, 1.0):
            return 1
        if datacenter in (None, False, 0, 0.0):
            return 0
        raise _OfferNormalizationError('invalid_hosting_type')
    if isinstance(value, bool):
        return int(value)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise _OfferNormalizationError('invalid_hosting_type') from exc
    if (not math.isfinite(numeric_value) or numeric_value < 0 or
            not numeric_value.is_integer()):
        raise _OfferNormalizationError('invalid_hosting_type')
    return int(numeric_value)


def _normalize_offer(offer: Any) -> Tuple[str, Dict[str, Any]]:
    """Convert one SDK 1.5.0 offer into a validated catalog row."""
    if not isinstance(offer, dict):
        raise _OfferNormalizationError('invalid_offer')
    gpu_name = offer.get('gpu_name')
    if not isinstance(gpu_name, str) or not gpu_name.strip():
        raise _OfferNormalizationError('invalid_shape')
    try:
        per_gpu_vram_mib = get_per_gpu_vram_mib(offer)
        instance_type = create_instance_type(offer, per_gpu_vram_mib)
        accelerator_count = int(float(offer['num_gpus']))
        cpu_cores = int(float(offer['cpu_cores']))
        cpu_ram_mib = int(float(offer['cpu_ram']))
    except (KeyError, TypeError, ValueError) as exc:
        raise _OfferNormalizationError('invalid_shape') from exc

    try:
        price = _nonnegative_finite_number(offer.get('dph_total'))
    except (TypeError, ValueError) as exc:
        raise _OfferNormalizationError('invalid_price') from exc
    raw_spot_price = offer.get('min_bid')
    try:
        spot_price = (price if raw_spot_price is None else
                      _nonnegative_finite_number(raw_spot_price))
    except (TypeError, ValueError) as exc:
        raise _OfferNormalizationError('invalid_spot_price') from exc

    raw_total_vram = float(offer['gpu_total_ram'])
    total_vram_mib: Any = (int(raw_total_vram)
                           if raw_total_vram.is_integer() else raw_total_vram)
    accelerator_name = vast.canonicalize_accelerator_name(
        gpu_name, per_gpu_vram_mib)
    entry = {
        'InstanceType': instance_type,
        'AcceleratorName': accelerator_name,
        'AcceleratorCount': accelerator_count,
        'vCPUs': cpu_cores,
        'MemoryGiB': cpu_ram_mib / 1024,
        'GpuInfo': json.dumps({
            'Gpus': [{
                'Name': accelerator_name,
                'Count': accelerator_count,
                'MemoryInfo': {
                    'SizeInMiB': per_gpu_vram_mib
                }
            }],
            'TotalGpuMemoryInMiB': total_vram_mib
        }).replace('"', '\''),
        'Price': price,
        'SpotPrice': spot_price,
        'Region': _normalize_optional_region(offer.get('geolocation')),
        'HostingType': _normalize_optional_hosting_type(offer),
    }
    return instance_type, entry


def _format_rejection_counts(rejection_counts: Dict[str, int]) -> str:
    """Format bounded aggregate diagnostics without provider offer contents."""
    return ', '.join(f'{reason}={count}'
                     for reason, count in sorted(rejection_counts.items()))


def fetch_vast_catalog() -> List[Dict[str, Any]]:
    """Fetch and normalize the current Vast offers into catalog rows."""
    seen: Set[Tuple[str, str, str]] = set()
    # InstanceList is the buffered list to emit to
    # the CSV.
    csv_list = []

    # InstanceType and gpuInfo are basically just stubs
    # so that the dictwriter is happy without weird
    # code.
    # Vast has a wide variety of machines, some of
    # which will have less diskspace and network
    # bandwidth than others.
    #
    # The machine normally have high specificity
    # in the vast catalog - this is fairly unique
    # to Vast and can make bucketing them into
    # instance types difficult.
    #
    # The flags
    #
    #   * georegion consolidates geographic areas
    #
    #   * inet_down makes sure that only machines
    #     with "reasonable" downlink speed are
    #     considered
    #
    #   * disk_space sets a lower limit of how
    #     much space is availble to be allocated
    #     in order to ensure that machines with
    #     small disk pools aren't listed
    #
    offer_list = vast.vast().search_offers(query=_CATALOG_QUERY,
                                           type='on-demand',
                                           order='dph_total',
                                           limit=10000,
                                           storage=_CATALOG_DISK_SIZE_GIB,
                                           no_default=True)

    price_map: Dict[str, List] = collections.defaultdict(list)
    rejection_counts: Dict[str, int] = collections.Counter()
    try:
        offers = list(offer_list)
    except TypeError as exc:
        raise ValueError('Vast catalog contains no usable offers; '
                         'invalid_response=1.') from exc
    for offer in offers:
        try:
            instance_type, entry = _normalize_offer(offer)
        except _OfferNormalizationError as exc:
            rejection_counts[exc.reason] += 1
            continue
        price_map[instance_type].append(entry)

    if rejection_counts:
        logger.warning(
            'Vast catalog skipped malformed offers: offers=%d usable=%d '
            'rejected=%d %s.', len(offers), sum(map(len, price_map.values())),
            sum(rejection_counts.values()),
            _format_rejection_counts(rejection_counts))
    if not price_map:
        if not rejection_counts:
            rejection_counts['empty_response'] = 1
        raise ValueError('Vast catalog contains no usable offers; ' +
                         _format_rejection_counts(rejection_counts) + '.')

    for instance_list in price_map.values():
        price_list = sorted([x['Price'] for x in instance_list])
        index = math.ceil(0.5 * len(price_list)) - 1
        price_target = price_list[index]
        to_list: List = []
        for instance in instance_list:
            if instance['Price'] <= price_target:
                instance['Price'] = '{:.2f}'.format(price_target)
                to_list.append(instance)

        max_bid = max([x.get('SpotPrice') for x in to_list])
        for instance in to_list:
            hosting_type = instance.get('HostingType', 0)
            raw_region = instance['Region']
            try:
                country_code = vast.extract_country_code(raw_region)
            except ValueError:
                geographic_key = f'raw:{str(raw_region).strip().casefold()}'
            else:
                geographic_key = country_code or 'any'
            deduplication_key = (instance['InstanceType'], geographic_key,
                                 str(hosting_type))
            if deduplication_key in seen:
                continue
            instance['SpotPrice'] = f'{max_bid:.2f}'
            csv_list.append(instance)
            seen.add(deduplication_key)

    return csv_list


def save_catalog(instances: List[Dict[str, Any]], output_file: str) -> None:
    """Save previously fetched Vast catalog rows to a CSV file."""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile,
                                fieldnames=[x[1] for x in _MAPPED_KEYS])
        writer.writeheader()

        for instance in instances:
            writer.writerow(instance)


def main() -> None:
    """Generate the Vast CSV used by hosted catalog publishing jobs."""
    parser = argparse.ArgumentParser(
        description='Update Vast catalog for SkyPilot')
    parser.add_argument('--output', default='vast/vms.csv')
    args = parser.parse_args()
    save_catalog(fetch_vast_catalog(), args.output)


if __name__ == '__main__':
    main()
