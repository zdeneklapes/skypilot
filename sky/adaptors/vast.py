"""Vast cloud adaptor."""

import dataclasses
import functools
import math
import re
from typing import Any, Dict, Optional, Tuple

from sky.utils import annotations

_vast_sdk = None
_COUNTRY_CODE_PATTERN = re.compile(r'^[A-Za-z]{2}$')
_CONTINENT_CODES = frozenset({'AF', 'AN', 'AS', 'EU', 'LC', 'NA', 'OC', 'SA'})
_MIN_RELIABILITY = 0.99
_MIN_NETWORK_BANDWIDTH_MBPS = 1000
_DIRECT_SEARCH_LIMIT = 10000
_ACCELERATOR_NAME_ALIASES = {
    'teslav100': 'V100',
    'teslat4': 'T4',
    'teslap100': 'P100',
    'qrtx6000': 'RTX6000',
    'qrtx8000': 'RTX8000',
}
_ACCELERATOR_MEMORY_VARIANTS = {
    ('A100', 80 * 1024): 'A100-80GB',
    ('V100', 32 * 1024): 'V100-32GB',
}


@dataclasses.dataclass(frozen=True)
class NumericConstraint:
    """One exact, minimum, ratio, or unspecified numeric requirement."""

    mode: str
    value: Optional[float]


@dataclasses.dataclass(frozen=True)
class VastInstanceTypeMetadata:
    """Metadata encoded directly in a legacy or v2 Vast instance type."""

    gpu_name: str
    num_gpus: int
    gpu_ram_mib: Optional[int]
    cpu_cores: int
    cpu_ram_mib: int


@dataclasses.dataclass(frozen=True)
class VastOfferRequirements:
    """Requirements that a live Vast offer must satisfy."""

    gpu_name: str
    num_gpus: int
    gpu_ram_mib: int
    cpu: NumericConstraint
    memory: NumericConstraint
    disk_size: int
    country_code: Optional[str]
    datacenter_only: bool
    reliable_hosts: bool
    network_tier: str
    use_spot: bool
    max_hourly_cost: Optional[float]
    requested_accelerator_name: Optional[str] = None

    @property
    def cpu_cores(self) -> Optional[float]:
        """Return the CPU threshold retained for compatibility."""
        return self.cpu.value

    @property
    def cpu_ram_mib(self) -> Optional[float]:
        """Return the absolute RAM threshold, excluding ratio constraints."""
        if self.memory.mode == 'ratio':
            return None
        return self.memory.value


@dataclasses.dataclass(frozen=True)
class LiveOfferQueryResult:
    """The matching live Vast offers and sanitized query diagnostics."""

    offers: Tuple[Dict[str, Any], ...]
    error: Optional[str]
    offers_examined: int
    rejection_counts: Tuple[Tuple[str, int], ...]


def import_package(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _vast_sdk

        if _vast_sdk is None:
            try:
                # isort: off
                from vastai.sdk import VastAI  # pylint: disable=import-outside-toplevel
                # isort: on
                _vast_sdk = VastAI()
            except ImportError as e:
                raise ImportError(f'Fail to import dependencies for vast: {e}\n'
                                  'Try pip install "skypilot[vast]"') from None
        return func(*args, **kwargs)

    return wrapper


@import_package
def vast():
    """Return the vast package."""
    return _vast_sdk


def extract_country_code(region: Optional[str]) -> Optional[str]:
    """Return a country code from a normalized or raw Vast catalog region.

    Vast raw regions use ``locality, country, continent``.  The continent is
    also a two-letter code, so accepting a trailing code would silently turn
    ``France, FR, EU`` into ``EU``.  Reject malformed values instead.
    """
    if region is None:
        return None
    if not isinstance(region, str):
        raise ValueError(f'Vast region must be a string, got {region!r}.')

    if region.strip().lower() == 'any':
        return None

    normalized_region = region.strip()
    if _COUNTRY_CODE_PATTERN.fullmatch(normalized_region):
        return normalized_region.upper()

    parts = [part.strip() for part in normalized_region.split(',')]
    final_part = parts[-1].upper()
    if len(parts) == 2 and _COUNTRY_CODE_PATTERN.fullmatch(final_part):
        first_part = parts[0].upper()
        if final_part in _CONTINENT_CODES:
            if _COUNTRY_CODE_PATTERN.fullmatch(first_part):
                return first_part
            if parts[0]:
                return final_part
        elif parts[0]:
            return final_part
    elif len(parts) >= 3:
        if final_part in _CONTINENT_CODES:
            country_part = parts[-2].upper()
            locality_parts = parts[:-2]
            valid_locality = (all(locality_parts) or locality_parts == [''])
            if (valid_locality and
                    _COUNTRY_CODE_PATTERN.fullmatch(country_part)):
                return country_part
        elif (_COUNTRY_CODE_PATTERN.fullmatch(final_part) and all(parts[:-1])):
            return final_part
    raise ValueError('Vast region must be a two-letter country code or a raw '
                     '"locality, country, continent" value; '
                     f'could not extract a country from {region!r}.')


def _normalize_gpu_name(gpu_name: Any) -> str:
    """Normalize equivalent space and underscore GPU spellings."""
    return re.sub(r'[\s_]+', ' ', str(gpu_name or '')).strip().casefold()


def _normalize_accelerator_alias(accelerator_name: Any) -> str:
    """Normalize harmless spelling differences in accelerator aliases."""
    return re.sub(r'[\s_]+', '', str(accelerator_name or '')).casefold()


def canonicalize_accelerator_name(gpu_name: Any, gpu_ram_mib: int) -> str:
    """Map a raw Vast GPU shape to its SkyPilot accelerator name."""
    normalized_gpu = re.sub(r'[\s_]+', '', str(gpu_name or '').strip())
    normalized_gpu = re.sub('Ada', '-Ada', normalized_gpu, flags=re.IGNORECASE)
    normalized_gpu = re.sub(r'(Ti|PCIE|SXM4|SXM|NVL)$',
                            '',
                            normalized_gpu,
                            flags=re.IGNORECASE)
    normalized_gpu = re.sub(r'(RTX\d0\d0)(S|D)$',
                            r'\1',
                            normalized_gpu,
                            flags=re.IGNORECASE)
    normalized_gpu = _ACCELERATOR_NAME_ALIASES.get(normalized_gpu.casefold(),
                                                   normalized_gpu)
    normalized_alias = _normalize_accelerator_alias(normalized_gpu)
    for (variant_gpu, variant_ram_mib), variant_name in (
            _ACCELERATOR_MEMORY_VARIANTS.items()):
        if (gpu_ram_mib == variant_ram_mib and
                normalized_alias == _normalize_accelerator_alias(variant_gpu)):
            return variant_name
    return normalized_gpu


def _minimum_accelerator_memory_mib(accelerator_name: str) -> int:
    """Return the query minimum needed by a memory-specific accelerator."""
    normalized_name = _normalize_accelerator_alias(accelerator_name)
    for (_,
         gpu_ram_mib), variant_name in (_ACCELERATOR_MEMORY_VARIANTS.items()):
        if _normalize_accelerator_alias(variant_name) == normalized_name:
            return gpu_ram_mib
    return 1


def _positive_finite_number(value: Any) -> Optional[float]:
    """Return a positive finite number or None for malformed input."""
    if isinstance(value, bool):
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(normalized) or normalized <= 0:
        return None
    return normalized


def _positive_integral_number(value: Any) -> Optional[int]:
    """Return a positive integer without truncating provider metadata."""
    normalized = _positive_finite_number(value)
    if normalized is None or not normalized.is_integer():
        return None
    return int(normalized)


def _format_number(value: float) -> str:
    """Format query numbers without unnecessary decimal suffixes."""
    return str(int(value)) if value.is_integer() else str(value)


def _parse_cpu_constraint(cpus: Optional[str]) -> NumericConstraint:
    """Parse SkyPilot CPU syntax without losing exact/minimum semantics."""
    if cpus is None:
        return NumericConstraint('unspecified', None)
    value = str(cpus).strip()
    mode = 'minimum' if value.endswith('+') else 'exact'
    number = _positive_finite_number(value[:-1] if mode == 'minimum' else value)
    if number is None:
        raise ValueError(f'Invalid Vast CPU requirement {cpus!r}.')
    return NumericConstraint(mode, number)


def _parse_memory_constraint(memory: Optional[str]) -> NumericConstraint:
    """Parse absolute or per-CPU SkyPilot memory syntax."""
    if memory is None:
        return NumericConstraint('unspecified', None)
    value = str(memory).strip()
    if value.endswith('+'):
        mode = 'minimum'
        numeric_value = value[:-1]
    elif value.endswith('x'):
        mode = 'ratio'
        numeric_value = value[:-1]
    else:
        mode = 'exact'
        numeric_value = value
    number = _positive_finite_number(numeric_value)
    if number is None:
        raise ValueError(f'Invalid Vast memory requirement {memory!r}.')
    if mode != 'ratio':
        number *= 1024
    return NumericConstraint(mode, number)


def _minimum_offer_value(offer: Dict[str, Any], key: str,
                         minimum: float) -> bool:
    """Return whether an offer has a finite numeric value at least minimum."""
    try:
        value = float(offer[key])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(value) and value >= minimum


def _is_true(offer: Dict[str, Any], key: str) -> bool:
    """Interpret Vast boolean fields without accepting arbitrary values."""
    value = offer.get(key)
    return value is True or value == 1 or value == 'true'


def _is_false(offer: Dict[str, Any], key: str) -> bool:
    """Interpret explicit false Vast boolean fields without coercion."""
    value = offer.get(key)
    return value is False or value == 0 or value == 'false'


def get_instance_type_metadata(instance_type: str) -> VastInstanceTypeMetadata:
    """Parse metadata embedded in a legacy or v2 Vast instance type."""
    parts = instance_type.split('-')
    is_legacy_instance_type = parts[0] != 'vastv2'
    try:
        if is_legacy_instance_type:
            if not parts[0].endswith('x'):
                raise ValueError
            num_gpus = int(parts[0][:-1])
            gpu_ram_mib = None
            cpu_cores = int(parts[-2])
            cpu_ram_mib = int(parts[-1])
            gpu_name = '-'.join(parts[1:-2]).replace('_', ' ')
        else:
            if not parts[1].endswith('x'):
                raise ValueError
            num_gpus = int(parts[1][:-1])
            gpu_ram_mib = int(parts[-3])
            cpu_cores = int(parts[-2])
            cpu_ram_mib = int(parts[-1])
            gpu_name = '-'.join(parts[2:-3]).replace('_', ' ')
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f'Invalid Vast instance type {instance_type!r}.') from exc
    numeric_metadata = [num_gpus, cpu_cores, cpu_ram_mib]
    if gpu_ram_mib is not None:
        numeric_metadata.append(gpu_ram_mib)
    if not gpu_name or min(numeric_metadata) <= 0:
        raise ValueError(f'Invalid Vast instance type {instance_type!r}.')
    return VastInstanceTypeMetadata(
        gpu_name=gpu_name,
        num_gpus=num_gpus,
        gpu_ram_mib=gpu_ram_mib,
        cpu_cores=cpu_cores,
        cpu_ram_mib=cpu_ram_mib,
    )


def get_offer_requirements(
        instance_type: str,
        region: Optional[str],
        disk_size: int,
        datacenter_only: bool,
        reliable_hosts: bool,
        network_tier: Any,
        *,
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
        use_resource_constraints: bool = False,
        use_spot: bool = False,
        max_hourly_cost: Optional[float] = None,
        resolved_shape: bool = False) -> VastOfferRequirements:
    """Parse a stable Vast instance type into its live-offer requirements."""
    metadata = get_instance_type_metadata(instance_type)
    is_legacy_instance_type = metadata.gpu_ram_mib is None
    try:
        normalized_disk_size = int(disk_size)
    except ValueError as exc:
        raise ValueError(
            f'Invalid Vast instance type {instance_type!r}.') from exc
    num_gpus = metadata.num_gpus
    cpu_cores = metadata.cpu_cores
    cpu_ram_mib = metadata.cpu_ram_mib
    gpu_name = metadata.gpu_name
    if is_legacy_instance_type:
        # Import lazily: the catalog generator imports this adapter.
        # pylint: disable=import-outside-toplevel
        from sky.catalog import vast_catalog
        gpu_ram_mib = vast_catalog.get_legacy_per_gpu_vram_mib(
            instance_type, num_gpus)
    else:
        assert metadata.gpu_ram_mib is not None
        gpu_ram_mib = metadata.gpu_ram_mib
    if (not gpu_name or min(num_gpus, gpu_ram_mib, cpu_cores, cpu_ram_mib,
                            normalized_disk_size) <= 0):
        raise ValueError(f'Invalid Vast instance type {instance_type!r}.')

    if resolved_shape:
        cpu_constraint = NumericConstraint('exact', float(cpu_cores))
        memory_constraint = NumericConstraint('exact', float(cpu_ram_mib))
    elif use_resource_constraints:
        cpu_constraint = _parse_cpu_constraint(cpus)
        memory_constraint = _parse_memory_constraint(memory)
    else:
        # Explicit legacy/v2 instance types retain their historical minimum
        # semantics unless they carry provider-resolved metadata.
        cpu_constraint = NumericConstraint('minimum', float(cpu_cores))
        memory_constraint = NumericConstraint('minimum', float(cpu_ram_mib))

    normalized_max_cost = None
    if max_hourly_cost is not None:
        normalized_max_cost = _positive_finite_number(max_hourly_cost)
        if normalized_max_cost is None:
            raise ValueError('Vast max_hourly_cost must be a positive finite '
                             f'number, got {max_hourly_cost!r}.')

    normalized_network_tier = str(getattr(network_tier, 'value',
                                          network_tier)).lower()
    if normalized_network_tier not in {'standard', 'best'}:
        raise ValueError(
            f'Invalid Vast network tier {network_tier!r}; expected standard '
            'or best.')
    return VastOfferRequirements(
        gpu_name=gpu_name,
        num_gpus=num_gpus,
        gpu_ram_mib=gpu_ram_mib,
        cpu=cpu_constraint,
        memory=memory_constraint,
        disk_size=normalized_disk_size,
        country_code=extract_country_code(region),
        datacenter_only=datacenter_only,
        reliable_hosts=reliable_hosts,
        network_tier=normalized_network_tier,
        use_spot=use_spot,
        max_hourly_cost=normalized_max_cost,
    )


def get_accelerator_offer_requirements(
        accelerator_name: str,
        accelerator_count: Any,
        region: Optional[str],
        disk_size: int,
        datacenter_only: bool,
        reliable_hosts: bool,
        network_tier: Any,
        *,
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
        use_spot: bool = False,
        max_hourly_cost: Optional[float] = None) -> VastOfferRequirements:
    """Build a catalog-independent live requirement for one accelerator."""
    normalized_name = str(accelerator_name).strip()
    normalized_count = _positive_integral_number(accelerator_count)
    try:
        normalized_disk_size = int(disk_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid Vast disk size {disk_size!r}.') from exc
    if (not normalized_name or normalized_count is None or
            normalized_disk_size <= 0):
        raise ValueError('Vast accelerator name, count, and disk size must be '
                         'positive values.')

    normalized_max_cost = None
    if max_hourly_cost is not None:
        normalized_max_cost = _positive_finite_number(max_hourly_cost)
        if normalized_max_cost is None:
            raise ValueError('Vast max_hourly_cost must be a positive finite '
                             f'number, got {max_hourly_cost!r}.')
    normalized_network_tier = str(getattr(network_tier, 'value',
                                          network_tier)).lower()
    if normalized_network_tier not in {'standard', 'best'}:
        raise ValueError(
            f'Invalid Vast network tier {network_tier!r}; expected standard '
            'or best.')
    return VastOfferRequirements(
        gpu_name='',
        num_gpus=normalized_count,
        gpu_ram_mib=_minimum_accelerator_memory_mib(normalized_name),
        cpu=_parse_cpu_constraint(cpus),
        memory=_parse_memory_constraint(memory),
        disk_size=normalized_disk_size,
        country_code=extract_country_code(region),
        datacenter_only=datacenter_only,
        reliable_hosts=reliable_hosts,
        network_tier=normalized_network_tier,
        use_spot=use_spot,
        max_hourly_cost=normalized_max_cost,
        requested_accelerator_name=normalized_name,
    )


def build_offer_query(requirements: VastOfferRequirements) -> str:
    """Build an SDK-safe exact query equivalent to live-offer matching."""
    # Vast SDK 1.5.0 preprocesses query values with an alphanumeric parser.
    # Use integral GiB for its cpu_ram filter, then validate exact MiB values
    # from returned offers in offer_matches_requirements().
    # Catalog-only chunking and geographic bucketing would weaken final
    # admission, so this query deliberately does not include those flags.
    query = [
        'rentable=true',
        'rented=false',
        'external=false',
        f'disk_space>={requirements.disk_size}',
        f'num_gpus={requirements.num_gpus}',
        f'gpu_ram>={math.ceil(requirements.gpu_ram_mib / 1024)}',
    ]
    if requirements.requested_accelerator_name is None:
        query.append(f'gpu_name={requirements.gpu_name.replace(" ", "_")}')
    if requirements.cpu.mode != 'unspecified':
        assert requirements.cpu.value is not None
        operator = '=' if requirements.cpu.mode == 'exact' else '>='
        query.append('cpu_cores' + operator +
                     _format_number(requirements.cpu.value))
    memory_minimum_mib = requirements.memory.value
    if requirements.memory.mode == 'ratio':
        memory_minimum_mib = None
        if requirements.cpu.value is not None:
            assert requirements.memory.value is not None
            memory_minimum_mib = (requirements.memory.value *
                                  requirements.cpu.value * 1024)
    if memory_minimum_mib is not None:
        query.append(f'cpu_ram>={math.ceil(memory_minimum_mib / 1024)}')
    if requirements.country_code is not None:
        query.insert(2, f'geolocation={requirements.country_code}')
    if requirements.datacenter_only:
        query.append('datacenter=true')
    if requirements.reliable_hosts:
        query.extend([
            'verified=true',
            'datacenter=true',
            f'reliability>={_MIN_RELIABILITY}',
            f'inet_down>={_MIN_NETWORK_BANDWIDTH_MBPS}',
        ])
    if requirements.network_tier == 'best':
        if not requirements.reliable_hosts:
            query.append(f'inet_down>={_MIN_NETWORK_BANDWIDTH_MBPS}')
        query.append(f'inet_up>={_MIN_NETWORK_BANDWIDTH_MBPS}')
    return ' '.join(query)


def _offer_rejection_reason(
        offer: Any, requirements: VastOfferRequirements) -> Optional[str]:
    """Return a sanitized first unmet requirement for a live Vast offer."""
    if not isinstance(offer, dict):
        return 'malformed'
    external = offer.get('external')
    if (not _is_true(offer, 'rentable') or not _is_false(offer, 'rented') or
        (external is not None and not _is_false(offer, 'external'))):
        return 'availability'
    num_gpus = _positive_integral_number(offer.get('num_gpus'))
    if num_gpus is None:
        return 'malformed'
    if num_gpus != requirements.num_gpus:
        return 'gpu'
    offer_gpu_ram = _positive_integral_number(offer.get('gpu_ram'))
    if requirements.requested_accelerator_name is not None:
        if offer_gpu_ram is None:
            return 'vram'
        actual_accelerator_name = canonicalize_accelerator_name(
            offer.get('gpu_name'), offer_gpu_ram)
        if (_normalize_accelerator_alias(actual_accelerator_name) !=
                _normalize_accelerator_alias(
                    requirements.requested_accelerator_name)):
            return 'gpu'
    elif (_normalize_gpu_name(offer.get('gpu_name')) != _normalize_gpu_name(
            requirements.gpu_name)):
        return 'gpu'
    if offer_gpu_ram is None or offer_gpu_ram < requirements.gpu_ram_mib:
        return 'vram'
    offer_cpu = _positive_integral_number(offer.get('cpu_cores'))
    if offer_cpu is None:
        return 'cpu'
    if requirements.cpu.value is not None:
        if requirements.cpu.mode == 'exact':
            if offer_cpu != requirements.cpu.value:
                return 'cpu'
        elif offer_cpu < requirements.cpu.value:
            return 'cpu'
    offer_memory_mib = _positive_integral_number(offer.get('cpu_ram'))
    if offer_memory_mib is None:
        return 'ram'
    if requirements.memory.value is not None:
        if requirements.memory.mode == 'exact':
            if offer_memory_mib != requirements.memory.value:
                return 'ram'
        elif requirements.memory.mode == 'minimum':
            if offer_memory_mib < requirements.memory.value:
                return 'ram'
        elif offer_memory_mib / 1024 < requirements.memory.value * offer_cpu:
            return 'ram'
    if not _minimum_offer_value(offer, 'disk_space', requirements.disk_size):
        return 'disk'
    if requirements.country_code is not None:
        try:
            offer_country = extract_country_code(offer.get('geolocation'))
        except ValueError:
            return 'country'
        if offer_country != requirements.country_code:
            return 'country'

    requires_datacenter = (requirements.datacenter_only or
                           requirements.reliable_hosts)
    if (requires_datacenter and
        (not _is_true(offer, 'datacenter') or
         not _minimum_offer_value(offer, 'hosting_type', 1))):
        return 'host_policy'
    if requirements.reliable_hosts:
        if (not offer_is_verified(offer) or
                not _minimum_offer_value(offer, 'reliability', _MIN_RELIABILITY)
                or not _minimum_offer_value(offer, 'inet_down',
                                            _MIN_NETWORK_BANDWIDTH_MBPS)):
            return 'host_policy'
    if requirements.network_tier == 'best':
        if (not _minimum_offer_value(offer, 'inet_down',
                                     _MIN_NETWORK_BANDWIDTH_MBPS) or
                not _minimum_offer_value(offer, 'inet_up',
                                         _MIN_NETWORK_BANDWIDTH_MBPS)):
            return 'network'
    price = get_offer_hourly_price(offer, requirements.use_spot)
    if price is None:
        return 'price'
    if (requirements.max_hourly_cost is not None and
            price > requirements.max_hourly_cost):
        return 'price'
    return None


def offer_is_verified(offer: Dict[str, Any]) -> bool:
    """Interpret Vast verification state from current response fields."""
    verification = offer.get('verification')
    if verification is not None:
        return (isinstance(verification, str) and
                verification.strip().casefold() == 'verified')
    vericode = offer.get('vericode')
    if isinstance(vericode, bool):
        return False
    if isinstance(vericode, (int, float)):
        return math.isfinite(float(vericode)) and float(vericode) == 1
    return False


def get_offer_hourly_price(offer: Dict[str, Any],
                           use_spot: bool) -> Optional[float]:
    """Return the finite non-negative live price for the requested market."""
    price_key = 'min_bid' if use_spot else 'dph_total'
    try:
        price = float(offer[price_key])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(price) or price < 0:
        return None
    return price


def build_instance_type_from_offer(offer: Dict[str, Any]) -> str:
    """Build a stable v2 type from one locally admitted live offer."""
    gpu_name = str(offer.get('gpu_name') or '').strip()
    num_gpus = _positive_integral_number(offer.get('num_gpus'))
    gpu_ram_mib = _positive_integral_number(offer.get('gpu_ram'))
    cpu_cores = _positive_integral_number(offer.get('cpu_cores'))
    cpu_ram_mib = _positive_integral_number(offer.get('cpu_ram'))
    if (not gpu_name or num_gpus is None or gpu_ram_mib is None or
            cpu_cores is None or cpu_ram_mib is None):
        raise ValueError('Vast offer has invalid concrete shape metadata.')
    encoded_gpu_name = re.sub(r'\s', '_', gpu_name)
    return (f'vastv2-{num_gpus}x-{encoded_gpu_name}-{gpu_ram_mib}-'
            f'{cpu_cores}-{cpu_ram_mib}')


def _search_offer_kwargs(requirements: VastOfferRequirements) -> Dict[str, Any]:
    """Return explicit SDK arguments shared by feasibility and provisioning."""
    kwargs = {
        'query': build_offer_query(requirements),
        'order': 'min_bid' if requirements.use_spot else 'dph_total',
        'type': 'bid' if requirements.use_spot else 'on-demand',
        'storage': requirements.disk_size,
        'no_default': True,
    }
    if requirements.requested_accelerator_name is not None:
        kwargs['limit'] = _DIRECT_SEARCH_LIMIT
    return kwargs


def search_offers(requirements: VastOfferRequirements) -> Any:
    """Search Vast without SDK-added host or shape restrictions."""
    return vast().search_offers(**_search_offer_kwargs(requirements))


def offer_matches_requirements(offer: Any,
                               requirements: VastOfferRequirements) -> bool:
    """Return whether a live offer satisfies every SkyPilot Vast policy."""
    return _offer_rejection_reason(offer, requirements) is None


@annotations.lru_cache(scope='request')
def get_live_offer_matches(
        requirements: VastOfferRequirements) -> LiveOfferQueryResult:
    """Fetch and locally validate targeted live offers for this requirement."""
    try:
        offers = search_offers(requirements)
    except Exception as exc:  # pylint: disable=broad-except
        return LiveOfferQueryResult(
            offers=(),
            error=f'Vast live-offer query failed ({type(exc).__name__}).',
            offers_examined=0,
            rejection_counts=(),
        )
    if not isinstance(offers, list):
        return LiveOfferQueryResult(
            offers=(),
            error=('Vast returned an unexpected live-offer response.'),
            offers_examined=0,
            rejection_counts=(),
        )
    matching_offers = []
    rejection_counts: Dict[str, int] = {}
    for offer in offers:
        rejection_reason = _offer_rejection_reason(offer, requirements)
        if rejection_reason is None:
            matching_offers.append(offer)
            continue
        rejection_counts[rejection_reason] = (
            rejection_counts.get(rejection_reason, 0) + 1)
    return LiveOfferQueryResult(
        offers=tuple(matching_offers),
        error=None,
        offers_examined=len(offers),
        rejection_counts=tuple(sorted(rejection_counts.items())),
    )
