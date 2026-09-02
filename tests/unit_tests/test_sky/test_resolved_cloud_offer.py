"""Regression tests for catalog-independent resolved cloud offers."""

import pickle
from unittest import mock

import pytest

from sky import clouds
from sky import dag as dag_lib
from sky import resources as resources_lib
from sky import task as task_lib
from sky.utils import dag_utils
from sky.utils import resources_utils

_OFFER = {
    'schema_version': 1,
    'provider': 'runpod',
    'instance_type': '1x_RTXPRO6000-WK_SECURE',
    'region': 'CZ',
    'accelerator_name': 'RTXPRO6000-WK',
    'accelerator_count': 1,
    'cpu_count': 16,
    'memory_gib': 96,
    'device_memory_gib': 96,
    'hourly_price': 1.89,
    'use_spot': False,
}


def _matching_resource(**overrides):
    values = {
        'cloud': clouds.RunPod(),
        'instance_type': _OFFER['instance_type'],
        'region': _OFFER['region'],
        'accelerators': {
            _OFFER['accelerator_name']: _OFFER['accelerator_count']
        },
        'use_spot': _OFFER['use_spot'],
    }
    values.update(overrides)
    return resources_lib.Resources(**values)


def test_resolved_cloud_offer_round_trips_in_task_metadata_and_pickle():
    """A selected offer stays JSON metadata and survives task/resource reloads."""
    with dag_lib.Dag() as dag:
        original = task_lib.Task(resources=_matching_resource())
        original.set_resolved_cloud_offer(_OFFER)

    resource = next(iter(original.resources))
    assert resource.resolved_cloud_offer == resources_lib.ResolvedCloudOffer(
        **_OFFER)
    assert original.metadata['resolved_cloud_offer'] == _OFFER

    yaml_config = original.to_yaml_config()
    assert '_resolved_cloud_offer' not in yaml_config['resources']
    reloaded = task_lib.Task.from_yaml_config(yaml_config)
    reloaded_resource = next(iter(reloaded.resources))
    assert reloaded_resource.resolved_cloud_offer == resource.resolved_cloud_offer
    reloaded_dag = dag_utils.load_chain_dag_from_yaml_str(
        dag_utils.dump_dag_to_yaml_str(dag))
    controller_resource = next(iter(reloaded_dag.tasks[0].resources))
    assert controller_resource.resolved_cloud_offer == resource.resolved_cloud_offer
    assert resource.copy().resolved_cloud_offer == resource.resolved_cloud_offer
    assert pickle.loads(pickle.dumps(resource)).resolved_cloud_offer == (
        resource.resolved_cloud_offer)


@pytest.mark.parametrize('resource_overrides', [
    {
        'cloud': clouds.AWS()
    },
    {
        'instance_type': '1x_DIFFERENT_SECURE'
    },
    {
        'region': 'US'
    },
    {
        'accelerators': {
            'A100': 1
        }
    },
    {
        'accelerators': {
            _OFFER['accelerator_name']: 2
        }
    },
    {
        'use_spot': True
    },
])
def test_resolved_cloud_offer_rejects_nonmatching_resource(resource_overrides):
    """Provider, placement, accelerator, and spot mismatches fail closed."""
    task = task_lib.Task(resources=_matching_resource(**resource_overrides))

    with pytest.raises(ValueError, match='match exactly one task resource'):
        task.set_resolved_cloud_offer(_OFFER)


def test_resolved_cloud_offer_rejects_malformed_transport_mapping():
    """The fixed transport shape rejects unknown fields and raw payloads."""
    malformed_offer = dict(_OFFER, raw_provider_response={'token': 'secret'})

    with pytest.raises(ValueError, match='Malformed resolved cloud offer'):
        resources_lib.ResolvedCloudOffer.from_json_mapping(malformed_offer)


def test_runpod_resolved_offer_never_reads_missing_local_catalog():
    """Offer-backed RunPod controller preparation uses offer data and live zones."""
    cloud = clouds.RunPod()
    with dag_lib.Dag() as dag:
        task = task_lib.Task(resources=_matching_resource())
        task.set_resolved_cloud_offer(_OFFER)
    resource = next(iter(task.resources))

    with mock.patch.object(
            type(cloud),
            'get_vcpus_mem_from_instance_type',
            side_effect=AssertionError('local catalog must not be read')), \
         mock.patch.object(
             type(cloud),
             'get_accelerators_from_instance_type',
             side_effect=AssertionError('local catalog must not be read')), \
         mock.patch.object(
             type(cloud),
             'instance_type_to_hourly_cost',
             side_effect=AssertionError('local catalog must not be read')), \
         mock.patch.object(
             type(cloud),
             'instance_type_exists',
             side_effect=AssertionError('local catalog must not be read')), \
         mock.patch(
             'sky.clouds.runpod.catalog.get_region_zones_for_instance_type',
             side_effect=AssertionError('local catalog must not be read')), \
         mock.patch(
             'sky.resources.accelerator_registry.canonicalize_accelerator_name',
             side_effect=AssertionError('local catalog must not be read')), \
         mock.patch(
             'sky.adaptors.runpod.available_data_center_ids_for_instance_type',
             return_value={'CZ-1'}) as live_capacity:
        resource.validate()
        dag.validate(skip_file_mounts=True, skip_workdir=True)
        assert 'RTXPRO6000-WK' in repr(dag)
        assert 'RTXPRO6000-WK' in repr(resource)
        assert resource.cpus == '16'
        assert resource.memory == '96'
        assert resource.accelerators == {'RTXPRO6000-WK': 1}
        assert resource.get_cost(3600) == 1.89
        feasible = cloud._get_feasible_launchable_resources(resource)
        assert feasible.resources_list == [resource]

        regions = resource.get_valid_regions_for_launchable()
        assert [(region.name, region.zones) for region in regions] == [('CZ',
                                                                        [])]
        zones = list(
            cloud.zones_provision_loop_for_resources(resource,
                                                     region='CZ',
                                                     num_nodes=1))
        assert [[zone.name for zone in zone_list] for zone_list in zones
               ] == [['CZ-1']]
        live_capacity.assert_called_once_with(resource.instance_type, ['CZ'],
                                              force_refresh=True)

        deploy_variables = cloud.make_deploy_resources_variables(
            resources=resource,
            cluster_name=resources_utils.ClusterName('offer', 'offer'),
            region=clouds.Region('CZ'),
            zones=[clouds.Zone('CZ-1')],
            num_nodes=1,
        )

    assert deploy_variables['availability_zone'] == 'CZ-1'
    assert deploy_variables['bid_per_gpu'] == '1.89'


def test_runpod_resolved_offer_treats_unavailable_live_capacity_as_no_zones():
    """A failed live lookup must yield no capacity without falling back to CSV."""
    cloud = clouds.RunPod()
    task = task_lib.Task(resources=_matching_resource())
    task.set_resolved_cloud_offer(_OFFER)
    resource = next(iter(task.resources))

    with mock.patch(
            'sky.adaptors.runpod.available_data_center_ids_for_instance_type',
            return_value=None), mock.patch(
                'sky.clouds.runpod.catalog.get_region_zones_for_instance_type',
                side_effect=AssertionError('local catalog must not be read')):
        zones = list(
            cloud.zones_provision_loop_for_resources(resource,
                                                     region='CZ',
                                                     num_nodes=1))

    assert zones == []
