"""Regression tests for mutually compatible SkyPilot image profiles."""

from sky.setup_files import dependencies


def test_all_except_azure_profile_includes_vast_without_azure_cli():
    """Ensure the Vast server profile avoids Azure's conflicting CLI stack."""
    profile = dependencies.extras_require['all-except-azure']

    assert 'vastai-sdk==1.5.0' in profile
    assert not any(
        requirement.startswith('azure-cli') for requirement in profile)
