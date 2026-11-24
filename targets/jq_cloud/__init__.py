"""Offline JoinQuant deployment target."""
from targets.jq_cloud.compatibility import TARGET_ID, check_bundle_compatibility
from targets.jq_cloud.compiler import export_bundle
from targets.jq_cloud.verifier import verify_bundle_alignment


class JqCloudTarget:
    """Compile A-share strategy bundles and verify their virtual cloud replay."""

    target_id = TARGET_ID
    check_compatibility = staticmethod(check_bundle_compatibility)
    export = staticmethod(export_bundle)
    verify_alignment = staticmethod(verify_bundle_alignment)


__all__ = ["JqCloudTarget"]
