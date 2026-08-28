"""Channel implementations.

Importing this package registers every built-in channel. New channels:
add a module here implementing :class:`sub2api.core.channel.BaseChannel`,
decorate it with ``@register``, and import it below — no server changes.
"""

from . import adal_backend, adal_cli, adal_cloud, adal_sdk, echo  # noqa: F401  (import = registration)
