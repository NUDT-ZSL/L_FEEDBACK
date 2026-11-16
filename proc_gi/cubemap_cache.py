from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, Sequence, TypeVar

from .clustering import VirtualPointLight

ResourceT = TypeVar("ResourceT")


@dataclass
class CacheEntry(Generic[ResourceT]):
    vpl: VirtualPointLight
    resource: ResourceT


@dataclass
class CacheSyncResult(Generic[ResourceT]):
    entries: dict[str, CacheEntry[ResourceT]]
    previous_entries: dict[str, CacheEntry[ResourceT]]
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    stale_resources: list[ResourceT] = field(default_factory=list)

    def release_stale(self, release: Callable[[ResourceT], None]) -> None:
        active_ids = {id(entry.resource) for entry in self.entries.values()}
        released_ids: set[int] = set()
        for resource in self.stale_resources:
            identity = id(resource)
            if identity in active_ids or identity in released_ids:
                continue
            release(resource)
            released_ids.add(identity)
        self.stale_resources.clear()


class VPLCubeScheduler(Generic[ResourceT]):
    """Signature-versioned cubemap resource cache.

    Changed VPLs receive a fresh resource so the previous cubemap remains valid
    for the one-frame incremental subtraction pass. The stale resource is
    released explicitly after accumulation.
    """

    def __init__(self) -> None:
        self.entries: dict[str, CacheEntry[ResourceT]] = {}

    def sync(
        self,
        vpls: Sequence[VirtualPointLight],
        create: Callable[[VirtualPointLight], ResourceT],
    ) -> CacheSyncResult[ResourceT]:
        previous = dict(self.entries)
        result = CacheSyncResult(entries={}, previous_entries=previous)
        incoming_ids = {vpl.id for vpl in vpls}
        for vpl in sorted(vpls, key=lambda item: item.id):
            old = previous.get(vpl.id)
            if old is not None and old.vpl.signature == vpl.signature:
                result.entries[vpl.id] = old
                result.reused.append(vpl.id)
                continue
            resource = create(vpl)
            result.entries[vpl.id] = CacheEntry(vpl=vpl, resource=resource)
            if old is None:
                result.added.append(vpl.id)
            else:
                result.changed.append(vpl.id)
                result.stale_resources.append(old.resource)
        for resource_id, old in previous.items():
            if resource_id not in incoming_ids:
                result.removed.append(resource_id)
                result.stale_resources.append(old.resource)
        self.entries = result.entries
        return result

    def clear(self, release: Callable[[ResourceT], None]) -> None:
        seen: set[int] = set()
        for entry in self.entries.values():
            identity = id(entry.resource)
            if identity not in seen:
                release(entry.resource)
                seen.add(identity)
        self.entries.clear()

