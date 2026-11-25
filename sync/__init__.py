"""sync package of the Game Translation HUD: sharing a translation database through a link."""

from sync.download import SyncError, database_info, direct_url, download, sync_database

__all__ = ["SyncError", "database_info", "direct_url", "download", "sync_database"]
