mode: after
anchor: - If the same behaviour has parallel implementations (sync and async variants
---
- Twin checklist, before any edit ask where its twin is: (1) sync vs async (Client.request and AsyncClient.request, def route and async def route): share one pure helper; (2) dict or container methods (get, __getitem__, __setitem__, __contains__, copy): route all of them through one key-normalising helper; (3) function entry vs class entry (a top-level get or lookup function vs Session.get or a registry class): the function delegates to the class; (4) serialize vs deserialize (to_dict, dump, encode vs from_dict, load, decode): check the round trip; (5) module definition vs `__init__.py` / `__all__` re-export: run `git grep -n '__all__'`.
