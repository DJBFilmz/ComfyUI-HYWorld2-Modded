import json


def loads_repaired(text):
    raw = str(text or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    if not raw:
        raise ValueError("Cannot parse empty JSON text.")
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        from json_repair import repair_json
    except Exception as exc:
        raise ValueError("Invalid JSON and json-repair is not installed.") from exc
    try:
        return repair_json(raw, return_objects=True)
    except TypeError:
        return json.loads(repair_json(raw))


def load_repaired(handle):
    try:
        return json.load(handle)
    except Exception:
        handle.seek(0)
        return loads_repaired(handle.read())


def dump_json(data, handle, indent=2):
    json.dump(data, handle, indent=indent)
