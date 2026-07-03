import json

def repair_json_unescaped_quotes(s: str) -> str:
    result = []
    in_string = False
    escape = False
    n = len(s)

    def next_nonspace_char(idx):
        while idx < n and s[idx].isspace():
            idx += 1
        return s[idx] if idx < n else ''

    for i, ch in enumerate(s):
        if not in_string:
            if ch == '"':
                in_string = True
            result.append(ch)
            continue

        if escape:
            result.append(ch)
            escape = False
            continue

        if ch == '\\':
            result.append(ch)
            escape = True
            continue

        if ch == '"':
            nxt = next_nonspace_char(i + 1)
            if nxt in [',', '}', ']', ':', '']:
                in_string = False
                result.append(ch)
            else:
                result.append('\\"')
            continue

        result.append(ch)

    return ''.join(result)


def safe_json_loads(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        repaired = repair_json_unescaped_quotes(s)
        return json.loads(repaired)
