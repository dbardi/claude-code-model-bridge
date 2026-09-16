"""Reads one string field out of JSON that is still arriving."""

import json


class StringFieldReader:
    """Yields a JSON string field's characters as the surrounding JSON arrives.

    Claude streams schema-validated answers as fragments of one JSON object,
    so the prose a caller should see live is buried inside unfinished JSON.
    This reads that one field and emits only characters that are complete:
    a fragment ending mid-escape (`\\u00e9` split across two deltas) waits
    until the escape resolves rather than emitting a broken character.
    """

    def __init__(self, field: str) -> None:
        self._opening = json.dumps(field) + ":"
        self._buffer = ""
        self._reading = False
        self._done = False

    def feed(self, fragment: str) -> str:
        """Adds a fragment, returning whatever text became complete."""
        if self._done:
            return ""
        self._buffer += fragment
        if not self._reading and not self._start_reading():
            return ""
        return self._take()

    def _start_reading(self) -> bool:
        """Finds the field's opening quote, discarding everything before it."""
        collapsed = self._buffer.replace(" ", "")
        if self._opening not in collapsed:
            return False
        index = self._buffer.find(self._opening.rstrip(":"))
        rest = self._buffer[index + len(self._opening.rstrip(":")) :]
        quote = rest.find('"', rest.find(":") + 1 if ":" in rest else 0)
        if quote == -1:
            return False
        self._buffer = rest[quote + 1 :]
        self._reading = True
        return True

    def _take(self) -> str:
        """Decodes as much of the string as is unambiguously complete."""
        text = ""
        while self._buffer:
            character = self._buffer[0]
            if character == '"':
                self._buffer = ""
                self._done = True
                break
            if character == "\\":
                escape = self._escape()
                if escape is None:
                    break
                text += escape
                continue
            text += character
            self._buffer = self._buffer[1:]
        return text

    def _escape(self) -> str | None:
        """Decodes one escape sequence, or None while it is still incomplete."""
        length = 6 if self._buffer[1:2] == "u" else 2
        if len(self._buffer) < length:
            return None
        decoded = json.loads(f'"{self._buffer[:length]}"')
        self._buffer = self._buffer[length:]
        return decoded
