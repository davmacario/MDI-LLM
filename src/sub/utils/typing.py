from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

FileType = Union[str, Path]


# Define JSON value types
JSONValue = Union[str, int, float, bool, None]

# Define JSON object types
JSONObject = Dict[
    str, "JSONType"
]  # A JSON object with string keys and values of JSONType
JSONArray = List["JSONType"]  # A JSON array with values of JSONType

# Define the overarching JSON type that includes all valid JSON structures
JSONType = Union[JSONValue, JSONObject, JSONArray]
