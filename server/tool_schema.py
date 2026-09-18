"""Tool and response-format schema normalization and llguidance grammars."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field

from jsonschema.exceptions import SchemaError
from referencing import Registry

if __package__:
    from .errors import APIError
    from .schema_validation import build_validator
else:  # ``python server/server.py`` from the repo root.
    from errors import APIError
    from schema_validation import build_validator

MAX_JSON_NESTING = 256

# The chat template's tool-call framing. The projector, the parser and the
# grammars must agree byte for byte, so every piece is spelled here once.
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
FUNCTION_OPEN = "\n<function="
FUNCTION_CLOSE = "</function>\n</tool_call>"
PARAMETER_OPEN = "<parameter="
PARAMETER_CLOSE = "\n</parameter>\n"
THINK_END_TOKEN_ID = 248069  # the chat template's think-close token


LOCAL_REGISTRY = Registry()


def _reject_json_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def strict_json_loads(value):
    return json.loads(value, parse_constant=_reject_json_constant)


@dataclass(frozen=True)
class ToolPolicy:
    validators: dict
    schemas: dict
    required: bool
    parallel: bool
    namespaces: dict = field(default_factory=dict)


def json_value(value):
    value = value.strip()
    try:
        parsed = json.loads(value, parse_constant=str)
        pending = [(parsed, 0)]
        while pending:
            item, depth = pending.pop()
            if isinstance(item, dict):
                children = item.values()
            elif isinstance(item, list):
                children = item
            else:
                continue
            depth += 1
            if depth > MAX_JSON_NESTING:
                return value
            pending.extend((child, depth) for child in children)
        json.dumps(parsed, allow_nan=False)
        return parsed
    except (ValueError, RecursionError):
        return value


# JSON Schema keywords whose values are schemas: maps from names to schemas,
# then single schemas or lists of schemas. ``dependencies`` holds a schema or
# a list of property names per entry and is told apart by shape.
SCHEMA_MAP_KEYWORDS = {
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
}
SUBSCHEMA_KEYWORDS = {
    "items",
    "prefixItems",
    "additionalItems",
    "contains",
    "additionalProperties",
    "unevaluatedItems",
    "unevaluatedProperties",
    "propertyNames",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "contentSchema",
}


def _schemas(schema):
    """Yield ``schema`` and, depth first, every schema nested under it.

    Only schema positions are visited, so property names and literal const,
    enum, default and examples data are never mistaken for schemas.
    """
    yield schema
    if not isinstance(schema, dict):
        return
    for key, item in schema.items():
        if key in SCHEMA_MAP_KEYWORDS and isinstance(item, dict):
            children = item.values()
        elif key == "dependencies" and isinstance(item, dict):
            children = (child for child in item.values() if not isinstance(child, list))
        elif key in SUBSCHEMA_KEYWORDS:
            children = item if isinstance(item, list) else (item,)
        else:
            continue
        for child in children:
            yield from _schemas(child)


def _remote_ref(schema):
    for node in _schemas(schema):
        if isinstance(node, dict):
            if "$schema" in node and not isinstance(node["$schema"], str):
                raise APIError(400, "$schema must be a string")
            for key in ("$ref", "$dynamicRef", "$recursiveRef"):
                ref = node.get(key)
                if isinstance(ref, str) and not ref.startswith("#"):
                    return ref
    return None


SCHEMA_ANNOTATIONS = {
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
}


# Raw string parameters are framed by the tool-call grammar and validated
# against their complete JSON Schema after parsing. Keeping these assertions
# out of the grammar preserves multiline string payloads.
STRING_SCHEMA_POST_VALIDATION_KEYWORDS = {
    "allOf",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
}


def _grammar_compatible_schema(schema):
    """Guide generation with supported constraints; validate the original."""
    output = copy.deepcopy(schema)
    for node in _schemas(output):
        if isinstance(node, dict):
            node.pop("propertyNames", None)
            node.pop("pattern", None)
    if isinstance(output, dict):
        output["x-guidance"] = {"lenient": True}
    return output


def _resolve_tool_schema(schema, root):
    seen = set()
    while isinstance(schema, dict) and "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str) or not (ref == "#" or ref.startswith("#/")):
            raise APIError(400, "unsupported tool parameter reference")
        constrained = bool(set(schema) - {"$ref"} - SCHEMA_ANNOTATIONS)
        if ref in seen:
            raise APIError(400, "cyclic direct tool parameter reference")
        seen.add(ref)
        current = root
        if ref != "#":
            for part in ref[2:].split("/"):
                part = part.replace("~1", "/").replace("~0", "~")
                if not isinstance(current, dict) or part not in current:
                    raise APIError(400, f"unresolved tool parameter reference: {ref}")
                current = current[part]
        if constrained:
            # Keep the reference and its intersecting assertions together in
            # the JSON grammar instead of choosing a raw-string encoding.
            return None
        schema = current
    return schema


def _schema_with_root(schema, root):
    def local_refs(value):
        for node in _schemas(value):
            ref = node.get("$ref") if isinstance(node, dict) else None
            if isinstance(ref, str) and ref.startswith("#"):
                yield node

    if next(local_refs(schema), None) is None:
        return schema
    root_defs = root.get("$defs", {}) if isinstance(root, dict) else {}
    schema_defs = schema.get("$defs", {}) if isinstance(schema, dict) else {}
    name = "__splash_root"
    while name in root_defs or name in schema_defs:
        name += "_"
    prefix = f"#/$defs/{name}"

    def rebase(value):
        output = copy.deepcopy(value)
        for node in local_refs(output):
            node["$ref"] = prefix + node["$ref"][1:]
        return output

    output = rebase(schema)
    definitions = dict(output.get("$defs", {}))
    definitions[name] = rebase(root)
    output["$defs"] = definitions
    return output


def raw_string_schema(schema, root):
    schema = _resolve_tool_schema(schema, root)
    if not isinstance(schema, dict):
        return None
    union = schema.get("anyOf", schema.get("oneOf"))
    if union is not None:
        options = []
        has_other_type = False
        allows_null = False
        for option_schema in union:
            resolved = _resolve_tool_schema(option_schema, root)
            null_only = isinstance(resolved, dict) and (
                resolved.get("type") == "null"
                or resolved.get("const", object()) is None
                or (
                    isinstance(resolved.get("enum"), list)
                    and resolved["enum"]
                    and all(value is None for value in resolved["enum"])
                )
            )
            if null_only:
                allows_null = True
                continue
            option = raw_string_schema(option_schema, root)
            if option is None:
                has_other_type = True
            else:
                options.append(option)
        if not options:
            return None
        if has_other_type:
            return None
        if allows_null:
            return None
        if set(schema) - {"anyOf", "oneOf"} - SCHEMA_ANNOTATIONS:
            return None
        if any(option[0] == "raw" for option in options):
            return "raw", None
        values = sum((option[1] for option in options), [])
        return "literal", values

    schema_type = schema.get("type")
    if isinstance(schema_type, list) and "string" in schema_type:
        if set(schema_type) - {"string", "null"}:
            return None
        if "null" in schema_type:
            return None
        schema_type = "string"
    values = schema.get("enum")
    if "const" in schema:
        values = [schema["const"]]
    if schema_type == "null":
        return None
    if schema_type != "string" and not (
        isinstance(values, list) and any(isinstance(value, str) for value in values)
    ):
        return None
    unsupported = (
        set(schema)
        - {
            "type",
            "enum",
            "const",
            "$defs",
            "definitions",
        }
        - SCHEMA_ANNOTATIONS
        - STRING_SCHEMA_POST_VALIDATION_KEYWORDS
    )
    if unsupported:
        return None
    if values is None:
        return "raw", None
    if any(value is not None and not isinstance(value, str) for value in values):
        return None
    if None in values:
        return None
    if any(
        isinstance(value, str) and PARAMETER_CLOSE.rstrip("\n") in value
        for value in values
    ):
        raise APIError(400, "string tool parameter enum contains XML framing")
    return "literal", values


def _tool_arguments_grammar(schema):
    if schema is True:
        schema = {}
    if not isinstance(schema, dict):
        raise APIError(400, "tool parameters must allow a JSON object")
    schema_type = schema.get("type")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    unsupported = set(schema) & {
        "$ref",
        "$dynamicRef",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "dependentRequired",
        "dependentSchemas",
        "enum",
        "const",
        "minProperties",
        "maxProperties",
        "patternProperties",
    }
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise APIError(400, f"unsupported top-level tool schema keyword(s): {names}")
    if schema_type not in (None, "object") or not isinstance(properties, dict):
        raise APIError(400, "tool parameters must be a top-level JSON object")
    if any(name not in properties for name in required):
        raise APIError(400, "required tool parameters must declare properties")
    rules = []
    sequence = []
    for index, (name, value_schema) in enumerate(properties.items()):
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or any(character in name for character in "<>\n\r")
        ):
            raise APIError(400, "invalid tool parameter name")
        string_schema = raw_string_schema(value_schema, schema)
        value_schema = _schema_with_root(value_schema, schema)
        value_schema = _grammar_compatible_schema(value_schema)
        rule = f"parameter_{index}"
        suffix = "" if name in required else "?"
        sequence.append(rule + suffix)
        prefix = json.dumps(f"{PARAMETER_OPEN}{name}>\n")
        closing = json.dumps(PARAMETER_CLOSE)
        if string_schema is None:
            rules.append(
                f"{rule}: {prefix} "
                f"%json {json.dumps(value_schema, separators=(',', ':'))} "
                f"{closing}"
            )
        elif string_schema[0] == "raw":
            value_rule = f"{rule}_value"
            rules.append(f"{rule}: {prefix} {value_rule}")
            rules.append(f"{value_rule}[suffix={closing}]: /(?s:.*)/")
        else:
            choices = []
            for choice_index, value in enumerate(string_schema[1]):
                text = "null" if value is None else value
                if text:
                    choices.append(json.dumps(text))
                else:
                    empty_rule = f"{rule}_empty_{choice_index}"
                    rules.append(f"{empty_rule}:")
                    choices.append(empty_rule)
            rules.append(f"{rule}: {prefix} ({' | '.join(choices)}) {closing}")
    start = " ".join(sequence)
    return (
        "%llguidance {}\nstart:"
        + (" " + start if start else "")
        + "\n"
        + "\n".join(rules)
        + "\n"
    )


def json_grammar(schema, thinking):
    start = "start: " + ("think " if thinking else "") + "WS %json "
    grammar = [
        "%llguidance {}",
        start + json.dumps(_grammar_compatible_schema(schema), separators=(",", ":")),
    ]
    if thinking:
        grammar.append(f"think: TEXT <[{THINK_END_TOKEN_ID}]>")
        grammar.append("TEXT: /(.|\\n)*/")
    grammar.append("WS: /[ \\n\\r\\t]*/")
    return "\n".join(grammar) + "\n"


def normalize_response_format(value):
    if value in (None, {"type": "text"}):
        return None, None
    if not isinstance(value, dict):
        raise APIError(400, "response_format must be an object")
    kind = value.get("type")
    if kind == "json_object":
        schema = {"type": "object"}
    elif kind == "json_schema":
        wrapper = value.get("json_schema")
        schema = wrapper.get("schema") if isinstance(wrapper, dict) else None
        if not isinstance(schema, (dict, bool)):
            raise APIError(400, "response_format.json_schema.schema is required")
    else:
        raise APIError(400, "unsupported response_format")
    if ref := _remote_ref(schema):
        raise APIError(400, f"remote schema reference is not allowed: {ref}")
    try:
        validator = build_validator(schema, _schemas, LOCAL_REGISTRY)
    except SchemaError as error:
        raise APIError(400, f"invalid response schema: {error.message}") from error
    return schema, validator


def normalize_tools(tools, tool_choice, parallel, namespaces=None):
    if parallel is None:
        parallel = True
    if not isinstance(parallel, bool):
        raise APIError(400, "parallel_tool_calls must be a boolean")
    if tools is None:
        if tool_choice not in (None, "none", "auto"):
            raise APIError(400, "tool_choice requires tools")
        return None, None
    if not isinstance(tools, list):
        raise APIError(400, "tools must be an array")
    validators = {}
    schemas = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise APIError(400, "only function tools are supported")
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name) is None
        ):
            raise APIError(400, "tool name must match [A-Za-z0-9_-]{1,128}")
        if name in validators:
            raise APIError(400, f"duplicate tool name: {name}")
        schema = function.get("parameters")
        if schema is None:
            schema = {}
        if not isinstance(schema, (dict, bool)):
            raise APIError(400, f"invalid tool schema for {name}")
        if ref := _remote_ref(schema):
            raise APIError(400, f"remote tool schema reference is not allowed: {ref}")
        try:
            validators[name] = build_validator(schema, _schemas, LOCAL_REGISTRY)
            schemas[name] = schema
        except SchemaError as error:
            raise APIError(
                400, f"invalid tool schema for {name}: {error.message}"
            ) from error
    choice = "auto" if tool_choice is None else tool_choice
    if not tools:
        if choice not in ("auto", "none"):
            raise APIError(400, "tool_choice requires at least one tool")
        return None, None
    if choice == "none":
        return None, None
    if isinstance(choice, dict):
        function = choice.get("function", {})
        name = function.get("name") if isinstance(function, dict) else None
        if (
            choice.get("type") != "function"
            or not isinstance(name, str)
            or name not in validators
        ):
            raise APIError(400, "invalid named tool_choice")
        # The prompt keeps every tool; the grammar and validators force the call.
        validators = {name: validators[name]}
        schemas = {name: schemas[name]}
    elif choice not in ("auto", "required"):
        raise APIError(400, "invalid tool_choice")
    policy = ToolPolicy(
        validators,
        schemas,
        choice == "required" or isinstance(choice, dict),
        parallel,
        namespaces or {},
    )
    return tools, policy


THINK_END = "</think>"


def tool_grammar(policy, thinking, response_schema=None):
    side_grammars = []
    tag_rules = []
    for index, (name, schema) in enumerate(policy.schemas.items()):
        grammar_name = f"arguments_{index}"
        side_grammars.append(
            {"name": grammar_name, "lark_grammar": _tool_arguments_grammar(schema)}
        )
        tag_rules.append(
            f"tool_{index}: {'WS' if response_schema is not None else 'TEXT'} {TOOL_CALL_OPEN} "
            f"{json.dumps(FUNCTION_OPEN + name + '>' + chr(10))} "
            f"@{grammar_name} {json.dumps(FUNCTION_CLOSE.removesuffix(TOOL_CALL_CLOSE))} "
            f"{TOOL_CALL_CLOSE}"
        )
    tool_choice = (
        "(" + " | ".join(f"tool_{index}" for index in range(len(tag_rules))) + ")"
    )
    thinking_prefix = "think " if thinking else ""
    if response_schema is not None:
        calls = tool_choice + ("+" if policy.parallel else "") + " WS"
        body = calls if policy.required else f"({calls} | answer)"
        start = f"start: {thinking_prefix}{body}"
    elif policy.required:
        body = tool_choice + ("+" if policy.parallel else "")
        start = f"start: {thinking_prefix}{body}"
    else:
        body = tool_choice + ("*" if policy.parallel else "?")
        start = f"start: {thinking_prefix}{body} tail"
    main = ["%llguidance {}", start]
    if response_schema is not None:
        main.extend(
            [
                "answer: WS %json "
                + json.dumps(
                    _grammar_compatible_schema(response_schema), separators=(",", ":")
                )
                + " WS",
                r"WS: /[ \n\r\t]*/",
            ]
        )
    if thinking:
        main.append(f"think: TEXT <[{THINK_END_TOKEN_ID}]>")
    main.extend(
        [
            "tail: TEXT",
            *tag_rules,
            r"TEXT: /(?s:.*)/ & ~/(?s:.*)(<tool_call>|<\/think>)(?s:.*)/",
        ]
    )
    side_grammars.insert(
        0, {"name": "tool_output", "lark_grammar": "\n".join(main) + "\n"}
    )
    return json.dumps({"grammars": side_grammars}, separators=(",", ":"))
