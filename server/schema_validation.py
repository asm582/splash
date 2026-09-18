"""JSON Schema validation with bounded regular-expression evaluation."""

import copy
from functools import lru_cache

import regex
from jsonschema import ValidationError, validators

if __package__:
    from .errors import APIError
else:
    from errors import APIError


class SchemaEvaluationError(Exception):
    """A validator could not evaluate an already accepted schema."""


def _matches(pattern, value):
    try:
        return regex.search(pattern, value, timeout=0.05, concurrent=True) is not None
    except TimeoutError:
        raise SchemaEvaluationError(
            "schema pattern exceeded its evaluation time limit"
        ) from None
    except regex.error:
        raise SchemaEvaluationError("schema pattern could not be evaluated") from None


def _pattern(validator, pattern, instance, schema):
    if validator.is_type(instance, "string") and not _matches(pattern, instance):
        yield ValidationError("string does not match its pattern")


def _pattern_properties(validator, patterns, instance, schema):
    if validator.is_type(instance, "object"):
        for pattern, subschema in patterns.items():
            for key, value in instance.items():
                if _matches(pattern, key):
                    yield from validator.descend(
                        value, subschema, path=key, schema_path=pattern
                    )


def _additional_properties(validator, additional, instance, schema):
    if validator.is_type(instance, "object"):
        properties = schema.get("properties", {})
        patterns = schema.get("patternProperties", {})
        for key, value in instance.items():
            if key not in properties and not any(
                _matches(pattern, key) for pattern in patterns
            ):
                if additional is False:
                    yield ValidationError(
                        "additional property is not allowed", path=[key]
                    )
                elif isinstance(additional, dict):
                    yield from validator.descend(value, additional, path=key)


@lru_cache(maxsize=8)
def _bounded_class(base):
    return validators.extend(
        base,
        {
            "pattern": _pattern,
            "patternProperties": _pattern_properties,
            "additionalProperties": _additional_properties,
        },
    )


def build_validator(schema, nodes, registry):
    base = validators.validator_for(schema)
    base.check_schema(schema)
    validated = copy.deepcopy(schema)
    # A document uses one dialect. Removing identical declarations prevents
    # jsonschema.evolve from replacing the bounded class at a local reference.
    for node in nodes(validated):
        if isinstance(node, dict) and "$schema" in node:
            if validators.validator_for(node) is not base:
                raise APIError(400, "mixed schema dialects are not supported")
            node.pop("$schema")
    return _bounded_class(base)(validated, registry=registry)
