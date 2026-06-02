"""Reference serializers.

First-party adapters subclass their port explicitly (CONTRIBUTING convention;
mypy completeness-checks the implementation) — so these declare
``Serializer``, like every other adapter family.

This package is a façade: implementations live in sibling modules; import the
public names here.
"""

from denselinkage.serialize.fieldwise_serializer import FieldwiseSerializer
from denselinkage.serialize.template_serializer import TemplateSerializer
from denselinkage.serialize.whole_row_serializer import (
    WholeRowSerializer,
    default_serializer,
)

__all__ = [
    "FieldwiseSerializer",
    "TemplateSerializer",
    "WholeRowSerializer",
    "default_serializer",
]
