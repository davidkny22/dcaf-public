"""Framework output assembly (§13, Def 13.1-13.4).

Two complementary assemblers:
- dcaf.output.results: In-memory component-level assembler (ComponentOutput, assemble_output)
- dcaf.output.schema: Full-run JSON schema assembler (assemble_output, validate_output)

Both expose assemble_output and assemble_component_output with different signatures.
Import directly from the submodule when you need a specific variant.
"""

from dcaf.output.results import (
    ComponentOutput,
    assemble_component_output as assemble_component_output_simple,
    assemble_output as assemble_output_simple,
)
from dcaf.output.schema import (
    assemble_output,
    assemble_component_output,
    assemble_discovery_summary,
    assemble_domain_summary,
    validate_output,
)

__all__ = [
    # In-memory assembler (dcaf.output.results)
    "ComponentOutput",
    "assemble_component_output_simple",
    "assemble_output_simple",
    # Full-run JSON schema assembler (dcaf.output.schema)
    "assemble_output",
    "assemble_component_output",
    "assemble_discovery_summary",
    "assemble_domain_summary",
    "validate_output",
]
