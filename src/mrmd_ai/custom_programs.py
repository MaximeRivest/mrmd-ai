"""
Custom Program Factory - Generate DSPy modules from user-defined templates.

This module allows users to create custom AI commands without writing code.
Users define their commands with:
- name: Display name for the command
- inputType: What text to process (selection, cursor, fullDoc)
- outputType: What to do with result (replace, insert)
- instructions: Natural language instructions for the AI

The factory generates DSPy Signature and Module classes dynamically.
"""

from typing import Any
import dspy


def create_custom_signature(
    name: str,
    instructions: str,
    input_type: str = "selection",
    output_type: str = "replace",
) -> type:
    """Create a DSPy Signature class from user configuration.

    Args:
        name: Command name (used for class naming)
        instructions: User's instructions for the AI
        input_type: "selection" | "cursor" | "fullDoc"
        output_type: "replace" | "insert"

    Returns:
        A DSPy Signature class
    """
    # Build the docstring from user instructions
    docstring = f"""{instructions}

IMPORTANT RULES:
- Output ONLY the result text, no explanations or meta-commentary
- Maintain appropriate formatting (markdown, code style, etc.)
- Be concise and direct
"""

    # Define input fields based on input type
    if input_type == "selection":
        input_fields = {
            "text": dspy.InputField(desc="The selected text to process"),
            "local_context": dspy.InputField(desc="Text surrounding the selection for context"),
            "document_context": dspy.InputField(desc="Broader document context"),
        }
    elif input_type == "cursor":
        input_fields = {
            "text_before_cursor": dspy.InputField(desc="Text before the cursor position"),
            "local_context": dspy.InputField(desc="Text surrounding the cursor for context"),
            "document_context": dspy.InputField(desc="Broader document context"),
        }
    else:  # fullDoc
        input_fields = {
            "document_context": dspy.InputField(desc="The full document content"),
        }

    # Define output field
    output_field_name = "result"
    output_fields = {
        output_field_name: dspy.OutputField(desc="The processed result text. Output ONLY the result.")
    }

    # Create the Signature class dynamically
    # Clean name for class naming (remove spaces, special chars)
    class_name = "".join(c for c in name if c.isalnum()) + "Signature"

    signature_class = type(
        class_name,
        (dspy.Signature,),
        {
            "__doc__": docstring,
            "__annotations__": {
                **{k: str for k in input_fields},
                **{k: str for k in output_fields},
            },
            **input_fields,
            **output_fields,
        }
    )

    return signature_class


def create_custom_module(
    name: str,
    instructions: str,
    input_type: str = "selection",
    output_type: str = "replace",
) -> type:
    """Create a DSPy Module class from user configuration.

    Args:
        name: Command name
        instructions: User's instructions for the AI
        input_type: "selection" | "cursor" | "fullDoc"
        output_type: "replace" | "insert"

    Returns:
        A DSPy Module class (not instance)
    """
    signature = create_custom_signature(name, instructions, input_type, output_type)

    # Clean name for class naming
    class_name = "".join(c for c in name if c.isalnum()) + "Predict"

    class CustomModule(dspy.Module):
        """Dynamically generated custom command module."""

        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict(signature)
            self._input_type = input_type
            self._output_type = output_type

        def forward(self, **kwargs) -> Any:
            return self.predictor(**kwargs)

    # Set the class name
    CustomModule.__name__ = class_name
    CustomModule.__qualname__ = class_name

    return CustomModule


class CustomProgramRegistry:
    """Registry for user-defined custom programs.

    Manages creation and caching of custom DSPy modules.
    """

    def __init__(self):
        self._modules: dict[str, type] = {}
        self._configs: dict[str, dict] = {}

    def register(self, program_id: str, config: dict) -> type:
        """Register a custom program from configuration.

        Args:
            program_id: Unique identifier for this program
            config: Dict with name, instructions, inputType, outputType

        Returns:
            The generated Module class
        """
        module_class = create_custom_module(
            name=config.get("name", program_id),
            instructions=config.get("instructions", "Process this text."),
            input_type=config.get("inputType", "selection"),
            output_type=config.get("outputType", "replace"),
        )

        self._modules[program_id] = module_class
        self._configs[program_id] = config

        return module_class

    def get(self, program_id: str) -> type | None:
        """Get a registered module class by ID."""
        return self._modules.get(program_id)

    def get_config(self, program_id: str) -> dict | None:
        """Get the configuration for a registered program."""
        return self._configs.get(program_id)

    def unregister(self, program_id: str) -> bool:
        """Remove a program from the registry."""
        if program_id in self._modules:
            del self._modules[program_id]
            del self._configs[program_id]
            return True
        return False

    def clear(self):
        """Clear all registered programs."""
        self._modules.clear()
        self._configs.clear()

    def list_programs(self) -> list[str]:
        """List all registered program IDs."""
        return list(self._modules.keys())

    def is_registered(self, program_id: str) -> bool:
        """Check if a program is registered."""
        return program_id in self._modules


# Global registry instance
custom_registry = CustomProgramRegistry()


def register_custom_programs(commands: list[dict]) -> None:
    """Register multiple custom programs from a list of command configs.

    Args:
        commands: List of command configurations, each with:
            - id or program: Unique identifier
            - name: Display name
            - instructions: AI instructions
            - inputType: selection | cursor | fullDoc
            - outputType: replace | insert
    """
    for cmd in commands:
        program_id = cmd.get("program") or cmd.get("id")
        if program_id:
            custom_registry.register(program_id, cmd)


def get_custom_program(program_id: str) -> type | None:
    """Get a custom program module class by ID."""
    return custom_registry.get(program_id)
