"""
Juice Level System for MRMD AI Programs.

Juice levels control the quality/cost tradeoff of AI responses:
- Level 0: Kimi K2 on Groq (fast, cheap, default)
- Level 1: Claude Sonnet 4.5 (better quality)
- Level 2: Gemini 3 Pro with thinking (deep reasoning)
- Level 3: Claude Opus 4.5 with high thinking (maximum single-model quality)
- Level 4: Multi-model merger (Grok 4 + Sonnet 4.5 + Gemini 3 + Opus 4.5, synthesized by Gemini 3)
"""

from enum import IntEnum
from typing import Any, Callable
from dataclasses import dataclass, field
import dspy


class JuiceLevel(IntEnum):
    """Progressive quality levels for AI responses."""

    # Fast & cheap - Kimi K2 on Groq
    QUICK = 0

    # Better quality - Sonnet 4.5
    BALANCED = 1

    # Deep reasoning - Gemini 3 with thinking
    DEEP = 2

    # Maximum single-model - Opus 4.5 with high thinking
    MAXIMUM = 3

    # Multi-model merger - all models synthesized
    ULTIMATE = 4


class ReasoningLevel(IntEnum):
    """Independent reasoning/thinking budget control.

    This is separate from JuiceLevel and controls how much "thinking"
    the model does, independent of which model is selected.
    """

    # No extended thinking - fastest responses
    OFF = 0

    # Minimal reasoning
    MINIMAL = 1

    # Low reasoning effort
    LOW = 2

    # Medium reasoning effort
    MEDIUM = 3

    # High reasoning effort
    HIGH = 4

    # Maximum reasoning budget
    MAXIMUM = 5


# Map reasoning levels to thinking budgets and reasoning_effort values
# For Anthropic: uses `thinking={"type": "enabled", "budget_tokens": X}`
# For others: uses `reasoning_effort` ("low", "medium", "high")
# Note: Anthropic requires max_tokens > thinking.budget_tokens
REASONING_CONFIGS: dict[ReasoningLevel, dict] = {
    ReasoningLevel.OFF: {
        "budget_tokens": None,  # No thinking
        "reasoning_effort": None,
        "temperature": None,  # None means use model default
    },
    ReasoningLevel.MINIMAL: {
        "budget_tokens": 1024,  # Minimum thinking budget
        "reasoning_effort": "low",
        "temperature": 1.0,  # Required for Anthropic extended thinking
        "max_tokens": 4096,  # Must be > budget_tokens
    },
    ReasoningLevel.LOW: {
        "budget_tokens": 4096,
        "reasoning_effort": "low",
        "temperature": 1.0,
        "max_tokens": 8192,
    },
    ReasoningLevel.MEDIUM: {
        "budget_tokens": 8192,
        "reasoning_effort": "medium",
        "temperature": 1.0,
        "max_tokens": 16000,
    },
    ReasoningLevel.HIGH: {
        "budget_tokens": 16384,
        "reasoning_effort": "high",
        "temperature": 1.0,
        "max_tokens": 24000,
    },
    ReasoningLevel.MAXIMUM: {
        "budget_tokens": 32768,  # Maximum thinking budget
        "reasoning_effort": "high",
        "temperature": 1.0,
        "max_tokens": 48000,  # Must be > budget_tokens
    },
}


REASONING_DESCRIPTIONS = {
    ReasoningLevel.OFF: "Off - No extended thinking",
    ReasoningLevel.MINIMAL: "Minimal - Light reasoning",
    ReasoningLevel.LOW: "Low - Some reasoning",
    ReasoningLevel.MEDIUM: "Medium - Moderate reasoning",
    ReasoningLevel.HIGH: "High - Deep reasoning",
    ReasoningLevel.MAXIMUM: "Maximum - Full reasoning budget",
}


@dataclass
class ModelConfig:
    """Configuration for a model at a specific juice level."""
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    reasoning_effort: str | None = None
    thinking: dict | None = None
    supports_reasoning: bool = True  # Whether the model supports reasoning_effort
    extra_kwargs: dict = field(default_factory=dict)

    def to_lm_kwargs(self) -> dict:
        """Convert to dspy.LM kwargs."""
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_kwargs,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.thinking:
            kwargs["thinking"] = self.thinking
        return kwargs


# Model configurations for each juice level
# supports_reasoning indicates if the model/provider supports reasoning_effort parameter
JUICE_MODELS: dict[JuiceLevel, ModelConfig] = {
    JuiceLevel.QUICK: ModelConfig(
        model="groq/moonshotai/kimi-k2-instruct-0905",
        temperature=0.7,
        max_tokens=4096,
        supports_reasoning=False,  # Groq does NOT support reasoning_effort
    ),
    JuiceLevel.BALANCED: ModelConfig(
        model="anthropic/claude-sonnet-4-5",
        temperature=0.7,
        max_tokens=4096,
        supports_reasoning=True,  # Anthropic supports reasoning_effort
    ),
    JuiceLevel.DEEP: ModelConfig(
        model="gemini/gemini-3-pro-preview",
        temperature=1.0,
        max_tokens=16000,
        reasoning_effort="high",
        supports_reasoning=True,  # Gemini supports reasoning_effort
    ),
    JuiceLevel.MAXIMUM: ModelConfig(
        model="anthropic/claude-opus-4-5",
        temperature=1.0,
        max_tokens=16000,
        reasoning_effort="high",
        supports_reasoning=True,  # Anthropic supports reasoning_effort
    ),
}

# For ULTIMATE level, we use all 4 models with highest thinking
# Grok 4, GPT-5.1, Gemini 3, Opus 4.5
# NOTE: Anthropic requires temperature=1 when using extended thinking
ULTIMATE_MODELS: list[ModelConfig] = [
    ModelConfig(
        model="openrouter/x-ai/grok-4",
        temperature=0.7,
        max_tokens=8192,
        supports_reasoning=True,  # Grok 4 supports reasoning
    ),
    ModelConfig(
        model="openai/gpt-5.2",
        temperature=1.0,
        max_tokens=16000,
        reasoning_effort="high",
        supports_reasoning=True,  # OpenAI supports reasoning
    ),
    ModelConfig(
        model="gemini/gemini-3-pro-preview",
        temperature=1.0,
        max_tokens=16000,
        reasoning_effort="high",
        supports_reasoning=True,  # Gemini supports reasoning
    ),
    ModelConfig(
        model="anthropic/claude-opus-4-5",
        temperature=1.0,  # Must be 1 for extended thinking
        max_tokens=16000,
        reasoning_effort="high",
        supports_reasoning=True,  # Anthropic supports reasoning
    ),
]

# Synthesizer model for ULTIMATE level (Gemini 3 synthesizes all responses)
SYNTHESIZER_MODEL = ModelConfig(
    model="gemini/gemini-3-pro-preview",
    temperature=0.7,
    max_tokens=32000,
    reasoning_effort="high",
    supports_reasoning=True,
)


def get_api_key_for_model(model: str, api_keys: dict | None) -> str | None:
    """Get the appropriate API key for a model based on its provider.

    Uses LiteLLM model naming convention: provider/model-name
    Supports any provider that the user has configured in settings.

    Args:
        model: Model identifier (e.g., "anthropic/claude-sonnet-4-5")
        api_keys: Dict of provider -> API key

    Returns:
        API key string or None if not found/provided.
    """
    if not api_keys:
        return None

    model_lower = model.lower()

    # Extract provider from model name (LiteLLM format: provider/model-name)
    if "/" in model:
        provider = model.split("/")[0].lower()
        # Check for direct provider match
        if provider in api_keys and api_keys[provider]:
            return api_keys[provider]

    # Fallback: Check for known provider patterns in model name
    # This handles cases like "claude-3-sonnet" without prefix
    provider_patterns = {
        "anthropic": ["anthropic/", "claude"],
        "openai": ["openai/", "gpt-", "o1-", "o3-"],
        "groq": ["groq/"],
        "gemini": ["gemini/", "gemini-"],
        "openrouter": ["openrouter/"],
        "together_ai": ["together_ai/", "together/"],
        "fireworks_ai": ["fireworks_ai/", "fireworks/"],
        "mistral": ["mistral/"],
        "cohere": ["cohere/"],
        "deepseek": ["deepseek/"],
        "ollama": ["ollama/"],
        "azure": ["azure/"],
        "bedrock": ["bedrock/"],
        "vertex_ai": ["vertex_ai/"],
    }

    for provider, patterns in provider_patterns.items():
        for pattern in patterns:
            if pattern in model_lower:
                if provider in api_keys and api_keys[provider]:
                    return api_keys[provider]
                break

    return None


def get_lm(
    juice: JuiceLevel | int = JuiceLevel.QUICK,
    reasoning: ReasoningLevel | int | None = None,
    api_keys: dict | None = None,
    model_override: str | None = None,
) -> dspy.LM:
    """Get a dspy.LM configured for the specified juice and reasoning levels.

    Args:
        juice: Juice level (0-3). Level 4 (ULTIMATE) requires special handling.
        reasoning: Optional reasoning level (0-5). If None, uses juice level's default.
        api_keys: Optional dict of provider -> API key. If provided, overrides env vars.
        model_override: Optional model to use instead of the default for this juice level.

    Returns:
        Configured dspy.LM instance.
    """
    if isinstance(juice, int):
        juice = JuiceLevel(juice)

    if juice == JuiceLevel.ULTIMATE:
        raise ValueError("ULTIMATE juice level requires multi-model merger. Use JuicedProgram instead.")

    config = JUICE_MODELS[juice]
    kwargs = config.to_lm_kwargs()

    # Apply model override if provided
    if model_override:
        kwargs["model"] = model_override

    # Get API key for this model's provider
    api_key = get_api_key_for_model(kwargs["model"], api_keys)
    if api_key:
        kwargs["api_key"] = api_key

    # Apply reasoning level overrides if specified AND model supports reasoning
    if reasoning is not None and config.supports_reasoning:
        if isinstance(reasoning, int):
            reasoning = ReasoningLevel(reasoning)

        # Skip if reasoning is OFF
        if reasoning == ReasoningLevel.OFF:
            # Remove any existing reasoning params
            kwargs.pop("reasoning_effort", None)
            kwargs.pop("thinking", None)
            return dspy.LM(**kwargs)

        reasoning_config = REASONING_CONFIGS[reasoning]
        model = config.model.lower()

        # Determine provider and use appropriate parameter format
        is_anthropic = "anthropic/" in model or "claude" in model
        is_gemini = "gemini" in model
        is_openai = "openai/" in model or "gpt" in model

        # Apply temperature (required for Anthropic extended thinking)
        if reasoning_config.get("temperature") is not None:
            kwargs["temperature"] = reasoning_config["temperature"]

        # Apply max_tokens
        if reasoning_config.get("max_tokens") is not None:
            kwargs["max_tokens"] = reasoning_config["max_tokens"]

        if is_anthropic:
            # Anthropic uses explicit thinking parameter with budget_tokens
            budget = reasoning_config.get("budget_tokens", 1024)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            # Remove reasoning_effort if present (not used for thinking)
            kwargs.pop("reasoning_effort", None)
        else:
            # Other providers use reasoning_effort
            if reasoning_config["reasoning_effort"] is not None:
                kwargs["reasoning_effort"] = reasoning_config["reasoning_effort"]

    return dspy.LM(**kwargs)


class SynthesizeResponses(dspy.Signature):
    """Synthesize multiple AI model responses into one optimal final answer.

    You are given responses from multiple AI models for the same task.
    Your job is to create the BEST possible response by:
    1. Identifying the strongest elements from each model's response
    2. Resolving any contradictions (prefer the most accurate/well-reasoned answer)
    3. Combining complementary insights that don't conflict
    4. Maintaining the original format and style expected for the task
    5. Being concise - don't add unnecessary elaboration

    For grammar/spelling fixes: Pick the most correct version, don't over-correct.
    For text completion: Choose the most natural, coherent continuation.
    For code: Select the cleanest, most idiomatic solution.
    For lists: You may combine unique items if appropriate.

    Output ONLY the synthesized response - no explanations or meta-commentary.
    """

    original_input: str = dspy.InputField(desc="The original input/task that was given to all models")
    model_responses: str = dspy.InputField(desc="Responses from multiple AI models, each labeled with model name")
    synthesized_response: str = dspy.OutputField(desc="The single best response, synthesized from all model outputs. Output ONLY the response content.")


class JuicedProgram:
    """Wrapper that runs any DSPy program with configurable juice levels.

    For levels 0-3, uses a single model with increasing capability.
    For level 4 (ULTIMATE), runs all models in parallel and synthesizes.
    """

    def __init__(
        self,
        program: dspy.Module,
        juice: JuiceLevel | int = JuiceLevel.QUICK,
        reasoning: ReasoningLevel | int | None = None,
        progress_callback: Callable[[str, dict], None] | None = None,
        api_keys: dict | None = None,
        model_override: str | None = None,
    ):
        """Initialize a juiced program.

        Args:
            program: The DSPy program/module to wrap.
            juice: Juice level (0-4).
            reasoning: Optional reasoning level (0-5). If None, uses juice level's default.
            progress_callback: Optional callback for progress events.
                              Called with (event_type, data) where event_type is:
                              - "status": General status update
                              - "model_start": A model is starting (ultimate mode)
                              - "model_complete": A model finished (ultimate mode)
            api_keys: Optional dict of provider -> API key. Overrides env vars.
            model_override: Optional model to use instead of the default for this juice level.
        """
        self.program = program
        self.juice = JuiceLevel(juice) if isinstance(juice, int) else juice
        self.reasoning = ReasoningLevel(reasoning) if isinstance(reasoning, int) else reasoning
        self.progress_callback = progress_callback
        self.api_keys = api_keys
        self.model_override = model_override

    def _emit(self, event_type: str, data: dict):
        """Emit a progress event if callback is set."""
        if self.progress_callback:
            self.progress_callback(event_type, data)

    def __call__(self, **kwargs) -> Any:
        """Run the program with the configured juice level."""
        if self.juice == JuiceLevel.ULTIMATE:
            return self._run_ultimate(**kwargs)
        else:
            return self._run_single(**kwargs)

    def _run_single(self, **kwargs) -> Any:
        """Run with a single model at the specified juice level."""
        config = JUICE_MODELS[self.juice]

        # Use model override if provided, otherwise use default for juice level
        actual_model = self.model_override if self.model_override else config.model
        model_name = actual_model.split("/")[-1]

        reasoning_desc = ""
        if self.reasoning is not None:
            reasoning_desc = f" (reasoning={self.reasoning.name})"

        self._emit("status", {
            "step": "calling_model",
            "model": model_name,
            "model_full": actual_model,
            "reasoning_level": self.reasoning.value if self.reasoning else None,
        })

        lm = get_lm(self.juice, self.reasoning, api_keys=self.api_keys, model_override=self.model_override)
        with dspy.context(lm=lm):
            result = self.program(**kwargs)

        self._emit("status", {
            "step": "model_complete",
            "model": model_name
        })

        return result

    def _run_ultimate(self, **kwargs) -> Any:
        """Run with all models in PARALLEL and merge results."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        # Track which models are running
        model_names = [cfg.model.split("/")[-1] for cfg in ULTIMATE_MODELS]
        models_status = {name: "pending" for name in model_names}
        status_lock = threading.Lock()

        self._emit("status", {
            "step": "starting_multi_model",
            "models": model_names,
            "total": len(model_names),
            "reasoning_level": self.reasoning.value if self.reasoning else None,
        })

        def run_model(config):
            """Run a single model - called in parallel."""
            lm_kwargs = config.to_lm_kwargs()

            # Apply reasoning level overrides if specified AND model supports reasoning
            if self.reasoning is not None and self.reasoning != ReasoningLevel.OFF and config.supports_reasoning:
                reasoning_config = REASONING_CONFIGS[self.reasoning]
                model = config.model.lower()

                # Determine provider
                is_anthropic = "anthropic/" in model or "claude" in model

                # Apply temperature and max_tokens
                if reasoning_config.get("temperature") is not None:
                    lm_kwargs["temperature"] = reasoning_config["temperature"]
                if reasoning_config.get("max_tokens") is not None:
                    lm_kwargs["max_tokens"] = reasoning_config["max_tokens"]

                if is_anthropic:
                    # Anthropic uses thinking parameter with budget_tokens
                    budget = reasoning_config.get("budget_tokens", 1024)
                    lm_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                    lm_kwargs.pop("reasoning_effort", None)
                else:
                    # Other providers use reasoning_effort
                    if reasoning_config["reasoning_effort"] is not None:
                        lm_kwargs["reasoning_effort"] = reasoning_config["reasoning_effort"]

            # Apply API key for this model's provider
            api_key = get_api_key_for_model(config.model, self.api_keys)
            if api_key:
                lm_kwargs["api_key"] = api_key

            lm = dspy.LM(**lm_kwargs)
            model_name = config.model.split("/")[-1]

            # Emit model start
            with status_lock:
                models_status[model_name] = "running"
            self._emit("model_start", {
                "model": model_name,
                "models_status": dict(models_status)
            })

            try:
                with dspy.context(lm=lm):
                    result = self.program(**kwargs)

                # Extract response text from DSPy Prediction for streaming
                response_data = {}
                if hasattr(result, "_store") and result._store:
                    response_data = dict(result._store)

                # Emit model complete WITH the actual response
                with status_lock:
                    models_status[model_name] = "complete"
                self._emit("model_complete", {
                    "model": model_name,
                    "success": True,
                    "models_status": dict(models_status),
                    "response": response_data,  # Include actual response!
                })

                return {"model": model_name, "result": result, "error": None}
            except Exception as e:
                # Emit model error
                with status_lock:
                    models_status[model_name] = "error"
                self._emit("model_complete", {
                    "model": model_name,
                    "success": False,
                    "error": str(e),
                    "models_status": dict(models_status),
                    "response": None,
                })
                return {"model": model_name, "result": None, "error": str(e)}

        # Run all 4 models in parallel
        model_results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_model, config) for config in ULTIMATE_MODELS]
            for future in as_completed(futures):
                model_results.append(future.result())

        # Emit synthesizing status
        self._emit("status", {
            "step": "synthesizing",
            "models_completed": len([r for r in model_results if r["result"] is not None])
        })

        # Merge results using AI synthesis
        return self._merge_results(model_results, kwargs)

    def _merge_results(self, model_results: list, original_input: dict) -> Any:
        """Merge results from multiple models using AI synthesis.

        Uses SYNTHESIZER_MODEL to intelligently combine responses from all models.
        """
        # Get successful results
        successful = [r for r in model_results if r["result"] is not None]
        if not successful:
            # All failed - return error
            errors = [r["error"] for r in model_results if r["error"]]
            raise RuntimeError(f"All models failed: {errors}")

        # If only one model succeeded, just return its result
        if len(successful) == 1:
            result = successful[0]["result"]
            if hasattr(result, "_store"):
                result._individual_responses = [{
                    "model": successful[0]["model"],
                    "response": str(result._store),
                    "error": None
                }]
            return result

        # Collect individual responses
        individual_responses = []
        model_outputs = {}  # model_name -> {field: value}

        for r in model_results:
            model_name = r["model"]
            if r["result"] is not None and hasattr(r["result"], "_store"):
                store = r["result"]._store
                model_outputs[model_name] = dict(store)
                # Get main output text for display
                output_text = None
                for key, value in store.items():
                    if isinstance(value, str) and len(value) > 10:
                        output_text = value
                        break
                individual_responses.append({
                    "model": model_name,
                    "response": output_text or str(store),
                    "error": None
                })
            elif r["error"]:
                individual_responses.append({
                    "model": model_name,
                    "response": None,
                    "error": r["error"]
                })

        # Use first result as template for output fields
        base_result = successful[0]["result"]
        base_store = base_result._store if hasattr(base_result, "_store") else {}

        # Format original input for synthesizer
        input_text = self._format_input(original_input)

        # Create synthesized result
        merged = {}

        # Configure synthesizer LM with API key if provided
        synth_kwargs = SYNTHESIZER_MODEL.to_lm_kwargs()
        api_key = get_api_key_for_model(SYNTHESIZER_MODEL.model, self.api_keys)
        if api_key:
            synth_kwargs["api_key"] = api_key
        synth_lm = dspy.LM(**synth_kwargs)

        # Synthesize each output field
        for field_name, base_value in base_store.items():
            # Collect this field's values from all models
            field_values = {}
            for model_name, outputs in model_outputs.items():
                if field_name in outputs:
                    field_values[model_name] = outputs[field_name]

            if not field_values:
                merged[field_name] = base_value
                continue

            # Check if it's a list field (like synonyms)
            if isinstance(base_value, list):
                # For lists, combine unique values from all models
                combined = []
                seen = set()
                for model_name, values in field_values.items():
                    if isinstance(values, list):
                        for item in values:
                            # Get hashable key for deduplication
                            # Pydantic models aren't hashable, so convert to JSON
                            try:
                                if hasattr(item, 'model_dump_json'):
                                    # Pydantic v2 model
                                    item_key = item.model_dump_json()
                                elif hasattr(item, 'json'):
                                    # Pydantic v1 model
                                    item_key = item.json()
                                else:
                                    # Regular hashable item
                                    item_key = item
                            except TypeError:
                                # Fallback: convert to string representation
                                item_key = str(item)

                            if item_key not in seen:
                                combined.append(item)
                                seen.add(item_key)
                merged[field_name] = combined
            else:
                # For string/text fields, use AI synthesis
                responses_text = "\n\n".join([
                    f"=== {model_name} ===\n{value}"
                    for model_name, value in field_values.items()
                ])

                self._emit("status", {
                    "step": "synthesizing_field",
                    "field": field_name,
                    "model": SYNTHESIZER_MODEL.model.split("/")[-1]
                })

                try:
                    with dspy.context(lm=synth_lm):
                        predictor = dspy.Predict(SynthesizeResponses)
                        synth_result = predictor(
                            original_input=input_text,
                            model_responses=responses_text
                        )
                        merged[field_name] = synth_result.synthesized_response
                except Exception as e:
                    # Fallback to first model's response on synthesis error
                    print(f"[Synthesis] Error synthesizing {field_name}: {e}")
                    merged[field_name] = base_value

        # Return a result object with merged data
        class MergedResult:
            pass

        result = MergedResult()
        for key, value in merged.items():
            setattr(result, key, value)
        result._store = merged  # For extract_result in server.py
        result._individual_responses = individual_responses  # For UI display
        result._synthesized = True  # Mark as AI-synthesized

        return result

    def _format_input(self, kwargs: dict) -> str:
        """Format input kwargs as a readable string."""
        parts = []
        for key, value in kwargs.items():
            parts.append(f"{key}: {value}")
        return "\n".join(parts)


def juiced(juice: JuiceLevel | int = JuiceLevel.QUICK):
    """Decorator to run a DSPy program with a specific juice level.

    Usage:
        @juiced(JuiceLevel.DEEP)
        def my_program():
            return dspy.ChainOfThought(MySignature)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            program = func(*args, **kwargs)
            return JuicedProgram(program, juice)
        return wrapper
    return decorator


def run_with_juice(
    program: dspy.Module,
    juice: JuiceLevel | int,
    reasoning: ReasoningLevel | int | None = None,
    **kwargs
) -> Any:
    """Convenience function to run a program with a specific juice level.

    Args:
        program: The DSPy program to run.
        juice: Juice level (0-4).
        reasoning: Optional reasoning level (0-5). If None, uses juice level's default.
        **kwargs: Arguments to pass to the program.

    Returns:
        The program result.
    """
    juiced_program = JuicedProgram(program, juice, reasoning=reasoning)
    return juiced_program(**kwargs)


# Juice level descriptions for CLI/UI
JUICE_DESCRIPTIONS = {
    JuiceLevel.QUICK: "Quick (Kimi K2) - Fast & cheap",
    JuiceLevel.BALANCED: "Balanced (Sonnet 4.5) - Good quality",
    JuiceLevel.DEEP: "Deep (Gemini 3 thinking) - Thorough reasoning",
    JuiceLevel.MAXIMUM: "Maximum (Opus 4.5 thinking) - Best single model",
    JuiceLevel.ULTIMATE: "Ultimate (Multi-model merger) - All models synthesized",
}
