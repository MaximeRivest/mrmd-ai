"""
MRMD AI Server - Custom server with Juice Level support.

Replaces dspy-cli with a FastAPI server that supports juice levels
for progressive quality/cost tradeoff.
"""

import os
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from contextlib import asynccontextmanager

import dspy
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import json

# Thread pool for running blocking DSPy calls
_executor = ThreadPoolExecutor(max_workers=10)

from .juice import JuiceLevel, ReasoningLevel, JuicedProgram, get_lm, JUICE_MODELS, REASONING_DESCRIPTIONS
from .custom_programs import custom_registry, register_custom_programs
from .modules import (
    # Finish
    FinishSentencePredict,
    FinishParagraphPredict,
    FinishCodeLinePredict,
    FinishCodeSectionPredict,
    # Fix
    FixGrammarPredict,
    FixTranscriptionPredict,
    # Correct
    CorrectAndFinishLinePredict,
    CorrectAndFinishSectionPredict,
    # Code
    DocumentCodePredict,
    CompleteCodePredict,
    AddTypeHintsPredict,
    ImproveNamesPredict,
    ExplainCodePredict,
    RefactorCodePredict,
    FormatCodePredict,
    ProgramCodePredict,
    # Text
    GetSynonymsPredict,
    GetPhraseSynonymsPredict,
    ReformatMarkdownPredict,
    IdentifyReplacementPredict,
    # Document
    DocumentResponsePredict,
    DocumentSummaryPredict,
    DocumentAnalysisPredict,
    # Notebook
    NotebookNamePredict,
    # Edit (Ctrl-K and comments)
    EditAtCursorPredict,
    AddressCommentPredict,
    AddressAllCommentsPredict,
    AddressNearbyCommentPredict,
)


# Program registry - maps program names to their classes
PROGRAMS = {
    # Finish
    "FinishSentencePredict": FinishSentencePredict,
    "FinishParagraphPredict": FinishParagraphPredict,
    "FinishCodeLinePredict": FinishCodeLinePredict,
    "FinishCodeSectionPredict": FinishCodeSectionPredict,
    # Fix
    "FixGrammarPredict": FixGrammarPredict,
    "FixTranscriptionPredict": FixTranscriptionPredict,
    # Correct
    "CorrectAndFinishLinePredict": CorrectAndFinishLinePredict,
    "CorrectAndFinishSectionPredict": CorrectAndFinishSectionPredict,
    # Code
    "DocumentCodePredict": DocumentCodePredict,
    "CompleteCodePredict": CompleteCodePredict,
    "AddTypeHintsPredict": AddTypeHintsPredict,
    "ImproveNamesPredict": ImproveNamesPredict,
    "ExplainCodePredict": ExplainCodePredict,
    "RefactorCodePredict": RefactorCodePredict,
    "FormatCodePredict": FormatCodePredict,
    "ProgramCodePredict": ProgramCodePredict,
    # Text
    "GetSynonymsPredict": GetSynonymsPredict,
    "GetPhraseSynonymsPredict": GetPhraseSynonymsPredict,
    "ReformatMarkdownPredict": ReformatMarkdownPredict,
    "IdentifyReplacementPredict": IdentifyReplacementPredict,
    # Document
    "DocumentResponsePredict": DocumentResponsePredict,
    "DocumentSummaryPredict": DocumentSummaryPredict,
    "DocumentAnalysisPredict": DocumentAnalysisPredict,
    # Notebook
    "NotebookNamePredict": NotebookNamePredict,
    # Edit (Ctrl-K and comments)
    "EditAtCursorPredict": EditAtCursorPredict,
    "AddressCommentPredict": AddressCommentPredict,
    "AddressAllCommentsPredict": AddressAllCommentsPredict,
    "AddressNearbyCommentPredict": AddressNearbyCommentPredict,
}

# Cached program instances per juice level and reasoning level
# NOTE: Cache is only used when no custom API keys are provided
_program_cache: dict[tuple[str, int, int | None], JuicedProgram] = {}


# API key header names
API_KEY_HEADERS = {
    "anthropic": "X-Api-Key-Anthropic",
    "openai": "X-Api-Key-Openai",
    "groq": "X-Api-Key-Groq",
    "gemini": "X-Api-Key-Gemini",
    "openrouter": "X-Api-Key-Openrouter",
}


def extract_api_keys(request: Request) -> dict | None:
    """Extract API keys from request headers.

    Headers:
        X-Api-Key-Anthropic: Anthropic API key
        X-Api-Key-Openai: OpenAI API key
        X-Api-Key-Groq: Groq API key
        X-Api-Key-Gemini: Google Gemini API key
        X-Api-Key-Openrouter: OpenRouter API key

    Returns:
        Dict of provider -> key if any keys are provided, None otherwise.
    """
    api_keys = {}
    for provider, header in API_KEY_HEADERS.items():
        key = request.headers.get(header)
        if key:
            api_keys[provider] = key

    return api_keys if api_keys else None


def get_program(
    name: str,
    juice: int = 0,
    reasoning: int | None = None,
    api_keys: dict | None = None,
    model_override: str | None = None,
) -> JuicedProgram:
    """Get a JuicedProgram instance for the given program configuration.

    Args:
        name: Program name (can be built-in or custom)
        juice: Juice level (0-4)
        reasoning: Optional reasoning level (0-5)
        api_keys: Optional dict of provider -> API key
        model_override: Optional model to use instead of default

    Returns:
        Configured JuicedProgram instance.

    Note:
        Programs with custom API keys or model overrides are NOT cached,
        since they need fresh instances with the provided configuration.
        Custom programs are never cached.
    """
    # Check built-in programs first
    program_class = PROGRAMS.get(name)

    # If not found, check custom registry
    if program_class is None:
        program_class = custom_registry.get(name)

    if program_class is None:
        raise ValueError(f"Unknown program: {name}")

    # Custom programs and those with custom config are never cached
    is_custom = name not in PROGRAMS
    if api_keys or model_override or is_custom:
        program = program_class()
        return JuicedProgram(
            program,
            juice=juice,
            reasoning=reasoning,
            api_keys=api_keys,
            model_override=model_override,
        )

    # Use cache for standard built-in requests (no custom keys)
    cache_key = (name, juice, reasoning)
    if cache_key not in _program_cache:
        program = program_class()
        _program_cache[cache_key] = JuicedProgram(program, juice=juice, reasoning=reasoning)
    return _program_cache[cache_key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - configure default LM on startup."""
    # Configure default LM (juice level 0)
    default_lm = get_lm(JuiceLevel.QUICK)
    dspy.configure(lm=default_lm)
    print(f"[AI Server] Configured default LM: {JUICE_MODELS[JuiceLevel.QUICK].model}")
    yield
    print("[AI Server] Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="MRMD AI Server",
    description="AI server with Juice Level support for progressive quality/cost tradeoff",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/programs")
async def list_programs():
    """List available programs."""
    programs = []
    for name, cls in PROGRAMS.items():
        programs.append({
            "name": name,
            "endpoint": f"/{name}",
        })
    return {"programs": programs}


@app.get("/juice")
async def get_juice_levels():
    """Get available juice levels with their capabilities."""
    from .juice import JUICE_DESCRIPTIONS, JUICE_MODELS, JuiceLevel
    levels = []
    for level, desc in JUICE_DESCRIPTIONS.items():
        level_info = {
            "level": level.value,
            "description": desc,
        }
        # Add supports_reasoning for non-ULTIMATE levels
        if level != JuiceLevel.ULTIMATE and level in JUICE_MODELS:
            level_info["supports_reasoning"] = JUICE_MODELS[level].supports_reasoning
        else:
            # ULTIMATE level supports reasoning (all its sub-models do)
            level_info["supports_reasoning"] = True
        levels.append(level_info)
    return {"levels": levels}


@app.get("/reasoning")
async def get_reasoning_levels():
    """Get available reasoning levels."""
    return {
        "levels": [
            {"level": level.value, "description": desc}
            for level, desc in REASONING_DESCRIPTIONS.items()
        ]
    }


def extract_result(prediction: Any) -> dict:
    """Extract result from a DSPy prediction object."""
    result = {}

    if hasattr(prediction, "synthesized_response"):
        # Ultimate level returns synthesized response
        result["synthesized_response"] = prediction.synthesized_response

    # DSPy Prediction objects store outputs in _store
    if hasattr(prediction, "_store") and prediction._store:
        result.update(dict(prediction._store))
    else:
        # Fallback: try direct attribute access
        output_fields = [
            "synonyms", "original", "alternatives",
            "completion", "fixed_text", "corrected_completion",
            "documented_code", "typed_code", "improved_code",
            "explained_code", "refactored_code", "formatted_code",
            "reformatted_text", "text_to_replace", "replacement",
            "response", "summary", "analysis",  # Document-level fields
            "code",  # ProgramCodePredict output
            "edits",  # EditAtCursor and AddressComment outputs
        ]

        for field in output_fields:
            if hasattr(prediction, field):
                val = getattr(prediction, field)
                if val is not None:
                    result[field] = val

    # Include individual model responses for Ultimate level (juice=4)
    if hasattr(prediction, "_individual_responses"):
        result["_individual_responses"] = prediction._individual_responses

    return result


@app.post("/{program_name}")
async def run_program(program_name: str, request: Request):
    """Run a program with the given parameters."""
    # Get juice level from header
    juice_header = request.headers.get("X-Juice-Level", "0")
    try:
        juice_level = int(juice_header)
        juice_level = max(0, min(4, juice_level))  # Clamp to 0-4
    except ValueError:
        juice_level = 0

    # Get reasoning level from header (optional)
    reasoning_header = request.headers.get("X-Reasoning-Level")
    reasoning_level = None
    if reasoning_header is not None:
        try:
            reasoning_level = int(reasoning_header)
            reasoning_level = max(0, min(5, reasoning_level))  # Clamp to 0-5
        except ValueError:
            reasoning_level = None

    # Extract API keys from headers (optional)
    api_keys = extract_api_keys(request)

    # Get model override from header (optional)
    model_override = request.headers.get("X-Model-Override")

    # Get request body
    try:
        params = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Get program
    try:
        juiced_program = get_program(
            program_name,
            juice_level,
            reasoning_level,
            api_keys=api_keys,
            model_override=model_override,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Log the call and get model info
    from .juice import JUICE_DESCRIPTIONS, JUICE_MODELS, ULTIMATE_MODELS, JuiceLevel, ReasoningLevel
    juice_desc = JUICE_DESCRIPTIONS.get(JuiceLevel(juice_level), f"Level {juice_level}")
    reasoning_desc = ""
    if reasoning_level is not None:
        reasoning_desc = f" | {REASONING_DESCRIPTIONS.get(ReasoningLevel(reasoning_level), f'Reasoning {reasoning_level}')}"
    print(f"[AI] {program_name} @ {juice_desc}{reasoning_desc}", flush=True)

    # Get the model name for this juice level
    if juice_level == JuiceLevel.ULTIMATE:
        model_name = "multi-model"  # Ultimate uses multiple models
    else:
        model_config = JUICE_MODELS.get(JuiceLevel(juice_level))
        model_name = model_config.model if model_config else "unknown"

    # Run program in thread pool (DSPy calls are blocking)
    def run_sync():
        return juiced_program(**params)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, run_sync)
        response = extract_result(result)
        # Add model metadata to response
        response["_model"] = model_name
        response["_juice_level"] = juice_level
        response["_reasoning_level"] = reasoning_level
        # Serialize any Pydantic models to dicts for JSON compatibility
        return serialize_for_json(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def serialize_for_json(obj):
    """Recursively convert Pydantic models and other objects to JSON-serializable form."""
    if hasattr(obj, 'model_dump'):
        # Pydantic v2 model
        return obj.model_dump()
    elif hasattr(obj, 'dict'):
        # Pydantic v1 model
        return obj.dict()
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    else:
        return obj


def sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    serialized = serialize_for_json(data)
    return f"event: {event}\ndata: {json.dumps(serialized)}\n\n"


@app.post("/{program_name}/stream")
async def run_program_stream(program_name: str, request: Request):
    """Run a program with SSE streaming for status updates.

    Emits events:
    - status: Progress updates (step, model, etc.)
    - model_complete: When a model finishes (for ultimate mode)
    - result: Final result
    - error: If an error occurs
    """
    # Get juice level from header
    juice_header = request.headers.get("X-Juice-Level", "0")
    try:
        juice_level = int(juice_header)
        juice_level = max(0, min(4, juice_level))  # Clamp to 0-4
    except ValueError:
        juice_level = 0

    # Get reasoning level from header (optional)
    reasoning_header = request.headers.get("X-Reasoning-Level")
    reasoning_level = None
    if reasoning_header is not None:
        try:
            reasoning_level = int(reasoning_header)
            reasoning_level = max(0, min(5, reasoning_level))  # Clamp to 0-5
        except ValueError:
            reasoning_level = None

    # Extract API keys from headers (optional)
    api_keys = extract_api_keys(request)

    # Get model override from header (optional)
    model_override = request.headers.get("X-Model-Override")

    # Get request body
    try:
        params = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Check program exists
    if program_name not in PROGRAMS:
        raise HTTPException(status_code=404, detail=f"Unknown program: {program_name}")

    # Get model info
    from .juice import JUICE_DESCRIPTIONS, JUICE_MODELS, ULTIMATE_MODELS, JuiceLevel, ReasoningLevel, JuicedProgram
    juice_desc = JUICE_DESCRIPTIONS.get(JuiceLevel(juice_level), f"Level {juice_level}")
    reasoning_desc = ""
    if reasoning_level is not None:
        reasoning_desc = f" | {REASONING_DESCRIPTIONS.get(ReasoningLevel(reasoning_level), f'Reasoning {reasoning_level}')}"
    print(f"[AI Stream] {program_name} @ {juice_desc}{reasoning_desc}", flush=True)

    # Get model name(s) for display
    if juice_level == JuiceLevel.ULTIMATE:
        model_name = "multi-model"
        model_names = [cfg.model.split("/")[-1] for cfg in ULTIMATE_MODELS]
    else:
        model_config = JUICE_MODELS.get(JuiceLevel(juice_level))
        model_name = model_config.model if model_config else "unknown"
        model_names = [model_name.split("/")[-1] if "/" in model_name else model_name]

    async def event_generator():
        """Generate SSE events for the AI call."""
        import queue
        import threading

        # Queue for progress events from the sync execution
        progress_queue = queue.Queue()
        result_holder = {"result": None, "error": None}

        def progress_callback(event_type: str, data: dict):
            """Called by JuicedProgram to report progress."""
            progress_queue.put((event_type, data))

        def run_with_progress():
            """Run the program in a thread, emitting progress events."""
            try:
                # Create program with progress callback and optional API keys
                program_class = PROGRAMS[program_name]
                program = program_class()
                juiced = JuicedProgram(
                    program,
                    juice=juice_level,
                    reasoning=reasoning_level,
                    progress_callback=progress_callback,
                    api_keys=api_keys,
                    model_override=model_override,
                )

                # Emit starting event
                progress_callback("status", {
                    "step": "starting",
                    "model": model_name,
                    "juice_level": juice_level,
                    "juice_name": juice_desc,
                    "reasoning_level": reasoning_level,
                    "reasoning_name": reasoning_desc.strip(" |") if reasoning_desc else None,
                })

                # Run the program
                result = juiced(**params)
                result_holder["result"] = result

                # Signal completion
                progress_queue.put(("done", None))

            except Exception as e:
                import traceback
                traceback.print_exc()
                result_holder["error"] = str(e)
                progress_queue.put(("error", {"message": str(e)}))

        # Start execution in thread
        thread = threading.Thread(target=run_with_progress)
        thread.start()

        # Stream progress events
        while True:
            try:
                # Wait for events with timeout to allow checking if thread is done
                event_type, data = progress_queue.get(timeout=0.1)

                if event_type == "done":
                    # Send final result
                    if result_holder["result"] is not None:
                        response = extract_result(result_holder["result"])
                        response["_model"] = model_name
                        response["_juice_level"] = juice_level
                        response["_reasoning_level"] = reasoning_level
                        yield sse_event("result", response)
                    break

                elif event_type == "error":
                    yield sse_event("error", data)
                    break

                else:
                    # Status or model_complete event
                    yield sse_event(event_type, data)

            except queue.Empty:
                # Check if thread is still running
                if not thread.is_alive():
                    # Thread finished but didn't put done event - check for error
                    if result_holder["error"]:
                        yield sse_event("error", {"message": result_holder["error"]})
                    break
                continue

        thread.join(timeout=1.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# =============================================================================
# CUSTOM PROGRAMS API
# =============================================================================

class CustomProgramConfig(BaseModel):
    """Configuration for a custom program."""
    id: str  # Unique program ID (e.g., "Custom_cmd-123")
    name: str
    instructions: str
    inputType: str = "selection"  # selection | cursor | fullDoc
    outputType: str = "replace"  # replace | insert


class RegisterCustomProgramsRequest(BaseModel):
    """Request to register multiple custom programs."""
    programs: list[CustomProgramConfig]


@app.post("/api/custom-programs/register")
async def register_custom_programs_endpoint(request: RegisterCustomProgramsRequest):
    """Register custom programs from frontend configuration.

    This endpoint is called when the app starts or when settings change.
    It clears existing custom programs and registers the new ones.
    """
    # Clear existing custom programs
    custom_registry.clear()

    # Register new programs
    registered = []
    for prog in request.programs:
        try:
            custom_registry.register(
                program_id=prog.id,
                config={
                    "name": prog.name,
                    "instructions": prog.instructions,
                    "inputType": prog.inputType,
                    "outputType": prog.outputType,
                }
            )
            registered.append(prog.id)
            print(f"[Custom] Registered: {prog.name} ({prog.id})")
        except Exception as e:
            print(f"[Custom] Failed to register {prog.id}: {e}")

    return {"registered": registered, "count": len(registered)}


@app.get("/api/custom-programs")
async def list_custom_programs():
    """List all registered custom programs."""
    programs = []
    for program_id in custom_registry.list_programs():
        config = custom_registry.get_config(program_id)
        if config:
            programs.append({
                "id": program_id,
                "name": config.get("name", program_id),
                "inputType": config.get("inputType", "selection"),
                "outputType": config.get("outputType", "replace"),
            })
    return {"programs": programs}


@app.delete("/api/custom-programs/{program_id}")
async def unregister_custom_program(program_id: str):
    """Unregister a specific custom program."""
    if custom_registry.unregister(program_id):
        return {"success": True, "id": program_id}
    raise HTTPException(status_code=404, detail=f"Program not found: {program_id}")


def main():
    """Run the AI server."""
    import argparse
    parser = argparse.ArgumentParser(description="MRMD AI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=51790, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════╗
║          MRMD AI Server                  ║
║     with Juice Level Support             ║
╚══════════════════════════════════════════╝

  URL:  http://{args.host}:{args.port}/

  Juice Levels:
    0 = ⚡ Quick (Grok 4.1)
    1 = ⚖️ Balanced (Sonnet 4.5)
    2 = 🧠 Deep (Gemini 3 thinking)
    3 = 🚀 Maximum (Opus 4.5 thinking)
    4 = 🔥 Ultimate (Multi-model merger)

  Press Ctrl+C to stop
""")

    uvicorn.run(
        "mrmd_ai.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
