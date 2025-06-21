# mypy: disable - error - code = "no-untyped-def,misc"
import pathlib
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
sys.path.append(os.path.dirname(__file__))
from configuration import Configuration

# Define the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev server ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_frontend_router(build_dir="../frontend/dist"):
    """Creates a router to serve the React frontend.

    Args:
        build_dir: Path to the React build directory relative to this file.

    Returns:
        A Starlette application serving the frontend.
    """
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir

    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        print(
            f"WARN: Frontend build directory not found or incomplete at {build_path}. Serving frontend will likely fail."
        )
        # Return a dummy router if build isn't ready
        from starlette.routing import Route

        async def dummy_frontend(_):
            return Response(
                "Frontend not built. Run 'npm run build' in the frontend directory.",
                media_type="text/plain",
                status_code=503,
            )

        return Route("/{path:path}", endpoint=dummy_frontend)

    return StaticFiles(directory=build_path, html=True)


@app.get("/api/models")
async def get_available_models():
    """Get available models based on the current LLM provider configuration."""
    import asyncio
    config = await asyncio.to_thread(Configuration.from_runnable_config)
    
    # Define available models for each provider
    model_definitions = {
        "google": [
            {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "provider": "google"},
            {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "provider": "google"},
            {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "google"},
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro", "provider": "google"},
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "provider": "google"},
        ],
        "openai": [
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "provider": "openai"},
            {"id": "gpt-4", "name": "GPT-4", "provider": "openai"},
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "provider": "openai"},
            {"id": "gpt-4o", "name": "GPT-4o", "provider": "openai"},
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "openai"},
        ],
        "openai_compatible": [
            {"id": config.query_generator_model, "name": config.query_generator_model, "provider": "openai_compatible"},
            {"id": config.reflection_model, "name": config.reflection_model, "provider": "openai_compatible"},
            {"id": config.answer_model, "name": config.answer_model, "provider": "openai_compatible"},
        ]
    }
    
    # Return models for the current provider
    available_models = model_definitions.get(config.llm_provider, [])
    
    return {
        "models": available_models,
        "current_provider": config.llm_provider,
        "default_models": {
            "query_generator": config.query_generator_model,
            "reflection": config.reflection_model,
            "answer": config.answer_model
        }
    }


# Mount the frontend under /app to not conflict with the LangGraph API routes
app.mount(
    "/app",
    create_frontend_router(),
    name="frontend",
)
