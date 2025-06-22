import os
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any, Optional, Literal

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent."""

    # LLM Provider Configuration
    llm_provider: Literal["google", "openai", "openai_compatible"] = Field(
        default="google",
        metadata={
            "description": "The LLM provider to use. Options: 'google', 'openai', 'openai_compatible'"
        },
    )

    # OpenAI Compatible API Configuration
    openai_compatible_base_url: Optional[str] = Field(
        default=None,
        metadata={
            "description": "Base URL for OpenAI-compatible API endpoints (e.g., http://localhost:1234/v1)"
        },
    )

    openai_compatible_api_key: Optional[str] = Field(
        default=None,
        metadata={
            "description": "API key for OpenAI-compatible services"
        },
    )

    # Search Tool Configuration
    search_tool: Literal["google", "firecrawl", "brave", "none"] = Field(
        default="google",
        metadata={
            "description": "Search tool to use for web research. Options: 'google', 'firecrawl', 'brave', 'none'"
        },
    )

    # Firecrawl API Configuration
    firecrawl_api_key: Optional[str] = Field(
        default=None,
        metadata={
            "description": "API key for Firecrawl service"
        },
    )

    firecrawl_base_url: Optional[str] = Field(
        default="https://api.firecrawl.dev",
        metadata={
            "description": "Base URL for Firecrawl API"
        },
    )

    # Brave Search API Configuration
    brave_api_key: Optional[str] = Field(
        default=None,
        metadata={
            "description": "API key for Brave Search service"
        },
    )

    brave_base_url: Optional[str] = Field(
        default="https://api.search.brave.com",
        metadata={
            "description": "Base URL for Brave Search API"
        },
    )

    # Search Configuration
    max_search_results: int = Field(
        default=5,
        metadata={
            "description": "Maximum number of search results to process per query"
        },
    )

    max_content_length: int = Field(
        default=4000,
        metadata={
            "description": "Maximum length of content to extract from each web page"
        },
    )

    # Model Configuration
    query_generator_model: str = Field(
        default="gemini-2.0-flash",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reflection_model: str = Field(
        default="gemini-2.5-flash",
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    answer_model: str = Field(
        default="gemini-2.5-pro",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    @classmethod
    def load_config_file(cls, config_path: Optional[str] = None) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            # Try to find config.yaml in current directory or project root
            possible_paths = [
                Path("config.yaml"),
                Path("backend/config.yaml"),
                Path("src/config.yaml"),
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        
        return {}

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None, config_file: Optional[str] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig and optional YAML config file."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Load from YAML config file first
        file_config = cls.load_config_file(config_file)
        
        # Flatten nested config structure
        flattened_config = {}
        if file_config:
            flattened_config["llm_provider"] = file_config.get("llm_provider", "google")
            flattened_config["search_tool"] = file_config.get("search_tool", "google")
            
            if "openai_compatible" in file_config:
                flattened_config["openai_compatible_base_url"] = file_config["openai_compatible"].get("base_url")
                flattened_config["openai_compatible_api_key"] = file_config["openai_compatible"].get("api_key")
            
            if "firecrawl" in file_config:
                flattened_config["firecrawl_api_key"] = file_config["firecrawl"].get("api_key")
                flattened_config["firecrawl_base_url"] = file_config["firecrawl"].get("base_url")
            
            if "brave" in file_config:
                flattened_config["brave_api_key"] = file_config["brave"].get("api_key")
                flattened_config["brave_base_url"] = file_config["brave"].get("base_url")
            
            if "search" in file_config:
                flattened_config["max_search_results"] = file_config["search"].get("max_search_results")
                flattened_config["max_content_length"] = file_config["search"].get("max_content_length")
                
            if "models" in file_config:
                flattened_config["query_generator_model"] = file_config["models"].get("query_generator")
                flattened_config["reflection_model"] = file_config["models"].get("reflection") 
                flattened_config["answer_model"] = file_config["models"].get("answer")
            
            if "research" in file_config:
                flattened_config["number_of_initial_queries"] = file_config["research"].get("number_of_initial_queries")
                flattened_config["max_research_loops"] = file_config["research"].get("max_research_loops")

        # Get raw values from environment, runnable config, or file config (in order of precedence)
        raw_values: dict[str, Any] = {}
        for name in cls.model_fields.keys():
            # Priority: environment > runnable config > file config
            raw_values[name] = (
                os.environ.get(name.upper()) or 
                configurable.get(name) or 
                flattened_config.get(name)
            )

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        # Create instance and validate
        instance = cls(**values)
        instance._validate_configuration()
        return instance

    def _validate_configuration(self):
        """Validate configuration for consistency and completeness."""
        # Only validate requirements if actually using the provider/tool
        
        # Validate LLM provider requirements
        if self.llm_provider == "google":
            if not os.getenv("GEMINI_API_KEY"):
                print("Warning: GEMINI_API_KEY environment variable not found for Google provider")
        
        elif self.llm_provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                print("Warning: OPENAI_API_KEY environment variable not found for OpenAI provider")
        
        elif self.llm_provider == "openai_compatible":
            base_url = self.openai_compatible_base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
            api_key = self.openai_compatible_api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY")
            
            if not base_url:
                print("Warning: OpenAI compatible base URL not configured")
            if not api_key:
                print("Warning: OpenAI compatible API key not configured")
        
        # Validate search tool requirements
        if self.search_tool == "firecrawl":
            if not (self.firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")):
                print("Warning: Firecrawl API key not found when using Firecrawl search tool")
        
        elif self.search_tool == "brave":
            if not (self.brave_api_key or os.getenv("BRAVE_API_KEY")):
                print("Warning: Brave Search API key not found when using Brave search tool")
        
        elif self.search_tool == "google" and self.llm_provider != "google":
            print("Warning: Google search tool works best with Google LLM provider for grounding functionality")
        
        # Validate model compatibility (warn only, don't fail)
        self._validate_model_compatibility()

    def _validate_model_compatibility(self):
        """Validate model names are compatible with the selected provider."""
        model_warnings = []
        
        if self.llm_provider == "openai":
            openai_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]
            for field_name, model in [
                ("query_generator_model", self.query_generator_model),
                ("reflection_model", self.reflection_model),
                ("answer_model", self.answer_model)
            ]:
                if model not in openai_models:
                    model_warnings.append(f"Warning: {model} may not be compatible with OpenAI provider")
        
        elif self.llm_provider == "google":
            google_models = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
            for field_name, model in [
                ("query_generator_model", self.query_generator_model),
                ("reflection_model", self.reflection_model),
                ("answer_model", self.answer_model)
            ]:
                if model not in google_models:
                    model_warnings.append(f"Warning: {model} may not be a valid Google model")
        
        # Print warnings if any
        for warning in model_warnings:
            print(warning)

    def get_llm_client(self, model_name: str, temperature: float = 0.0, max_retries: int = 2):
        """Create an LLM client based on the configured provider."""
        if self.llm_provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_retries=max_retries,
                api_key=os.getenv("GEMINI_API_KEY"),
            )
        elif self.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_retries=max_retries,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif self.llm_provider == "openai_compatible":
            from langchain_openai import ChatOpenAI
            base_url = self.openai_compatible_base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
            api_key = self.openai_compatible_api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY")
            
            if not base_url:
                raise ValueError("OpenAI compatible base URL must be configured")
            if not api_key:
                raise ValueError("OpenAI compatible API key must be configured")
                
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_retries=max_retries,
                base_url=base_url,
                api_key=api_key,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
