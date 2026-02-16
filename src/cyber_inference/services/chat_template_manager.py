"""
Chat template management for Cyber-Inference.

Provides:
- Loading and caching of Jinja2 chat templates
- Rendering templates with messages and system prompts
- Template discovery and validation
"""

import logging
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, Template, TemplateError, TemplateSyntaxError

from cyber_inference.core.config import get_settings
from cyber_inference.core.logging import get_logger

logger = get_logger(__name__)


class ChatTemplateManager:
    """
    Manages custom Jinja2 chat templates for model inference.

    Templates are loaded from a configurable directory and cached in memory.
    Each template should be in Jinja2 format with access to:
    - messages: List of message dicts with 'role' and 'content'
    - system_prompt: Optional system prompt string
    """

    def __init__(self) -> None:
        """Initialize the chat template manager."""
        self.settings = get_settings()
        self.templates_dir = self.settings.chat_templates_dir
        self.template_cache: dict[str, Template] = {}
        self.env: Optional[Environment] = None

        self._initialize_env()
        self._load_builtin_templates()

    def _initialize_env(self) -> None:
        """Initialize Jinja2 environment."""
        if not self.templates_dir or not self.templates_dir.exists():
            logger.debug("No custom templates directory configured or found")
            self.env = Environment()
        else:
            logger.info(f"Initializing chat templates from: {self.templates_dir}")
            self.env = Environment(loader=FileSystemLoader(self.templates_dir))

    def _load_builtin_templates(self) -> None:
        """Load built-in default templates."""
        # Default simple template
        default_template = """{% for msg in messages %}{{ msg.role }}: {{ msg.content }}
{% endfor %}"""
        self.template_cache["default"] = Template(default_template)

        # Llama2 format
        llama2_template = (
            """{% if system_prompt %}[INST] <<SYS>>\n{{ system_prompt }}\n<</SYS>>\n{% endif %}\
{% for msg in messages %}\
{% if msg.role == "user" %}\
{{ msg.content }} [/INST]\n\
{% elif msg.role == "assistant" %}\
{{ msg.content }} </s><s>[INST]\n\
{% endif %}\
{% endfor %}"""
        )
        self.template_cache["llama2"] = Template(llama2_template)

        # Mistral format
        mistral_template = (
            """[INST] {% if system_prompt %}{{ system_prompt }}\n\n{% endif %}\
{% for msg in messages %}\
{% if msg.role == "user" %}\
{{ msg.content }}\
{% elif msg.role == "assistant" %}\
 [/INST] {{ msg.content }} [INST] \
{% endif %}\
{% endfor %}\
[/INST]"""
        )
        self.template_cache["mistral"] = Template(mistral_template)

        # OpenChat format
        openchat_template = (
            """GPT4 Correct Assistant: {% if system_prompt %}{{ system_prompt }}\n{% endif %}\
{% for msg in messages %}\
{% if msg.role == "user" %}\
GPT4 Correct User: {{ msg.content }}<|end_of_turn|>\n\
{% elif msg.role == "assistant" %}\
GPT4 Correct Assistant: {{ msg.content }}<|end_of_turn|>\n\
{% endif %}\
{% endfor %}\
GPT4 Correct Assistant:"""
        )
        self.template_cache["openchat"] = Template(openchat_template)

        logger.debug(f"Loaded {len(self.template_cache)} built-in templates")

    def load_template(self, model_name: str) -> Template:
        """
        Load a chat template for a specific model.

        Tries to load in this order:
        1. Custom template matching model name (from configured directory)
        2. Custom default template (from configured directory)
        3. Built-in template matching model name
        4. Built-in default template

        Args:
            model_name: Name of the model (e.g., 'llama2-7b')

        Returns:
            Jinja2 Template object
        """
        # Check cache first
        if model_name in self.template_cache:
            return self.template_cache[model_name]

        # Try to load from custom directory if configured
        if self.templates_dir and self.templates_dir.exists():
            template_path = self.templates_dir / f"{model_name}.jinja2"
            if template_path.exists():
                try:
                    with open(template_path, "r") as f:
                        content = f.read()
                    template = Template(content)
                    self.template_cache[model_name] = template
                    logger.info(f"Loaded custom template for {model_name}: {template_path}")
                    return template
                except (TemplateError, OSError) as e:
                    logger.warning(f"Failed to load custom template {template_path}: {e}")

        # Try to load custom default template
        if self.templates_dir and self.templates_dir.exists():
            default_path = self.templates_dir / "default.jinja2"
            if default_path.exists():
                try:
                    with open(default_path, "r") as f:
                        content = f.read()
                    template = Template(content)
                    self.template_cache["custom_default"] = template
                    logger.info(f"Using custom default template for {model_name}")
                    return template
                except (TemplateError, OSError) as e:
                    logger.warning(f"Failed to load custom default template: {e}")

        # Check if model-specific built-in exists
        if model_name in self.template_cache:
            logger.debug(f"Using built-in template for {model_name}")
            return self.template_cache[model_name]

        # Fall back to default template
        logger.debug(f"Using default template for {model_name}")
        return self.template_cache["default"]

    def render_chat_template(
        self,
        model_name: str,
        messages: list[dict],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Render a chat template with messages and system prompt.

        Args:
            model_name: Name of the model
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt string

        Returns:
            Rendered prompt string

        Raises:
            TemplateError: If template rendering fails
        """
        try:
            template = self.load_template(model_name)
            rendered = template.render(
                messages=messages,
                system_prompt=system_prompt,
            )
            logger.debug(
                f"Rendered chat template for {model_name} "
                f"with {len(messages)} messages"
            )
            return rendered
        except TemplateError as e:
            logger.error(f"Failed to render template for {model_name}: {e}")
            raise

    def get_available_templates(self) -> list[str]:
        """
        Get list of available template names.

        Returns:
            List of template names (built-in and custom)
        """
        templates = set(self.template_cache.keys())

        # Add custom templates from directory
        if self.templates_dir and self.templates_dir.exists():
            for template_file in self.templates_dir.glob("*.jinja2"):
                template_name = template_file.stem
                templates.add(template_name)

        return sorted(list(templates))

    def validate_templates(self) -> dict[str, bool | str]:
        """
        Validate all templates in the custom templates directory.

        Returns:
            Dict mapping template names to validation status/error message
        """
        validation_results: dict[str, bool | str] = {}

        if not self.templates_dir or not self.templates_dir.exists():
            logger.debug("No custom templates directory to validate")
            return validation_results

        for template_file in self.templates_dir.glob("*.jinja2"):
            template_name = template_file.stem
            try:
                with open(template_file, "r") as f:
                    content = f.read()
                Template(content)
                validation_results[template_name] = True
                logger.debug(f"Template {template_name} is valid")
            except TemplateError as e:
                error_msg = str(e)
                validation_results[template_name] = error_msg
                logger.warning(f"Template {template_name} has error: {error_msg}")

        return validation_results

    def get_template_info(self, template_name: str) -> Optional[dict]:
        """
        Get information about a specific template.

        Args:
            template_name: Name of the template

        Returns:
            Dict with template info or None if not found
        """
        # Check in cache
        if template_name in self.template_cache:
            return {
                "name": template_name,
                "source": "built-in",
                "path": None,
            }

        # Check in custom directory
        if self.templates_dir and self.templates_dir.exists():
            template_path = self.templates_dir / f"{template_name}.jinja2"
            if template_path.exists():
                return {
                    "name": template_name,
                    "source": "custom",
                    "path": str(template_path),
                }

        return None

    def get_template_path(self, name: str) -> Optional[Path]:
        """
        Get the filesystem path to a template file.

        Args:
            name: Template name

        Returns:
            Path to template file or None if not found
        """
        if self.templates_dir and self.templates_dir.exists():
            template_path = self.templates_dir / f"{name}.jinja2"
            if template_path.exists():
                return template_path
        return None
