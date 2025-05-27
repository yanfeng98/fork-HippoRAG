import os
import asyncio
import importlib
from string import Template
from dataclasses import dataclass, field
from typing import Dict, List, Union, Any

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplateManager:
    role_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "system": "system",
            "user": "user",
            "assistant": "assistant"
        },
        metadata={
            "help": "Mapping from default roles in prompte template files to specific LLM providers' defined roles."
        })
    templates: Dict[str, Union[Template, List[Dict[str, Any]]]] = field(
        init=False,
        default_factory=dict,
        metadata={
            "help":
                "A dict from prompt template names to templates. A prompt template can be a Template instance or a chat history which is a list of dict with content as Template instance."
        })

    def __post_init__(self) -> None:
        """
        Initialize the templates directory and load templates.
        """
        current_file_path: str = os.path.abspath(__file__)
        package_dir: str = os.path.dirname(current_file_path)

        # abs path to dir where each *.py file (exclude __init__.py) contains a variable prompt_template (a str or a chat history with content as raw str for being converted to a Template)
        self.templates_dir: str = os.path.join(package_dir, "templates")

        self._load_templates()

    def _load_templates(self) -> None:
        """
        Load all templates from Python scripts in the templates directory.
        """
        if not os.path.exists(self.templates_dir):
            logger.error(f"Templates directory '{self.templates_dir}' does not exist.")
            raise FileNotFoundError(f"Templates directory '{self.templates_dir}' does not exist.")

        logger.info(f"Loading templates from directory: {self.templates_dir}")
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                script_name: str = os.path.splitext(filename)[0]

                try:
                    try:
                        module_name: str = f"src.hipporag.prompts.templates.{script_name}"
                        module = importlib.import_module(module_name)
                    except ModuleNotFoundError:
                        module_name: str = f".prompts.templates.{script_name}"
                        module = importlib.import_module(module_name, 'hipporag')

                    if not hasattr(module, "prompt_template"):
                        logger.error(f"Module '{module_name}' does not define a 'prompt_template'.")
                        raise AttributeError(f"Module '{module_name}' does not define a 'prompt_template'.")

                    prompt_template: list[dict[str, str]] = module.prompt_template
                    logger.debug(f"Loaded template from {module_name}")

                    if isinstance(prompt_template, Template):
                        self.templates[script_name] = prompt_template
                    elif isinstance(prompt_template, str):
                        self.templates[script_name] = Template(prompt_template)
                    elif isinstance(prompt_template, list) and all(
                            isinstance(item, dict) and "role" in item and "content" in item
                            for item in prompt_template):
                        # Adjust roles based on the provided role mapping
                        for item in prompt_template:
                            item["role"] = self.role_mapping.get(item["role"], item["role"])
                            item["content"] = item["content"] if isinstance(item["content"], Template) else Template(
                                item["content"])
                        self.templates[script_name] = prompt_template
                    else:
                        raise TypeError(
                            f"Invalid prompt_template format in '{module_name}.py'. Must be a Template or List[Dict].")

                    logger.debug(f"Successfully loaded template '{script_name}' from '{module_name}.py'.")

                except Exception as e:
                    logger.error(f"Failed to load template from '{module_name}.py': {e}")
                    raise

    def render(self, name: str, **kwargs) -> Union[str, List[Dict[str, Any]]]:
        """
        Render a template with the provided variables.

        Args:
            name (str): The name of the template.
            kwargs: Placeholder values for the template.

        Returns:
            Union[str, List[Dict[str, Any]]]: The rendered template or chat history.

        Raises:
            ValueError: If a required variable is missing.
        """
        template: list[dict[str, str | Template]] = self.get_template(name)
        if isinstance(template, Template):
            # Render a single string template
            try:
                result = template.substitute(**kwargs)
                logger.debug(f"Successfully rendered template '{name}' with variables: {kwargs}.")
                return result
            except KeyError as e:
                logger.error(f"Missing variable for template '{name}': {e}")
                raise ValueError(f"Missing variable for template '{name}': {e}")
        elif isinstance(template, list):
            # Render a chat history
            try:
                rendered_list: list[dict[str, str]] = [{
                    "role": item["role"],
                    "content": item["content"].substitute(**kwargs)
                } for item in template]
                logger.debug(f"Successfully rendered chat history template '{name}' with variables: {kwargs}.")
                return rendered_list
            except KeyError as e:
                logger.error(f"Missing variable in chat history template '{name}': {e}")
                raise ValueError(f"Missing variable in chat history template '{name}': {e}")

    def get_template(self, name: str) -> Union[Template, List[Dict[str, Any]]]:
        """
        Retrieve a template by name.

        Args:
            name (str): The name of the template.

        Returns:
            Union[Template, List[Dict[str, Any]]]: The requested template.

        Raises:
            KeyError: If the template is not found.
        """
        if name not in self.templates:
            logger.error(f"Template '{name}' not found.")
            raise KeyError(f"Template '{name}' not found.")
        logger.debug(f"Retrieved template '{name}'.")

        return self.templates[name]

    def is_template_name_valid(self, name: str) -> bool:
        return name in self.templates

    def sync_render(self, name: str, **kwargs) -> Union[str, List[Dict[str, Any]]]:
        return asyncio.run(self.render(name, **kwargs))

    def list_template_names(self) -> List[str]:
        """
        List all available template names.

        Returns:
            List[str]: A list of template names.
        """
        logger.info("Listing all available template names.")

        return list(self.templates.keys())

    def print_template(self, name: str) -> None:
        """
        Print the prompt template string or chat history structure for the given template name.

        Args:
            name (str): The name of the template.

        Raises:
            KeyError: If the template is not found.
        """
        try:
            template = self.get_template(name)
            print(f"Template name: {name}")
            if isinstance(template, Template):
                print(template.template)
            elif isinstance(template, list):
                for item in template:
                    print(f"Role: {item['role']}, Content: {item['content']}")
            logger.info(f"Printed template '{name}'.")
        except KeyError as e:
            logger.error(f"Failed to print template '{name}': {e}")
            raise
