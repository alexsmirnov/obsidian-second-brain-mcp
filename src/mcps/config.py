# mcps/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class ServerConfig:
    prompts_dir: Path = field(default_factory=lambda: Path("./prompts"))
    cache_dir: Path = field(default_factory=lambda: Path("./cache"))
    tests_dir: Path = field(default_factory=lambda: Path("./tests"))
    library_docs: Dict[str, str] = field(default_factory=dict)
    project_paths: Dict[str, str] = field(default_factory=dict)


def create_config(
    prompts_dir: Path = Path("./prompts"),
    cache_dir: Path = Path("./cache"),
    tests_dir: Path = Path("./tests"),
    library_docs: Dict[str, str] | None = None,
    project_paths: Dict[str, str] | None = None,
) -> ServerConfig:
    """
    Creates a ServerConfig instance, ensuring directories exist and
    handling default values for library_docs and project_paths.
    """

    for dir_path in [prompts_dir, cache_dir, tests_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Use provided dictionaries or default to empty dictionaries
    library_docs = library_docs if library_docs is not None else {}
    project_paths = project_paths if project_paths is not None else {}

    return ServerConfig(
        prompts_dir=prompts_dir,
        cache_dir=cache_dir,
        tests_dir=tests_dir,
        library_docs=library_docs,
        project_paths=project_paths,
    )
