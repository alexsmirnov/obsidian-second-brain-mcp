"""Expose vault skills that opt in with `exposed: true` as MCP skill resources."""

import logging
from pathlib import Path

import frontmatter
import yaml
from fastmcp.server.providers.skills import SkillProvider, SkillsDirectoryProvider

logger = logging.getLogger("mcps")

SKILL_FILE_NAME = "SKILL.md"


def is_exposed(skill_file: Path) -> bool:
    """True when the skill's frontmatter sets `exposed` to YAML true."""
    try:
        metadata = frontmatter.load(skill_file).metadata
    except (yaml.YAMLError, UnicodeDecodeError, OSError):
        return False
    return metadata.get("exposed") is True


class ExposedSkillsProvider(SkillsDirectoryProvider):
    """Skills directory that serves only skills flagged `exposed: true`.

    Rescans on every request, so editing the flag in the vault takes effect
    without a server restart.
    """

    def __init__(self, root: Path) -> None:
        super().__init__(roots=root, reload=True, main_file_name=SKILL_FILE_NAME)

    def _discover_skills(self) -> None:
        """Build a provider for each exposed skill only.

        The flag is read before FastMCP touches the skill, so an unexposed or
        unreadable skill can neither slow discovery nor stop the server.
        """
        self.providers.clear()
        for root in self._roots:
            if not root.is_dir():
                continue
            for skill_dir in sorted(root.iterdir()):
                if not is_exposed(skill_dir / SKILL_FILE_NAME):
                    continue
                try:
                    self.providers.append(
                        SkillProvider(
                            skill_path=skill_dir,
                            main_file_name=SKILL_FILE_NAME,
                            supporting_files="template",
                        )
                    )
                except Exception:
                    logger.exception("Failed to load skill: %s", skill_dir.name)
        self._discovered = True
