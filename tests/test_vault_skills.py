from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp.exceptions import NotFoundError

from mcps.config import ServerConfig
from mcps.server import create_server
from mcps.tools.vault_skills import is_exposed


def write_skill(vault: Path, name: str, frontmatter_lines: str) -> Path:
    skill_file = vault / "Claude/skills" / name / "SKILL.md"
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    skill_file.write_text(
        f"---\nname: {name}\ndescription: Skill {name}\n{frontmatter_lines}---\n\n"
        f"Body of {name}\n"
    )
    return skill_file


async def skill_uris(vault: Path) -> set[str]:
    server = create_server(ServerConfig(vault_dir=vault))
    return {
        str(resource.uri)
        for resource in await server.mcp.list_resources()
        if str(resource.uri).startswith("skill://")
    }


class TestIsExposed:
    @pytest.mark.parametrize(
        ("frontmatter_lines", "expected"),
        [
            ("exposed: true\n", True),
            ("exposed: yes\n", True),
            ('exposed: "yes"\n', False),
            ("exposed: false\n", False),
            ("", False),
            ("exposed: [unclosed\n", False),
        ],
    )
    def test_flag_values(
        self, tmp_path: Path, frontmatter_lines: str, expected: bool
    ):
        skill_file = write_skill(tmp_path, "s", frontmatter_lines)

        assert is_exposed(skill_file) is expected

    def test_list_frontmatter_is_not_exposed(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("---\n- a\n- b\n---\nbody\n")

        assert is_exposed(skill_file) is False

    def test_missing_file_is_not_exposed(self, tmp_path: Path):
        assert is_exposed(tmp_path / "missing" / "SKILL.md") is False


class TestExposedSkillsProvider:
    async def test_only_skills_flagged_exposed_are_listed(self, tmp_path: Path):
        write_skill(tmp_path, "a", "exposed: true\n")
        write_skill(tmp_path, "b", "")
        write_skill(tmp_path, "c", 'exposed: "yes"\n')
        write_skill(tmp_path, "d", "exposed: [unclosed\n")

        uris = await skill_uris(tmp_path)

        assert "skill://a/SKILL.md" in uris
        assert not {uri for uri in uris if uri.split("/")[2] in {"b", "c", "d"}}

    async def test_flag_change_applies_without_restart(self, tmp_path: Path):
        write_skill(tmp_path, "b", "")
        server = create_server(ServerConfig(vault_dir=tmp_path))
        before = {str(r.uri) for r in await server.mcp.list_resources()}

        write_skill(tmp_path, "b", "exposed: true\n")
        after = {str(r.uri) for r in await server.mcp.list_resources()}

        assert "skill://b/SKILL.md" not in before
        assert "skill://b/SKILL.md" in after

    async def test_unreadable_skill_does_not_hide_the_others(self, tmp_path: Path):
        write_skill(tmp_path, "a", "exposed: true\n")
        broken = tmp_path / "Claude/skills/broken/SKILL.md"
        broken.parent.mkdir(parents=True)
        broken.write_bytes(b"---\nexposed: true\n---\n\xff\xfe invalid utf-8")

        uris = await skill_uris(tmp_path)

        assert "skill://a/SKILL.md" in uris
        assert not {uri for uri in uris if "broken" in uri}

    async def test_revoked_flag_hides_skill_on_next_read(self, tmp_path: Path):
        write_skill(tmp_path, "a", "exposed: true\n")
        server = create_server(ServerConfig(vault_dir=tmp_path))
        assert await server.mcp.read_resource("skill://a/SKILL.md")

        write_skill(tmp_path, "a", "exposed: false\n")

        with pytest.raises(NotFoundError):
            await server.mcp.read_resource("skill://a/SKILL.md")

    async def test_missing_skills_folder_exposes_nothing(self, tmp_path: Path):
        assert await skill_uris(tmp_path) == set()

    async def test_exposed_skill_can_be_read(self, tmp_path: Path):
        write_skill(tmp_path, "a", "exposed: true\n")
        server = create_server(ServerConfig(vault_dir=tmp_path))

        result = await server.mcp.read_resource("skill://a/SKILL.md")

        assert "Body of a" in str(result.contents[0].content)

    async def test_unexposed_skill_cannot_be_read(self, tmp_path: Path):
        write_skill(tmp_path, "a", "exposed: true\n")
        write_skill(tmp_path, "b", "")
        server = create_server(ServerConfig(vault_dir=tmp_path))

        assert await server.mcp.read_resource("skill://a/SKILL.md")
        with pytest.raises(NotFoundError):
            await server.mcp.read_resource("skill://b/SKILL.md")

    async def test_skills_are_absent_without_vault(self):
        server = create_server(ServerConfig(vault_dir=None))

        uris = {str(r.uri) for r in await server.mcp.list_resources()}

        assert not {uri for uri in uris if uri.startswith("skill://")}
