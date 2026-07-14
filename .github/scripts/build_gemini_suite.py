"""Verify and build a deterministic coordinated Gemini suite archive."""

import argparse
import ast
import gzip
import hashlib
import io
import re
import tarfile
import tomllib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    role: Literal["pipe", "companion", "catalog", "toggle", "release_notes"]
    version: str | int
    path: str
    sha256: str
    bytes: int = Field(ge=1)

    @field_validator("sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value


class Protocols(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    google_genai: str
    grounding_envelope: int
    model_catalog: int


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    suite: Literal["gemini-suite"]
    suite_version: str
    tag: str
    release_policy: Literal["coordinated_only"]
    protocols: Protocols
    install_order: list[str]
    artifacts: list[Artifact]

    @field_validator("artifacts")
    @classmethod
    def validate_artifacts(cls, artifacts: list[Artifact]) -> list[Artifact]:
        ids = [artifact.id for artifact in artifacts]
        paths = [artifact.path for artifact in artifacts]
        if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
            raise ValueError("artifact ids and paths must be unique")
        roles = {artifact.role for artifact in artifacts}
        if not {"pipe", "companion", "catalog"} <= roles:
            raise ValueError("manifest must contain a pipe, companion, and catalog")
        return artifacts


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def repository_root(manifest_path: Path) -> Path:
    root = manifest_path.resolve().parents[1]
    if not (root / "pyproject.toml").is_file():
        raise ValueError(f"manifest is not under a repository root: {manifest_path}")
    return root


def artifact_path(root: Path, artifact: Artifact) -> Path:
    candidate = (root / artifact.path).resolve()
    candidate.relative_to(root)
    if not candidate.is_file():
        raise FileNotFoundError(artifact.path)
    return candidate


def frontmatter(path: Path) -> dict[str, str]:
    match = re.match(r'\A"""\n(?P<body>.*?)\n"""', path.read_text(encoding="utf-8"), re.DOTALL)
    if match is None:
        raise ValueError(f"missing frontmatter in {path}")
    values: dict[str, str] = {}
    for line in match.group("body").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def constants(path: Path) -> dict[str, str | int]:
    values: dict[str, str | int] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        if value_node is None or not isinstance(value_node, ast.Constant):
            continue
        if not isinstance(value_node.value, (str, int)):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                values[target.id] = value_node.value
    return values


def verify_suite_contract(root: Path, manifest: Manifest) -> None:
    artifacts = {artifact.id: artifact for artifact in manifest.artifacts}
    for artifact in manifest.artifacts:
        path = artifact_path(root, artifact)
        if path.suffix == ".py":
            metadata = frontmatter(path)
            if metadata.get("id") != artifact.id:
                raise ValueError(f"frontmatter id mismatch for {artifact.path}")
            if metadata.get("version") != str(artifact.version):
                raise ValueError(f"frontmatter version mismatch for {artifact.path}")

    pipe = artifact_path(root, artifacts["gemini_manifold_google_genai"])
    companion = artifact_path(root, artifacts["gemini_manifold_companion"])
    pipe_metadata = frontmatter(pipe)
    pipe_constants = constants(pipe)
    companion_constants = constants(companion)
    expected_catalog_url = (
        "https://raw.githubusercontent.com/suurt8ll/open_webui_functions/"
        f"{manifest.tag}/plugins/pipes/gemini_models.yaml"
    )
    if pipe_constants.get("VERSION") != manifest.suite_version:
        raise ValueError("pipe VERSION does not match suite")
    if pipe_constants.get("RECOMMENDED_COMPANION_VERSION") != manifest.suite_version:
        raise ValueError("pipe companion requirement does not match suite")
    if pipe_constants.get("MODEL_CATALOG_SCHEMA_VERSION") != manifest.protocols.model_catalog:
        raise ValueError("pipe catalog protocol does not match manifest")
    if (
        pipe_constants.get("GROUNDING_ENVELOPE_PROTOCOL_VERSION")
        != manifest.protocols.grounding_envelope
    ):
        raise ValueError("pipe grounding protocol does not match manifest")
    if companion_constants.get("VERSION") != manifest.suite_version:
        raise ValueError("companion VERSION does not match suite")
    if companion_constants.get("MODEL_CATALOG_SCHEMA_VERSION") != manifest.protocols.model_catalog:
        raise ValueError("companion catalog protocol does not match manifest")
    if (
        companion_constants.get("GROUNDING_ENVELOPE_PROTOCOL_VERSION")
        != manifest.protocols.grounding_envelope
    ):
        raise ValueError("companion grounding protocol does not match manifest")
    if companion_constants.get("DEFAULT_MODEL_CONFIG_PATH") != expected_catalog_url:
        raise ValueError("companion catalog URL is not pinned to the suite tag")
    if (
        pipe_metadata.get("requirements")
        != f"google-genai=={manifest.protocols.google_genai}, pikepdf"
    ):
        raise ValueError("pipe SDK requirement does not match manifest")

    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    if f"google-genai=={manifest.protocols.google_genai}" not in dependencies:
        raise ValueError("project SDK requirement does not match manifest")
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    locked_google_genai = [
        package
        for package in lock["package"]
        if isinstance(package, dict) and package.get("name") == "google-genai"
    ]
    if (
        len(locked_google_genai) != 1
        or locked_google_genai[0].get("version") != manifest.protocols.google_genai
    ):
        raise ValueError("locked SDK version does not match manifest")
    catalog_path = artifact_path(root, artifacts["gemini_model_catalog"])
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    if (
        not isinstance(catalog, dict)
        or catalog.get("schema_version") != manifest.protocols.model_catalog
    ):
        raise ValueError("catalog schema does not match manifest")

    release_notes = artifact_path(root, artifacts["gemini_suite_release_notes"]).read_text(
        encoding="utf-8"
    )
    for heading in ("## Upgrade sequence", "## Rollback", "## Current limitations"):
        if heading not in release_notes:
            raise ValueError(f"release notes are missing {heading}")


def verify_manifest(manifest_path: Path, *, expected_tag: str | None = None) -> Manifest:
    manifest = load_manifest(manifest_path)
    root = repository_root(manifest_path)
    if expected_tag is not None and expected_tag != manifest.tag:
        raise ValueError(f"tag {expected_tag!r} does not match manifest tag {manifest.tag!r}")
    if manifest.tag != f"{manifest.suite}/v{manifest.suite_version}":
        raise ValueError("manifest tag and suite version disagree")
    deployable_ids = {
        artifact.id
        for artifact in manifest.artifacts
        if artifact.role not in {"catalog", "release_notes"}
    }
    if len(manifest.install_order) != len(set(manifest.install_order)):
        raise ValueError("install_order must not contain duplicates")
    if set(manifest.install_order) != deployable_ids:
        raise ValueError("install_order must contain every deployable artifact exactly once")
    for artifact in manifest.artifacts:
        path = artifact_path(root, artifact)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != artifact.sha256:
            raise ValueError(f"sha256 mismatch for {artifact.path}: {digest}")
        if len(content) != artifact.bytes:
            raise ValueError(f"byte-size mismatch for {artifact.path}: {len(content)}")
    verify_suite_contract(root, manifest)
    return manifest


def build_bundle(manifest_path: Path, output: Path, *, expected_tag: str | None = None) -> Path:
    manifest = verify_manifest(manifest_path, expected_tag=expected_tag)
    root = repository_root(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    members = [(manifest_path.resolve(), f"manifest/{manifest_path.name}")]
    members.extend(
        (artifact_path(root, artifact), artifact.path)
        for artifact in sorted(manifest.artifacts, key=lambda item: item.path)
    )
    with output.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for source, archive_name in members:
                    content = source.read_bytes()
                    info = tarfile.TarInfo(archive_name)
                    info.size = len(content)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    archive.addfile(info, io.BytesIO(content))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tag")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = verify_manifest(args.manifest, expected_tag=args.tag)
    if args.verify_only:
        print(f"verified {manifest.tag}")
        return
    output = args.output or Path("dist") / f"{manifest.suite}-v{manifest.suite_version}.tar.gz"
    build_bundle(args.manifest, output, expected_tag=args.tag)
    print(output)


if __name__ == "__main__":
    main()
