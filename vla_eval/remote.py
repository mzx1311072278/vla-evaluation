import ipaddress
import os
import re
import shlex
import stat
import unicodedata
from pathlib import Path, PurePosixPath

from vla_eval.config import RemoteSource

_SSH_USERNAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_DNS_LABEL_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z")
_RSYNC_VERSION_PATTERN = re.compile(r"^rsync\s+version\s+(\d+)\.(\d+)\.(\d+)(?=\s|$)", re.MULTILINE)
_SSH_CONNECT_TIMEOUT_SECONDS = 10
_MINIMUM_RSYNC_VERSION = (3, 2, 7)
_REMOTE_SEGMENT_PUNCTUATION = frozenset("._-")


def _contains_control_or_format_character(value: str) -> bool:
    return any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value)


def _validate_transport_safe_segment(segment: str, field_name: str) -> None:
    if segment in {".", ".."}:
        raise ValueError(f"{field_name} must not contain '.' or '..' segments")
    if segment.startswith("-"):
        raise ValueError(f"{field_name} segments must not start with '-'")
    if unicodedata.normalize("NFC", segment) != segment:
        raise ValueError(f"{field_name} must be NFC-normalized before use")
    if any(
        character not in _REMOTE_SEGMENT_PUNCTUATION
        and unicodedata.category(character)[0] not in {"L", "N"}
        for character in segment
    ):
        raise ValueError(
            f"{field_name} segments may contain only Unicode letters/numbers and '._-'"
        )


def normalize_remote_relative_path(value: str) -> str:
    """Validate and preserve a relative POSIX path for an rsync remote operand."""
    if not isinstance(value, str):
        raise TypeError("remote path must be a string")
    if not value or not value.strip():
        raise ValueError("remote path must not be empty or whitespace")
    if value.endswith("/"):
        raise ValueError("remote path must not have a trailing slash")
    if "\\" in value:
        raise ValueError("remote path must use POSIX separators")
    if _contains_control_or_format_character(value):
        raise ValueError("remote path must not contain control or format characters")
    if any(unicodedata.category(character) in {"Zl", "Zp"} for character in value):
        raise ValueError("remote path must not contain line or paragraph separators")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("remote path must be NFC-normalized before use")

    raw_segments = value.split("/")
    if any(not segment or not segment.strip() for segment in raw_segments):
        raise ValueError("remote path must not contain empty or whitespace-only segments")
    for segment in raw_segments:
        _validate_transport_safe_segment(segment, "remote path")

    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError("remote path must be relative")
    if str(path) != value:
        raise ValueError("remote path must already be normalized")
    return value


def _validate_remote_root(source: RemoteSource, remote_root: str) -> str:
    if len(source.roots) != 1:
        raise ValueError("rrsync transport requires exactly one configured remote root")
    if not isinstance(remote_root, str) or remote_root != source.roots[0]:
        raise ValueError("selected remote root must equal the source's sole forced root")
    if "\\" in remote_root or _contains_control_or_format_character(remote_root):
        raise ValueError("remote root is not a safe POSIX path")

    path = PurePosixPath(remote_root)
    if not path.is_absolute() or remote_root.startswith("//"):
        raise ValueError("remote root must be an absolute POSIX path")
    if ".." in path.parts or str(path) != remote_root:
        raise ValueError("remote root must already be canonical")
    for segment in path.parts:
        if segment != "/":
            _validate_transport_safe_segment(segment, "remote root")
    return remote_root


def _validate_username(username: str) -> str:
    if not isinstance(username, str) or _SSH_USERNAME_PATTERN.fullmatch(username) is None:
        raise ValueError("remote username is not a safe SSH token")
    return username


def _validated_host(host: str) -> tuple[str, bool]:
    if not isinstance(host, str) or not host or _contains_control_or_format_character(host):
        raise ValueError("remote host is invalid")

    if ":" in host:
        if "%" in host:
            raise ValueError("scoped IPv6 remote hosts are not supported")
        try:
            ipaddress.IPv6Address(host)
        except ipaddress.AddressValueError as error:
            raise ValueError("remote host is not a valid IPv6 address") from error
        return host, True

    try:
        ipaddress.IPv4Address(host)
    except ipaddress.AddressValueError:
        pass
    else:
        return host, False

    if all(character.isdigit() or character == "." for character in host):
        raise ValueError("remote host is not a valid IPv4 address")

    dns_name = host.removesuffix(".")
    if not dns_name or len(dns_name) > 253:
        raise ValueError("remote host is not a valid DNS name")
    if any(_DNS_LABEL_PATTERN.fullmatch(label) is None for label in dns_name.split(".")):
        raise ValueError("remote host is not a valid DNS name")
    return host, False


def _validate_port(port: int) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("remote port must be an integer from 1 to 65535")
    return port


def _validated_config_path(value: Path, field_name: str) -> Path:
    try:
        path = Path(value)
    except TypeError as error:
        raise ValueError(f"{field_name} must be a filesystem path") from error
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    if _contains_control_or_format_character(str(path)):
        raise ValueError(f"{field_name} must not contain control or format characters")
    return path


def ssh_command_from_trusted_config(source: RemoteSource) -> str:
    """Build rsync's shell-parsed SSH transport command from administrator config."""
    _validated_host(source.host)
    port = _validate_port(source.port)
    _validate_username(source.username)
    key_path = _validated_config_path(source.key_path, "SSH key path")
    known_hosts_path = _validated_config_path(source.known_hosts_path, "known-hosts path")

    return shlex.join(
        [
            "ssh",
            "-F",
            "none",
            "-T",
            "-p",
            str(port),
            "-i",
            str(key_path),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "PermitLocalCommand=no",
            "-o",
            "RequestTTY=no",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={known_hosts_path}",
            "-o",
            f"ConnectTimeout={_SSH_CONNECT_TIMEOUT_SECONDS}",
        ]
    )


def validate_rsync_version_output(output: str) -> tuple[int, int, int]:
    """Require rsync >=3.2.7 features, not a distribution security revision.

    Task 15 deployment must separately require the current Ubuntu security package.
    """
    if not isinstance(output, str):
        raise TypeError("rsync version output must be a string")
    match = _RSYNC_VERSION_PATTERN.search(output)
    if match is None:
        raise ValueError("could not parse a stable GNU rsync version")
    version = tuple(int(part) for part in match.groups())
    if version < _MINIMUM_RSYNC_VERSION:
        raise ValueError("GNU rsync 3.2.7 or newer is required")
    return version


def _lstat_components(
    path: Path, field_name: str, *, missing_message: str | None = None
) -> list[tuple[Path, os.stat_result]]:
    current = Path(path.anchor)
    inspected: list[tuple[Path, os.stat_result]] = []
    try:
        for component in (None, *path.parts[1:]):
            if component is not None:
                current /= component
            current_stat = os.lstat(current)
            if stat.S_ISLNK(current_stat.st_mode):
                raise ValueError(f"{field_name} must not contain symlink components")
            inspected.append((current, current_stat))
    except FileNotFoundError as error:
        raise ValueError(missing_message or f"{field_name} must exist") from error
    except OSError as error:
        raise ValueError(f"could not inspect {field_name}") from error
    return inspected


def _lstat_without_symlink_components(
    path: Path, field_name: str, *, missing_message: str | None = None
) -> os.stat_result:
    return _lstat_components(path, field_name, missing_message=missing_message)[-1][1]


def _service_has_access(path: Path, mode: int) -> bool:
    try:
        return os.access(path, mode, effective_ids=True)
    except (NotImplementedError, TypeError):
        return os.access(path, mode)


def _trust_anchor_components(
    path: Path,
    field_name: str,
    minimum_checked_ancestor: Path | None,
) -> list[tuple[Path, os.stat_result]]:
    if minimum_checked_ancestor is None:
        return _lstat_components(path, field_name)
    boundary = _absolute_lexical_path(
        minimum_checked_ancestor,
        "storage trust boundary",
    )
    if not _is_contained(path, boundary):
        raise ValueError(f"{field_name} must be at or below storage trust boundary")
    inspected: list[tuple[Path, os.stat_result]] = []
    try:
        for component in _directory_chain(boundary, path):
            component_stat = os.lstat(component)
            if stat.S_ISLNK(component_stat.st_mode):
                raise ValueError(f"{field_name} must not contain symlink components")
            inspected.append((component, component_stat))
    except FileNotFoundError as error:
        raise ValueError(f"{field_name} must exist") from error
    except OSError as error:
        raise ValueError(f"could not inspect {field_name}") from error
    return inspected


def _validate_trust_anchor_chain(
    path: Path,
    field_name: str,
    *,
    minimum_checked_ancestor: Path | None = None,
) -> None:
    allowed_owners = {0, os.geteuid()}
    for component, component_stat in _trust_anchor_components(
        path,
        field_name,
        minimum_checked_ancestor,
    ):
        if not stat.S_ISDIR(component_stat.st_mode):
            raise ValueError(f"{field_name} components must be existing directories")
        if component_stat.st_uid not in allowed_owners:
            raise ValueError(f"{field_name} must be owned by root or the service user")
        if component_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError(f"{field_name} must not be group or other writable")


def _directory_chain(root: Path, destination: Path) -> list[Path]:
    current = root
    directories = [current]
    for component in destination.relative_to(root).parts:
        current /= component
        directories.append(current)
    return directories


def _validate_credential_directory(path: Path) -> None:
    path_stat = _lstat_without_symlink_components(
        path,
        "credential directory",
        missing_message="credential directory must be an existing directory",
    )
    if not stat.S_ISDIR(path_stat.st_mode):
        raise ValueError("credential directory must be an existing directory")
    if path_stat.st_uid not in {0, os.geteuid()}:
        raise ValueError("credential directory must be owned by root or the service user")
    if path_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("credential directory must not be group or other writable")
    if path_stat.st_uid == os.geteuid() and path_stat.st_mode & stat.S_IWUSR:
        raise ValueError("service-owned credential directory must not be owner-writable")
    if not _service_has_access(path, os.X_OK):
        raise ValueError("credential directory must be searchable by the service user")


def validate_remote_source_files(source: RemoteSource, *, trusted_credentials_root: Path) -> None:
    """Validate sealed credentials; revalidate immediately before Popen.

    Component checks narrow replacement risk but cannot eliminate a same-owner TOCTOU race.
    The function never reads or returns credential contents.
    """
    credentials_root = _absolute_lexical_path(trusted_credentials_root, "trusted credentials root")
    key_path = _absolute_lexical_path(
        _validated_config_path(source.key_path, "SSH private key"), "SSH private key"
    )
    known_hosts_path = _absolute_lexical_path(
        _validated_config_path(source.known_hosts_path, "known-hosts file"),
        "known-hosts file",
    )
    for path in (key_path, known_hosts_path):
        if not _is_contained(path, credentials_root):
            raise ValueError("credential files must be within trusted credentials root")

    _validate_trust_anchor_chain(credentials_root, "trusted credentials root")
    credential_directories = {
        directory
        for path in (key_path, known_hosts_path)
        for directory in _directory_chain(credentials_root, path.parent)
    }
    for directory in sorted(credential_directories, key=lambda item: len(item.parts)):
        _validate_credential_directory(directory)

    key_stat = _lstat_without_symlink_components(key_path, "SSH private key")
    known_hosts_stat = _lstat_without_symlink_components(known_hosts_path, "known-hosts file")
    resolved_root = credentials_root.resolve(strict=True)
    for path in (key_path, known_hosts_path):
        if not _is_contained(path.resolve(strict=True), resolved_root):
            raise ValueError("resolved credential files must be within trusted credentials root")

    if not stat.S_ISREG(key_stat.st_mode):
        raise ValueError("SSH private key must be a regular file")
    if key_stat.st_uid != os.geteuid():
        raise ValueError("SSH private key must be owned by the service user")
    if key_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ValueError("SSH private key must not grant group or other permissions")
    if not key_stat.st_mode & stat.S_IRUSR or not _service_has_access(key_path, os.R_OK):
        raise ValueError("SSH private key must be readable by the service user")

    if not stat.S_ISREG(known_hosts_stat.st_mode):
        raise ValueError("known-hosts file must be a regular file")
    if known_hosts_stat.st_uid not in {0, os.geteuid()}:
        raise ValueError("known-hosts file must be owned by root or the service user")
    if known_hosts_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("known-hosts file must not be group or other writable")
    if not _service_has_access(known_hosts_path, os.R_OK):
        raise ValueError("known-hosts file must be readable by the service user")


def _absolute_lexical_path(value: Path, field_name: str) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(value)))
    except TypeError as error:
        raise ValueError(f"{field_name} must be a filesystem path") from error


def _is_contained(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _validate_service_writable_directory(
    path: Path,
    field_name: str,
    *,
    minimum_checked_ancestor: Path | None = None,
) -> None:
    if minimum_checked_ancestor is None:
        path_stat = _lstat_without_symlink_components(
            path,
            field_name,
            missing_message=f"{field_name} must be an existing directory",
        )
    else:
        path_stat = _trust_anchor_components(
            path,
            field_name,
            minimum_checked_ancestor,
        )[-1][1]
    if not stat.S_ISDIR(path_stat.st_mode):
        raise ValueError(f"{field_name} must be an existing directory")
    if path_stat.st_uid != os.geteuid():
        raise ValueError(f"{field_name} must be owned by the service user")
    if path_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(f"{field_name} must not be group or other writable")
    required_owner_mode = stat.S_IWUSR | stat.S_IXUSR
    if path_stat.st_mode & required_owner_mode != required_owner_mode:
        raise ValueError(f"{field_name} must be owner-writable and searchable")
    if not _service_has_access(path, os.W_OK | os.X_OK):
        raise ValueError(f"{field_name} must be writable and searchable by the service user")


def validate_staging_path(
    staging: Path,
    trusted_staging_root: Path,
    *,
    minimum_checked_ancestor: Path | None = None,
) -> Path:
    """Validate an existing staging directory inside a service-controlled root.

    Filesystem validation cannot eliminate a validate-to-use race. The execution layer must
    create these directories under a service-controlled root and revalidate immediately before
    starting rsync.
    """
    lexical_root = _absolute_lexical_path(trusted_staging_root, "trusted staging root")
    lexical_staging = _absolute_lexical_path(staging, "staging destination")
    if lexical_staging == Path(lexical_staging.anchor):
        raise ValueError("filesystem root cannot be used as a staging destination")
    if not _is_contained(lexical_staging, lexical_root):
        raise ValueError("staging destination must be within trusted staging root")

    _validate_trust_anchor_chain(
        lexical_root,
        "trusted staging root",
        minimum_checked_ancestor=minimum_checked_ancestor,
    )
    for directory in _directory_chain(lexical_root, lexical_staging):
        field_name = "staging destination" if directory == lexical_staging else "staging ancestor"
        _validate_service_writable_directory(
            directory,
            field_name,
            minimum_checked_ancestor=minimum_checked_ancestor,
        )

    resolved_root = lexical_root.resolve(strict=True)
    resolved_staging = lexical_staging.resolve(strict=True)
    if not _is_contained(resolved_staging, resolved_root):
        raise ValueError("resolved staging destination must be within trusted staging root")
    return resolved_staging


def build_rsync_argv(
    source: RemoteSource,
    remote_root: str,
    remote_relative_path: str,
    staging: Path,
    *,
    trusted_staging_root: Path,
    minimum_checked_ancestor: Path | None = None,
) -> list[str]:
    """Build an rrsync-compatible argv without executing rsync or a shell.

    Each source/key maps to exactly one forced ``rrsync -ro <remote_root>`` deployment root,
    so the remote operand is relative to that root. Standard rrsync rejects the remote ``-s``
    generated by ``--protect-args``; operands therefore use the conservative exact-segment
    policy. Task 9 must create and revalidate staging immediately before ``Popen``.
    """
    _validate_remote_root(source, remote_root)
    relative_path = normalize_remote_relative_path(remote_relative_path)
    username = _validate_username(source.username)
    host, is_ipv6 = _validated_host(source.host)
    destination = validate_staging_path(
        staging,
        trusted_staging_root,
        minimum_checked_ancestor=minimum_checked_ancestor,
    )

    rendered_host = f"[{host}]" if is_ipv6 else host
    remote_spec = f"{username}@{rendered_host}:{relative_path}/"
    return [
        "rsync",
        "-a",
        "--partial",
        "--append-verify",
        "--info=progress2",
        "--out-format=%i|%l|%n",
        "-e",
        ssh_command_from_trusted_config(source),
        "--",
        remote_spec,
        f"{destination}/",
    ]
