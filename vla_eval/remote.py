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
    if not isinstance(remote_root, str) or remote_root not in source.roots:
        raise ValueError("remote root is not configured for this source")
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
    """Parse version text and require standard rsync 3.2.7 or newer without executing it."""
    if not isinstance(output, str):
        raise TypeError("rsync version output must be a string")
    match = _RSYNC_VERSION_PATTERN.search(output)
    if match is None:
        raise ValueError("could not parse a stable GNU rsync version")
    version = tuple(int(part) for part in match.groups())
    if version < _MINIMUM_RSYNC_VERSION:
        raise ValueError("GNU rsync 3.2.7 or newer is required")
    return version


def _lstat_without_symlink_components(
    path: Path, field_name: str, *, missing_message: str | None = None
) -> os.stat_result:
    current = Path(path.anchor)
    try:
        current_stat = os.lstat(current)
        for component in path.parts[1:]:
            current /= component
            current_stat = os.lstat(current)
            if stat.S_ISLNK(current_stat.st_mode):
                raise ValueError(f"{field_name} must not contain symlink components")
    except FileNotFoundError as error:
        raise ValueError(missing_message or f"{field_name} must exist") from error
    except OSError as error:
        raise ValueError(f"could not inspect {field_name}") from error
    return current_stat


def _service_can_read(path: Path) -> bool:
    try:
        return os.access(path, os.R_OK, effective_ids=True)
    except TypeError:
        return os.access(path, os.R_OK)


def validate_remote_source_files(source: RemoteSource) -> None:
    """Validate credential paths at the execution boundary without reading their contents."""
    key_path = _validated_config_path(source.key_path, "SSH private key")
    known_hosts_path = _validated_config_path(source.known_hosts_path, "known-hosts file")
    key_stat = _lstat_without_symlink_components(key_path, "SSH private key")
    known_hosts_stat = _lstat_without_symlink_components(known_hosts_path, "known-hosts file")

    if not stat.S_ISREG(key_stat.st_mode):
        raise ValueError("SSH private key must be a regular file")
    if key_stat.st_uid != os.geteuid():
        raise ValueError("SSH private key must be owned by the service user")
    if key_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ValueError("SSH private key must not grant group or other permissions")
    if not key_stat.st_mode & stat.S_IRUSR or not _service_can_read(key_path):
        raise ValueError("SSH private key must be readable by the service user")

    if not stat.S_ISREG(known_hosts_stat.st_mode):
        raise ValueError("known-hosts file must be a regular file")
    if known_hosts_stat.st_uid not in {0, os.geteuid()}:
        raise ValueError("known-hosts file must be owned by root or the service user")
    if known_hosts_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("known-hosts file must not be group or other writable")
    if not _service_can_read(known_hosts_path):
        raise ValueError("known-hosts file must be readable by the service user")


def _absolute_lexical_path(value: Path, field_name: str) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(value)))
    except TypeError as error:
        raise ValueError(f"{field_name} must be a filesystem path") from error


def _is_contained(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _validate_owned_protected_directory(path: Path, field_name: str) -> None:
    path_stat = _lstat_without_symlink_components(
        path,
        field_name,
        missing_message=f"{field_name} must be an existing directory",
    )
    if not stat.S_ISDIR(path_stat.st_mode):
        raise ValueError(f"{field_name} must be an existing directory")
    if path_stat.st_uid != os.geteuid():
        raise ValueError(f"{field_name} must be owned by the service user")
    if path_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(f"{field_name} must not be group or other writable")


def validate_staging_path(staging: Path, trusted_staging_root: Path) -> Path:
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

    _validate_owned_protected_directory(lexical_root, "trusted staging root")
    current = lexical_root
    for component in lexical_staging.relative_to(lexical_root).parts:
        current /= component
        field_name = "staging destination" if current == lexical_staging else "staging ancestor"
        _validate_owned_protected_directory(current, field_name)

    resolved_root = lexical_root.resolve(strict=True)
    resolved_staging = lexical_staging.resolve(strict=True)
    if not _is_contained(resolved_staging, resolved_root):
        raise ValueError("resolved staging destination must be within trusted staging root")
    return resolved_staging


def _validated_staging_destination(staging: Path) -> Path:
    try:
        path = Path(staging)
    except TypeError as error:
        raise ValueError("staging destination must be a filesystem path") from error

    if path.is_symlink():
        raise ValueError("staging destination must not be a symlink")
    if path.exists():
        if not path.is_dir():
            raise ValueError("staging destination must be a directory")
    else:
        parent = path.parent
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError("a new staging destination must have a safe existing parent")

    resolved = path.resolve()
    if not resolved.is_absolute():
        raise ValueError("staging destination must resolve to an absolute path")
    if resolved == Path(resolved.anchor):
        raise ValueError("filesystem root cannot be used as a staging destination")
    return resolved


def build_rsync_argv(
    source: RemoteSource,
    remote_root: str,
    remote_relative_path: str,
    staging: Path,
    *,
    trusted_staging_root: Path | None = None,
) -> list[str]:
    """Build an rrsync-compatible argv without executing rsync or a shell.

    Standard rsync 3.2.7 ``rrsync -ro`` rejects the remote ``-s`` generated by
    ``--protect-args``. Remote operands therefore use the conservative exact-segment policy
    enforced above. Task 9 must pass ``trusted_staging_root`` after creating the staging
    directory, validate credentials, and revalidate staging immediately before ``Popen``.
    """
    root = _validate_remote_root(source, remote_root)
    relative_path = normalize_remote_relative_path(remote_relative_path)
    username = _validate_username(source.username)
    host, is_ipv6 = _validated_host(source.host)
    if trusted_staging_root is None:
        destination = _validated_staging_destination(staging)
    else:
        destination = validate_staging_path(staging, trusted_staging_root)

    rendered_host = f"[{host}]" if is_ipv6 else host
    remote_path = str(PurePosixPath(root) / relative_path)
    remote_spec = f"{username}@{rendered_host}:{remote_path}/"
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
