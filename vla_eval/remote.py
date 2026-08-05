import ipaddress
import re
import shlex
import unicodedata
from pathlib import Path, PurePosixPath

from vla_eval.config import RemoteSource

_SSH_USERNAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_DNS_LABEL_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z")
_SSH_CONNECT_TIMEOUT_SECONDS = 10


def _contains_control_or_format_character(value: str) -> bool:
    return any(unicodedata.category(character) in {"Cc", "Cf"} for character in value)


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

    raw_segments = value.split("/")
    if any(not segment or not segment.strip() for segment in raw_segments):
        raise ValueError("remote path must not contain empty or whitespace-only segments")
    if any(segment in {".", ".."} for segment in raw_segments):
        raise ValueError("remote path must not contain '.' or '..' segments")
    if any(segment.startswith("-") for segment in raw_segments):
        raise ValueError("remote path segments must not start with '-'")

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
    if not dns_name or len(host) > 253:
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
            "-p",
            str(port),
            "-i",
            str(key_path),
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={known_hosts_path}",
            "-o",
            f"ConnectTimeout={_SSH_CONNECT_TIMEOUT_SECONDS}",
        ]
    )


def validate_remote_source_files(source: RemoteSource) -> None:
    """Validate credential paths at the execution boundary without reading their contents."""
    configured_paths = (
        ("SSH key path", _validated_config_path(source.key_path, "SSH key path")),
        (
            "known-hosts path",
            _validated_config_path(source.known_hosts_path, "known-hosts path"),
        ),
    )
    for field_name, path in configured_paths:
        if path.is_symlink():
            raise ValueError(f"{field_name} must not be a symlink")
        if not path.is_file():
            raise ValueError(f"{field_name} must be an existing regular file")


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
) -> list[str]:
    """Build a fixed rsync argv without executing a shell or accessing credentials."""
    root = _validate_remote_root(source, remote_root)
    relative_path = normalize_remote_relative_path(remote_relative_path)
    username = _validate_username(source.username)
    host, is_ipv6 = _validated_host(source.host)
    destination = _validated_staging_destination(staging)

    rendered_host = f"[{host}]" if is_ipv6 else host
    remote_path = str(PurePosixPath(root) / relative_path)
    remote_spec = f"{username}@{rendered_host}:{remote_path}/"
    return [
        "rsync",
        "-a",
        "--partial",
        "--append-verify",
        "--protect-args",
        "--info=progress2",
        "--out-format=%i|%l|%n",
        "-e",
        ssh_command_from_trusted_config(source),
        "--",
        remote_spec,
        f"{destination}/",
    ]
