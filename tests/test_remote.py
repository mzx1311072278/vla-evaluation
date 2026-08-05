import os
import shlex
from dataclasses import replace

import pytest

from vla_eval.config import RemoteSource
from vla_eval.remote import (
    build_rsync_argv,
    normalize_remote_relative_path,
    ssh_command_from_trusted_config,
    validate_remote_source_files,
    validate_rsync_version_output,
    validate_staging_path,
)


@pytest.fixture
def remote_source(tmp_path):
    return RemoteSource(
        name="lab-a",
        host="10.0.0.8",
        port=22,
        username="eval-read",
        key_path=tmp_path / "key",
        known_hosts_path=tmp_path / "known_hosts",
        roots=("/data/rollouts",),
    )


@pytest.mark.parametrize("value", ["../secret", "/etc", "run\n--delete", "run\x00bad"])
def test_normalize_remote_path_rejects_unsafe_values(value):
    with pytest.raises(ValueError):
        normalize_remote_relative_path(value)


def test_build_rsync_argv_never_uses_shell(tmp_path):
    remote_source = RemoteSource(
        name="lab-a",
        host="10.0.0.8",
        port=22,
        username="eval-read",
        key_path=tmp_path / "key",
        known_hosts_path=tmp_path / "known_hosts",
        roots=("/data/rollouts",),
    )
    argv = build_rsync_argv(remote_source, "/data/rollouts", "run-1", tmp_path)
    assert argv[0] == "rsync"
    assert "--delete" not in argv
    assert argv[-1] == f"{tmp_path}/"
    assert "eval-read@10.0.0.8:/data/rollouts/run-1/" in argv


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        ".",
        "..",
        "run/../secret",
        "run/./frame",
        "run//frame",
        "run\\frame",
        "-delete",
        "run/--delete",
        "run/",
        "run\x7fbad",
        "run\x85bad",
        "run\ud800bad",
        "run\u2028bad",
        "run\u2029bad",
        "run\u202ebad",
        "run\u2066bad",
    ],
)
def test_normalize_remote_path_rejects_ambiguous_or_dangerous_segments(value):
    with pytest.raises(ValueError):
        normalize_remote_relative_path(value)


@pytest.mark.parametrize(
    "value", ["run-1/\u573a\u666f.01", ".hidden/frame.v2", "\u6570\u636e/\u56de\u653e_01"]
)
def test_normalize_remote_path_preserves_safe_names(value):
    assert normalize_remote_relative_path(value) == value


def test_normalize_remote_path_rejects_non_nfc_spelling():
    with pytest.raises(ValueError, match="NFC-normalized"):
        normalize_remote_relative_path("cafe\u0301/run")


def test_normalize_remote_path_preserves_nfc_unicode_name():
    value = "caf\u00e9/\u573a\u666f.01"

    assert normalize_remote_relative_path(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "run 1",
        "run/*",
        "run/?",
        "run/[abc]",
        "run/frame]",
        "run/$HOME",
        "run/frame;touch",
        "run/frame'quoted",
    ],
)
def test_normalize_remote_path_rejects_transport_unsafe_characters(value):
    with pytest.raises(ValueError, match="Unicode letters/numbers"):
        normalize_remote_relative_path(value)


def test_build_rsync_argv_has_a_fixed_option_allowlist(remote_source, tmp_path):
    argv = build_rsync_argv(remote_source, "/data/rollouts", "run-1/\u573a\u666f.01", tmp_path)

    assert argv[:7] == [
        "rsync",
        "-a",
        "--partial",
        "--append-verify",
        "--info=progress2",
        "--out-format=%i|%l|%n",
        "-e",
    ]
    assert argv[8:] == [
        "--",
        "eval-read@10.0.0.8:/data/rollouts/run-1/\u573a\u666f.01/",
        f"{tmp_path.resolve()}/",
    ]
    assert not {"-s", "--protect-args", "--delete", "--rsync-path"}.intersection(argv)
    assert all("shell=True" not in argument for argument in argv)


def test_build_rsync_argv_is_structurally_compatible_with_standard_rrsync(remote_source, tmp_path):
    argv = build_rsync_argv(remote_source, "/data/rollouts", "run-1/frame_01", tmp_path)

    assert "-s" not in argv
    assert "--protect-args" not in argv
    assert argv[-2] == "eval-read@10.0.0.8:/data/rollouts/run-1/frame_01/"
    assert not any(character in argv[-2] for character in "*?[]")


def test_build_rsync_argv_rejects_option_injection(remote_source, tmp_path):
    with pytest.raises(ValueError):
        build_rsync_argv(remote_source, "/data/rollouts", "run/--delete", tmp_path)


@pytest.mark.parametrize("remote_root", ["/data", "/data/rollouts-other"])
def test_build_rsync_argv_rejects_root_mismatch_and_prefix_confusion(
    remote_source, tmp_path, remote_root
):
    with pytest.raises(ValueError):
        build_rsync_argv(remote_source, remote_root, "run-1", tmp_path)


@pytest.mark.parametrize("root", ["data/rollouts", "//data/rollouts", "/data//rollouts", "/data/."])
def test_build_rsync_argv_defends_against_noncanonical_config_roots(remote_source, tmp_path, root):
    source = replace(remote_source, roots=(root,))

    with pytest.raises(ValueError):
        build_rsync_argv(source, root, "run-1", tmp_path)


@pytest.mark.parametrize(
    "root",
    ["/data/roll outs", "/data/roll*outs", "/data/roll;outs", "/data/cafe\u0301"],
)
def test_build_rsync_argv_rejects_transport_unsafe_config_roots(remote_source, tmp_path, root):
    source = replace(remote_source, roots=(root,))

    with pytest.raises(ValueError):
        build_rsync_argv(source, root, "run-1", tmp_path)


def test_build_rsync_argv_brackets_ipv6_host(remote_source, tmp_path):
    source = replace(remote_source, host="2001:db8::8")

    argv = build_rsync_argv(source, "/data/rollouts", "run-1", tmp_path)

    assert "eval-read@[2001:db8::8]:/data/rollouts/run-1/" in argv


def test_ssh_command_quotes_administrator_paths_and_enforces_strict_options(tmp_path):
    source = RemoteSource(
        name="lab-a",
        host="rollouts.example.com",
        port=2202,
        username="eval-read",
        key_path=tmp_path / "key $(touch should-not-run)",
        known_hosts_path=tmp_path / "known hosts;still-one-token",
        roots=("/data/rollouts",),
    )

    tokens = shlex.split(ssh_command_from_trusted_config(source))

    assert tokens == [
        "ssh",
        "-F",
        "none",
        "-T",
        "-p",
        "2202",
        "-i",
        str(source.key_path),
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
        f"UserKnownHostsFile={source.known_hosts_path}",
        "-o",
        "ConnectTimeout=10",
    ]
    assert "StrictHostKeyChecking=no" not in tokens


def test_ssh_command_accepts_maximum_length_rooted_dns_name(remote_source):
    host = ".".join(["a" * 63, "b" * 63, "c" * 63, "d" * 61]) + "."
    source = replace(remote_source, host=host)

    assert len(host) == 254
    assert shlex.split(ssh_command_from_trusted_config(source))[0] == "ssh"


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("rsync  version 3.2.7  protocol version 31\n", (3, 2, 7)),
        ("rsync  version 3.3.0  protocol version 31\nCapabilities:\n", (3, 3, 0)),
    ],
)
def test_validate_rsync_version_output_accepts_supported_gnu_rsync(output, expected):
    assert validate_rsync_version_output(output) == expected


@pytest.mark.parametrize(
    "output",
    [
        "rsync  version 3.2.6  protocol version 31\n",
        "rsync  version 3.2.7pre1  protocol version 31\n",
        "openrsync: protocol version 29\n",
        "not rsync output",
    ],
)
def test_validate_rsync_version_output_rejects_old_or_unrecognized_versions(output):
    with pytest.raises(ValueError):
        validate_rsync_version_output(output)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("username", "-oProxyCommand=bad"),
        ("username", "eval@admin"),
        ("username", "eval user"),
        ("host", "bad host"),
        ("host", "host;command"),
        ("host", "999.999.999.999"),
        ("host", "[2001:db8::8]"),
        ("port", 0),
        ("port", 65536),
        ("port", True),
    ],
)
def test_build_rsync_argv_rejects_invalid_endpoint_config(remote_source, tmp_path, field, value):
    source = replace(remote_source, **{field: value})

    with pytest.raises(ValueError):
        build_rsync_argv(source, "/data/rollouts", "run-1", tmp_path)


def test_ssh_command_rejects_relative_config_file_paths(remote_source):
    source = replace(remote_source, key_path=remote_source.key_path.name)

    with pytest.raises(ValueError):
        ssh_command_from_trusted_config(source)


def test_validate_remote_source_files_accepts_regular_files(remote_source):
    remote_source.key_path.write_text("private", encoding="utf-8")
    remote_source.key_path.chmod(0o600)
    remote_source.known_hosts_path.write_text("host key", encoding="utf-8")
    remote_source.known_hosts_path.chmod(0o644)

    validate_remote_source_files(remote_source)


@pytest.mark.parametrize("field", ["key_path", "known_hosts_path"])
def test_validate_remote_source_files_rejects_missing_files(remote_source, field):
    other_field = "known_hosts_path" if field == "key_path" else "key_path"
    other_path = getattr(remote_source, other_field)
    other_path.write_text("configured", encoding="utf-8")
    other_path.chmod(0o644 if other_field == "known_hosts_path" else 0o600)

    with pytest.raises(ValueError):
        validate_remote_source_files(remote_source)


def test_validate_remote_source_files_rejects_symlinks(remote_source, tmp_path):
    real_key = tmp_path / "real-key"
    real_key.write_text("private", encoding="utf-8")
    real_key.chmod(0o600)
    remote_source.key_path.symlink_to(real_key)
    remote_source.known_hosts_path.write_text("host key", encoding="utf-8")
    remote_source.known_hosts_path.chmod(0o644)

    with pytest.raises(ValueError):
        validate_remote_source_files(remote_source)


def test_validate_remote_source_files_rejects_symlink_ancestors(remote_source, tmp_path):
    credential_directory = tmp_path / "credentials"
    credential_directory.mkdir()
    key = credential_directory / "key"
    key.write_text("private", encoding="utf-8")
    key.chmod(0o600)
    known_hosts = credential_directory / "known_hosts"
    known_hosts.write_text("host key", encoding="utf-8")
    known_hosts.chmod(0o644)
    linked_directory = tmp_path / "linked-credentials"
    linked_directory.symlink_to(credential_directory, target_is_directory=True)
    source = replace(
        remote_source,
        key_path=linked_directory / "key",
        known_hosts_path=linked_directory / "known_hosts",
    )

    with pytest.raises(ValueError, match="symlink"):
        validate_remote_source_files(source)


def test_validate_remote_source_files_rejects_permissive_private_key(remote_source):
    remote_source.key_path.write_text("private", encoding="utf-8")
    remote_source.key_path.chmod(0o640)
    remote_source.known_hosts_path.write_text("host key", encoding="utf-8")
    remote_source.known_hosts_path.chmod(0o644)

    with pytest.raises(ValueError, match="group or other permissions"):
        validate_remote_source_files(remote_source)


def test_validate_remote_source_files_rejects_private_key_owned_by_another_uid(
    remote_source, monkeypatch
):
    remote_source.key_path.write_text("private", encoding="utf-8")
    remote_source.key_path.chmod(0o600)
    remote_source.known_hosts_path.write_text("host key", encoding="utf-8")
    remote_source.known_hosts_path.chmod(0o644)
    current_euid = os.geteuid()
    monkeypatch.setattr("vla_eval.remote.os.geteuid", lambda: current_euid + 1)

    with pytest.raises(ValueError, match="owned by the service user"):
        validate_remote_source_files(remote_source)


def test_validate_remote_source_files_rejects_group_writable_known_hosts(remote_source):
    remote_source.key_path.write_text("private", encoding="utf-8")
    remote_source.key_path.chmod(0o600)
    remote_source.known_hosts_path.write_text("host key", encoding="utf-8")
    remote_source.known_hosts_path.chmod(0o664)

    with pytest.raises(ValueError, match="group or other writable"):
        validate_remote_source_files(remote_source)


@pytest.mark.parametrize(("field", "mode"), [("key_path", 0o000), ("known_hosts_path", 0o200)])
def test_validate_remote_source_files_rejects_unreadable_files(remote_source, field, mode):
    remote_source.key_path.write_text("private", encoding="utf-8")
    remote_source.key_path.chmod(0o600)
    remote_source.known_hosts_path.write_text("host key", encoding="utf-8")
    remote_source.known_hosts_path.chmod(0o644)
    getattr(remote_source, field).chmod(mode)

    with pytest.raises(ValueError, match="readable"):
        validate_remote_source_files(remote_source)


def test_build_rsync_argv_accepts_missing_staging_with_existing_parent(remote_source, tmp_path):
    staging = tmp_path / "new-staging"

    argv = build_rsync_argv(remote_source, "/data/rollouts", "run-1", staging)

    assert argv[-1] == f"{staging.resolve()}/"


def test_build_rsync_argv_rejects_staging_with_missing_parent(remote_source, tmp_path):
    staging = tmp_path / "missing" / "staging"

    with pytest.raises(ValueError):
        build_rsync_argv(remote_source, "/data/rollouts", "run-1", staging)


def test_build_rsync_argv_rejects_existing_file_staging(remote_source, tmp_path):
    staging = tmp_path / "staging-file"
    staging.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError):
        build_rsync_argv(remote_source, "/data/rollouts", "run-1", staging)


def test_build_rsync_argv_rejects_symlink_staging(remote_source, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    staging = tmp_path / "staging-link"
    staging.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError):
        build_rsync_argv(remote_source, "/data/rollouts", "run-1", staging)


def test_build_rsync_argv_rejects_filesystem_root_staging(remote_source):
    with pytest.raises(ValueError):
        build_rsync_argv(remote_source, "/data/rollouts", "run-1", "/")


def test_validate_staging_path_accepts_owned_protected_directory(tmp_path):
    trusted_root = tmp_path / "staging"
    trusted_root.mkdir(mode=0o700)
    trusted_root.chmod(0o700)
    destination = trusted_root / "attempt-1"
    destination.mkdir(mode=0o700)
    destination.chmod(0o700)

    assert validate_staging_path(destination, trusted_root) == destination.resolve()


def test_build_rsync_argv_can_enforce_trusted_staging_root(remote_source, tmp_path):
    trusted_root = tmp_path / "staging"
    trusted_root.mkdir(mode=0o700)
    trusted_root.chmod(0o700)
    destination = trusted_root / "attempt-1"
    destination.mkdir(mode=0o700)
    destination.chmod(0o700)

    argv = build_rsync_argv(
        remote_source,
        "/data/rollouts",
        "run-1",
        destination,
        trusted_staging_root=trusted_root,
    )

    assert argv[-1] == f"{destination.resolve()}/"


def test_validate_staging_path_rejects_lexical_escape(tmp_path):
    trusted_root = tmp_path / "staging"
    trusted_root.mkdir(mode=0o700)
    trusted_root.chmod(0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    outside.chmod(0o700)

    with pytest.raises(ValueError, match="within trusted staging root"):
        validate_staging_path(trusted_root / ".." / "outside", trusted_root)


def test_validate_staging_path_rejects_symlink_components(tmp_path):
    trusted_root = tmp_path / "staging"
    trusted_root.mkdir(mode=0o700)
    trusted_root.chmod(0o700)
    actual = trusted_root / "actual"
    actual.mkdir(mode=0o700)
    actual.chmod(0o700)
    destination = actual / "attempt-1"
    destination.mkdir(mode=0o700)
    destination.chmod(0o700)
    linked = trusted_root / "linked"
    linked.symlink_to(actual, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        validate_staging_path(linked / "attempt-1", trusted_root)


def test_validate_staging_path_rejects_missing_destination(tmp_path):
    trusted_root = tmp_path / "staging"
    trusted_root.mkdir(mode=0o700)
    trusted_root.chmod(0o700)

    with pytest.raises(ValueError, match="existing directory"):
        validate_staging_path(trusted_root / "missing", trusted_root)


@pytest.mark.parametrize("unsafe_component", ["root", "intermediate", "parent", "destination"])
def test_validate_staging_path_rejects_writable_directories(tmp_path, unsafe_component):
    trusted_root = tmp_path / "staging"
    trusted_root.mkdir(mode=0o700)
    trusted_root.chmod(0o700)
    intermediate = trusted_root / "intermediate"
    intermediate.mkdir(mode=0o700)
    intermediate.chmod(0o700)
    parent = intermediate / "parent"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    destination = parent / "attempt-1"
    destination.mkdir(mode=0o700)
    destination.chmod(0o700)
    selected = {
        "root": trusted_root,
        "intermediate": intermediate,
        "parent": parent,
        "destination": destination,
    }[unsafe_component]
    selected.chmod(0o770)

    with pytest.raises(ValueError, match="group or other writable"):
        validate_staging_path(destination, trusted_root)


def test_validate_staging_path_rejects_wrong_owner(tmp_path, monkeypatch):
    trusted_root = tmp_path / "staging"
    trusted_root.mkdir(mode=0o700)
    trusted_root.chmod(0o700)
    destination = trusted_root / "attempt-1"
    destination.mkdir(mode=0o700)
    destination.chmod(0o700)
    current_euid = os.geteuid()
    monkeypatch.setattr("vla_eval.remote.os.geteuid", lambda: current_euid + 1)

    with pytest.raises(ValueError, match="owned by the service user"):
        validate_staging_path(destination, trusted_root)


def test_validate_staging_path_rejects_filesystem_root():
    with pytest.raises(ValueError, match="filesystem root"):
        validate_staging_path("/", "/")
