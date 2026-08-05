import shlex
from dataclasses import replace

import pytest

from vla_eval.config import RemoteSource
from vla_eval.remote import (
    build_rsync_argv,
    normalize_remote_relative_path,
    ssh_command_from_trusted_config,
    validate_remote_source_files,
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
    "value", ["run 1/\u573a\u666f.01", ".hidden/frame.v2", "\u6570\u636e/\u56de\u653e 01"]
)
def test_normalize_remote_path_preserves_safe_names(value):
    assert normalize_remote_relative_path(value) == value


def test_normalize_remote_path_rejects_non_nfc_spelling():
    with pytest.raises(ValueError, match="NFC-normalized"):
        normalize_remote_relative_path("cafe\u0301/run")


def test_normalize_remote_path_preserves_nfc_unicode_name():
    value = "caf\u00e9/\u573a\u666f.01"

    assert normalize_remote_relative_path(value) == value


def test_build_rsync_argv_has_a_fixed_option_allowlist(remote_source, tmp_path):
    argv = build_rsync_argv(remote_source, "/data/rollouts", "run 1/\u573a\u666f", tmp_path)

    assert argv[:8] == [
        "rsync",
        "-a",
        "--partial",
        "--append-verify",
        "--protect-args",
        "--info=progress2",
        "--out-format=%i|%l|%n",
        "-e",
    ]
    assert argv[9:] == [
        "--",
        "eval-read@10.0.0.8:/data/rollouts/run 1/\u573a\u666f/",
        f"{tmp_path.resolve()}/",
    ]
    assert not {"--delete", "--rsync-path"}.intersection(argv)
    assert all("shell=True" not in argument for argument in argv)


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
        "-p",
        "2202",
        "-i",
        str(source.key_path),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={source.known_hosts_path}",
        "-o",
        "ConnectTimeout=10",
    ]
    assert "StrictHostKeyChecking=no" not in tokens


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
    remote_source.known_hosts_path.write_text("host key", encoding="utf-8")

    validate_remote_source_files(remote_source)


@pytest.mark.parametrize("field", ["key_path", "known_hosts_path"])
def test_validate_remote_source_files_rejects_missing_files(remote_source, field):
    other_field = "known_hosts_path" if field == "key_path" else "key_path"
    getattr(remote_source, other_field).write_text("configured", encoding="utf-8")

    with pytest.raises(ValueError):
        validate_remote_source_files(remote_source)


def test_validate_remote_source_files_rejects_symlinks(remote_source, tmp_path):
    real_key = tmp_path / "real-key"
    real_key.write_text("private", encoding="utf-8")
    remote_source.key_path.symlink_to(real_key)
    remote_source.known_hosts_path.write_text("host key", encoding="utf-8")

    with pytest.raises(ValueError):
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
