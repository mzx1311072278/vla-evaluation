# Shared Storage Trust Boundary Design

## Summary

The evaluation service currently validates every ancestor of its trusted staging and inbox directories, from the filesystem root downward. This is appropriate on a dedicated host, but it rejects managed shared storage layouts such as `/czj/code/vla-evaluation/data` when `/czj` or `/czj/code` is intentionally writable by multiple platform users.

Add an explicit, opt-in storage trust mode that treats `data_root` as the first application-controlled directory. The default remains the existing strict behavior. The relaxed mode accepts the organizational guarantee that shared ancestors are managed outside the application, while retaining ownership, permission, symlink, containment, and atomic-publication checks at and below `data_root`.

## Goals

- Allow imports when `data_root` is under managed shared ancestors that are group- or other-writable.
- Preserve the current strict policy for every existing configuration.
- Require an explicit configuration choice before accepting the shared-ancestor risk.
- Keep all checks at and below `data_root`, including owner, mode, symlink, path-containment, and no-replace publication checks.
- Keep local-source and remote-source transport validation unchanged.
- Explain the selected trust mode in startup logs and deployment documentation.

## Non-Goals

- Do not make a group- or other-writable `data_root` acceptable.
- Do not protect against a platform user who can replace the entire `data_root` through a writable ancestor. That risk is accepted by the deployment operator in boundary mode.
- Do not modify filesystem permissions automatically.
- Do not change the configured local dataset roots or permit paths outside them.
- Do not add directory browsing or automatic dataset discovery to the Web form.
- Do not change the database schema.

## Configuration

Add an optional top-level application setting:

```yaml
storage_trust_mode: data_root_boundary
```

Supported values:

| Value | Behavior |
| --- | --- |
| `strict` | Default. Validate every path component from `/` through each trusted destination, matching current behavior. |
| `data_root_boundary` | Treat `data_root` as the first trusted component. Ignore owner and writable-mode checks above it, but validate `data_root` and every descendant used by imports. |

Invalid, blank, non-string, or unknown values fail configuration loading with a field-specific error. `AppConfig` stores the normalized mode and defaults it to `strict` so existing direct constructors and deployments remain compatible.

For the current server, the private configuration becomes:

```yaml
data_root: /czj/code/vla-evaluation/data
storage_trust_mode: data_root_boundary
```

The operator must keep these application-controlled directories owned by the service user and not writable by group or others:

```text
/czj/code/vla-evaluation/data
/czj/code/vla-evaluation/data/staging
/czj/code/vla-evaluation/data/inbox
```

## Trust Model

### Strict Mode

The validation chain remains:

```text
/ -> ... -> data_root -> staging|inbox -> task destination
```

Every directory must:

- be a real directory rather than a symbolic link;
- be owned by root or the effective service user;
- not be writable by group or others;
- provide the access required by the operation.

### Data-Root Boundary Mode

The validation chain becomes:

```text
data_root -> staging|inbox -> task destination
```

Components above `data_root` are outside the application's enforcement boundary. The application still requires that:

- `data_root` is an absolute, existing, non-symlink directory;
- `data_root` is owned by root or the effective service user;
- `data_root` is not writable by group or others;
- staging and inbox roots are strict descendants of `data_root`;
- every checked descendant is a real, service-owned directory without group/other write bits;
- resolved destinations remain contained inside the configured roots;
- staging and inbox remain on the same filesystem for atomic publication;
- publication never replaces an existing dataset;
- cancellation, fingerprint, and interrupted-publication checks continue to run.

The accepted residual risk is explicit: a user who can write the shared parent may be able to rename or replace the entire project directory. Organizational controls and platform isolation, rather than this application, cover that risk.

## Code Structure

### Configuration Layer

`vla_eval/config.py` parses `storage_trust_mode`, exposes constants or a constrained type for the two supported values, and adds the setting to `AppConfig` with the strict default.

### Task Boundary

`vla_eval/tasks.py` derives the import trust boundary from the immutable runtime configuration:

- strict mode passes no shortened boundary;
- boundary mode passes `config.data_root` as the minimum checked ancestor.

The persisted import job does not store this deployment setting. A worker always applies its current server configuration, consistent with the existing handling of trusted staging, inbox, credentials, and source roots.

### Filesystem Validation

`vla_eval/import_jobs.py` extends protected-directory validation with an optional minimum ancestor. A single helper builds the checked component chain and enforces these invariants:

1. The minimum ancestor is absolute and normalized.
2. The checked path equals the minimum ancestor or is below it.
3. The minimum ancestor itself is included in validation.
4. No component below the boundary is skipped.

Production staging creation, target-parent verification, completed-import loading, interrupted-import reconciliation, and final published-target validation all receive the same boundary. Injected test mode remains strict unless a test explicitly supplies a boundary through the production specification.

The local source resolver in `vla_eval/local.py` is unchanged. It continues to validate the configured source path from the filesystem root because source directories are read-only inputs and are not declared application-controlled by `storage_trust_mode`.

## Data Flow

```text
app.yaml
  -> load_config(storage_trust_mode)
  -> TaskRuntime(AppConfig)
  -> run_import_task
  -> ImportSpec(trusted data-root boundary or strict sentinel)
  -> execute_import
  -> staging/inbox validators use one consistent boundary
  -> rsync, preflight, fingerprint, atomic publication
```

No Web form field is added. The trust choice remains an operator-controlled server configuration and cannot be changed by a browser user or an import request.

## Logging and Errors

- Strict mode produces no new warning.
- Starting Web or worker runtime in `data_root_boundary` mode logs one warning containing the non-secret `data_root` path and a concise statement that ancestor permission checks are delegated to the platform.
- Boundary violations use specific errors, such as `trusted staging root must be at or below storage trust boundary`.
- Unsafe permissions or ownership at/below `data_root` retain the current errors.
- User-facing import failures remain sanitized; detailed filesystem errors stay in worker logs.

## Tests

### Configuration Tests

- Missing setting defaults to `strict`.
- Both supported values parse successfully.
- Blank, non-string, and unknown values are rejected.
- `AppConfig` representations do not expose secrets and include no new secret material.

### Filesystem Unit Tests

- Strict mode still rejects a trusted root below a group/other-writable ancestor.
- Boundary mode accepts writable ancestors above a protected `data_root`.
- Boundary mode rejects a writable, foreign-owned, symlinked, or inaccessible `data_root`.
- Boundary mode rejects writable, foreign-owned, or symlinked staging/inbox descendants.
- A checked path outside or above the declared boundary is rejected.
- Published-target and interrupted-import validators apply the same boundary as initial staging creation.
- Path traversal, source/target overlap, cross-filesystem publication, and existing-target protections remain covered.

### Task and Integration Tests

- `run_import_task` forwards no boundary in strict mode.
- `run_import_task` forwards `data_root` in boundary mode.
- A local import succeeds under a writable synthetic shared ancestor only in boundary mode.
- A boundary-mode import still fails when `data_root/staging` or `data_root/inbox` is writable by group/others.
- Existing strict-mode test suites pass unchanged.

## Deployment Procedure

After the change reaches the server:

1. Pull the new branch or merged `main` into `/czj/code/vla-evaluation/app`.
2. Add `storage_trust_mode: data_root_boundary` to the private `app.yaml`.
3. Verify `data`, `staging`, and `inbox` are service-owned and mode `700`.
4. Restart Web, transfer worker, and evaluation worker so all processes load the same configuration.
5. Run `python -m vla_eval.cli smoke`.
6. Submit a new local import and verify it reaches transfer, preflight, and READY.

Rollback removes the setting or changes it to `strict`, then restarts the processes. No database or artifact migration is required.
