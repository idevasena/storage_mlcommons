"""Regression tests for issue #303.

MPIClusterCollector used to write the helper collector script into a
``tempfile.TemporaryDirectory()`` on the launch host only, then invoke
``mpirun`` with that absolute path. On clusters with node-local ``/tmp``
the remote ranks could not find the script and ``mpirun`` aborted with
``[Errno 2] No such file or directory``.

These tests cover the three code paths introduced by the fix:

* default "stage-and-run" path — SCPs the script to each remote host
  before ``mpirun``, and cleans up afterwards;
* ``shared_tmp_dir`` opt-in — skips all SSH staging;
* partial staging failure — raises a descriptive error naming the bad host.

Also covers the X11 env injection that silences the
``Authorization required, but no authorization protocol specified``
noise reported in the original issue.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import List
from unittest import mock

import pytest

from mlpstorage_py import cluster_collector as cc
from mlpstorage_py.config import MPIRUN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_run(output_path_getter, write_output: bool = True):
    """Build a fake ``subprocess.run`` that fakes ``mpirun`` by writing the
    expected rank-0 output JSON to disk, and records every call.

    Parameters
    ----------
    output_path_getter: callable returning the expected cluster_info.json path
        at the time ``mpirun`` is invoked (resolved lazily so the per-run UUID
        path created inside ``collect()`` is honored).
    write_output: when False, ``mpirun`` "succeeds" but produces no output
        file — simulating a cluster where staging succeeded but mpirun itself
        failed to run the script on every rank.
    """
    calls: List[dict] = []

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list):
            argv = cmd
            kind = argv[0]
        else:
            argv = cmd.split()
            kind = "mpirun" if "mpirun" in cmd or "mpiexec" in cmd else argv[0]

        calls.append({
            "argv": argv,
            "kind": kind,
            "env": kwargs.get("env"),
            "shell": kwargs.get("shell", False),
        })

        # Successful mpirun: write the aggregated JSON rank 0 would produce.
        if kind in ("mpirun", "mpiexec") and write_output:
            output_path = output_path_getter()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "host-a": {"hostname": "host-a", "total_memory_kb": 1024},
                        "host-b": {"hostname": "host-b", "total_memory_kb": 2048},
                    },
                    f,
                )

        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    return fake_run, calls


def _collector(tmp_path, hosts, **kwargs):
    logger = logging.getLogger("test.cluster_collector")
    logger.setLevel(logging.DEBUG)
    return cc.MPIClusterCollector(
        hosts=hosts,
        mpi_bin=MPIRUN,
        logger=logger,
        timeout_seconds=30,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMPIStaging:
    """Default path: the collector must SCP the script to remote hosts."""

    def test_stages_script_on_each_remote_host(self, tmp_path, monkeypatch):
        collector = _collector(tmp_path, hosts=["host-a:1", "host-b:1"])

        # The tempdir is created inside collect() under tempfile.gettempdir().
        # Redirect it to tmp_path so the test is hermetic.
        monkeypatch.setattr(cc.tempfile, "gettempdir", lambda: str(tmp_path))

        # Make _is_localhost return True only for 'host-a' so host-b is staged.
        monkeypatch.setattr(cc, "_is_localhost",
                            lambda h: h in ("host-a", "localhost", "127.0.0.1"))

        output_holder = {}

        def capture_output_path(collector_self=collector, holder=output_holder):
            return holder["output_path"]

        fake_run, calls = _make_fake_run(capture_output_path)

        # Wrap subprocess.run so we can snapshot the output path the collector
        # picks on this invocation (needed by the fake mpirun above).
        original_makedirs = os.makedirs

        def spy_makedirs(path, *a, **kw):
            # The first makedirs in collect() is on working_dir.
            if "output_path" not in output_holder and str(tmp_path) in path:
                output_holder["output_path"] = os.path.join(
                    path, "cluster_info.json"
                )
            return original_makedirs(path, *a, **kw)

        monkeypatch.setattr(cc.os, "makedirs", spy_makedirs)
        monkeypatch.setattr(cc.subprocess, "run", fake_run)

        result = collector.collect()

        # mpirun was invoked
        mpi_calls = [c for c in calls if c["kind"] in ("mpirun", "mpiexec")]
        assert len(mpi_calls) == 1, f"expected 1 mpirun call, got {calls}"

        # The script was staged to host-b (only remote host) via ssh+scp
        ssh_calls = [c for c in calls if c["kind"] == "ssh"]
        scp_calls = [c for c in calls if c["kind"] == "scp"]
        assert any("host-b" in " ".join(c["argv"]) for c in ssh_calls), \
            "expected at least one ssh call targeting host-b"
        assert any("host-b" in " ".join(c["argv"]) for c in scp_calls), \
            "expected at least one scp call targeting host-b"

        # Cleanup was performed (rm -rf via ssh)
        rm_calls = [
            c for c in ssh_calls if any("rm -rf" in a for a in c["argv"])
        ]
        assert rm_calls, "expected remote rm -rf cleanup after mpirun"

        # Result shape unchanged from before the fix
        assert "host-a" in result and "host-b" in result

    def test_single_localhost_skips_staging(self, tmp_path, monkeypatch):
        """A localhost-only invocation must not SSH anywhere."""
        collector = _collector(tmp_path, hosts=["127.0.0.1:1"])
        monkeypatch.setattr(cc.tempfile, "gettempdir", lambda: str(tmp_path))

        output_holder = {}
        original_makedirs = os.makedirs

        def spy_makedirs(path, *a, **kw):
            if "output_path" not in output_holder and str(tmp_path) in path:
                output_holder["output_path"] = os.path.join(
                    path, "cluster_info.json"
                )
            return original_makedirs(path, *a, **kw)

        fake_run, calls = _make_fake_run(
            lambda: output_holder["output_path"]
        )

        monkeypatch.setattr(cc.os, "makedirs", spy_makedirs)
        monkeypatch.setattr(cc.subprocess, "run", fake_run)

        collector.collect()

        assert not any(c["kind"] in ("ssh", "scp") for c in calls), \
            "localhost-only run must not invoke ssh or scp"


class TestSharedTmpDir:
    """Opt-in fast path: when shared_tmp_dir is set, no SSH staging at all."""

    def test_shared_tmp_dir_skips_staging(self, tmp_path, monkeypatch):
        shared = tmp_path / "shared_scratch"
        shared.mkdir()

        collector = _collector(
            tmp_path,
            hosts=["host-a:1", "host-b:1"],
            shared_tmp_dir=str(shared),
        )
        monkeypatch.setattr(cc, "_is_localhost", lambda h: h == "host-a")

        output_holder = {}
        original_makedirs = os.makedirs

        def spy_makedirs(path, *a, **kw):
            if "output_path" not in output_holder and str(shared) in path:
                output_holder["output_path"] = os.path.join(
                    path, "cluster_info.json"
                )
            return original_makedirs(path, *a, **kw)

        fake_run, calls = _make_fake_run(
            lambda: output_holder["output_path"]
        )
        monkeypatch.setattr(cc.os, "makedirs", spy_makedirs)
        monkeypatch.setattr(cc.subprocess, "run", fake_run)

        collector.collect()

        # Zero ssh/scp calls when a shared tmpdir is provided
        assert not any(c["kind"] in ("ssh", "scp") for c in calls), \
            f"shared_tmp_dir path must not SSH; got {calls}"

        # The working dir must live under the shared path
        mpi_call = next(
            c for c in calls if c["kind"] in ("mpirun", "mpiexec")
        )
        joined = " ".join(mpi_call["argv"])
        assert str(shared) in joined, \
            f"mpirun command must use shared_tmp_dir path; got: {joined}"


class TestStagingFailure:
    """Staging failure must raise a clear error naming the bad host."""

    def test_stage_failure_raises_with_host_info(self, tmp_path, monkeypatch):
        collector = _collector(tmp_path, hosts=["host-a:1", "bad-host:1"])
        monkeypatch.setattr(cc.tempfile, "gettempdir", lambda: str(tmp_path))
        monkeypatch.setattr(cc, "_is_localhost", lambda h: h == "host-a")

        def fake_run(cmd, *args, **kwargs):
            # Every ssh/scp to bad-host fails
            argv = cmd if isinstance(cmd, list) else cmd.split()
            if argv[0] in ("ssh", "scp") and any(
                "bad-host" in a for a in argv
            ):
                return subprocess.CompletedProcess(
                    argv, 255, stdout="",
                    stderr="ssh: connect to host bad-host port 22: "
                           "Connection refused",
                )
            # mpirun should never be reached in this test
            if argv[0] in ("mpirun", "mpiexec"):
                pytest.fail(
                    "mpirun must not run when staging failed on any host"
                )
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        monkeypatch.setattr(cc.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError) as excinfo:
            collector.collect()

        msg = str(excinfo.value)
        assert "bad-host" in msg, f"error must name the failing host; got: {msg}"
        assert "stage" in msg.lower() or "staging" in msg.lower() \
            or "passwordless ssh" in msg.lower(), \
            f"error must mention staging/SSH; got: {msg}"


class TestX11Silence:
    """The mpirun subprocess must receive an env that disables X11 forwarding."""

    def test_plm_rsh_agent_disables_x11(self, tmp_path, monkeypatch):
        collector = _collector(tmp_path, hosts=["127.0.0.1:1"])
        monkeypatch.setattr(cc.tempfile, "gettempdir", lambda: str(tmp_path))

        output_holder = {}
        original_makedirs = os.makedirs

        def spy_makedirs(path, *a, **kw):
            if "output_path" not in output_holder and str(tmp_path) in path:
                output_holder["output_path"] = os.path.join(
                    path, "cluster_info.json"
                )
            return original_makedirs(path, *a, **kw)

        fake_run, calls = _make_fake_run(
            lambda: output_holder["output_path"]
        )
        monkeypatch.setattr(cc.os, "makedirs", spy_makedirs)
        monkeypatch.setattr(cc.subprocess, "run", fake_run)

        # Ensure the test environment does NOT pre-set PLM_RSH_AGENT, so
        # we are verifying that the collector itself injects it.
        monkeypatch.delenv("PLM_RSH_AGENT", raising=False)

        collector.collect()

        mpi_call = next(
            c for c in calls if c["kind"] in ("mpirun", "mpiexec")
        )
        env = mpi_call["env"]
        assert env is not None, "mpirun must be invoked with a custom env"
        assert "PLM_RSH_AGENT" in env, \
            "PLM_RSH_AGENT must be set to silence X11 warnings"
        assert "ForwardX11=no" in env["PLM_RSH_AGENT"], \
            f"PLM_RSH_AGENT must disable X11 forwarding; got {env['PLM_RSH_AGENT']!r}"

    def test_existing_plm_rsh_agent_is_preserved(self, tmp_path, monkeypatch):
        """If the user has their own PLM_RSH_AGENT, don't clobber it."""
        collector = _collector(tmp_path, hosts=["127.0.0.1:1"])
        monkeypatch.setattr(cc.tempfile, "gettempdir", lambda: str(tmp_path))
        monkeypatch.setenv("PLM_RSH_AGENT", "ssh -i /custom/key")

        output_holder = {}
        original_makedirs = os.makedirs

        def spy_makedirs(path, *a, **kw):
            if "output_path" not in output_holder and str(tmp_path) in path:
                output_holder["output_path"] = os.path.join(
                    path, "cluster_info.json"
                )
            return original_makedirs(path, *a, **kw)

        fake_run, calls = _make_fake_run(
            lambda: output_holder["output_path"]
        )
        monkeypatch.setattr(cc.os, "makedirs", spy_makedirs)
        monkeypatch.setattr(cc.subprocess, "run", fake_run)

        collector.collect()

        mpi_call = next(
            c for c in calls if c["kind"] in ("mpirun", "mpiexec")
        )
        assert mpi_call["env"]["PLM_RSH_AGENT"] == "ssh -i /custom/key", \
            "user-provided PLM_RSH_AGENT must be preserved"
