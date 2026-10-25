"""SSH arguments must preserve supplied JSON as one remote-shell argument."""
import json
import shlex
import types
import unittest
from unittest.mock import patch
from seju_face_lab.sns_explorer import RemoteInstagramFetcher

class RemoteQuotingTest(unittest.TestCase):
    def test_manifest_data_cannot_change_remote_command_structure(self):
        handles = ["normal.name", "a'b; harmless_marker #", 'dollar$(placeholder)', 'double"quote']
        fetcher = RemoteInstagramFetcher("synthetic-worker", python_bin="/synthetic path/python3")
        with patch("seju_face_lab.sns_explorer.subprocess.run", return_value=types.SimpleNamespace(returncode=0, stdout="{}", stderr="")) as run:
            fetcher.fetch_batch(handles)
        command = run.call_args.args[0][-1]
        try:
            arguments = shlex.split(command)
        except ValueError as error:
            self.fail(f"Manifest data changed shell syntax: {error}")
        self.assertEqual(arguments, [fetcher.python_bin, "-", json.dumps(handles)])
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertIn("input", run.call_args.kwargs)

if __name__ == "__main__":
    unittest.main()
