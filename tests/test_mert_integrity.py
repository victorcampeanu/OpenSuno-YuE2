"""Catch checkout transformations that break the upstream parent-code hashes."""
import ast
import hashlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MertIntegrityTest(unittest.TestCase):
    def test_parent_source_matches_sheetsage_expected_bytes(self):
        source = ast.parse((ROOT/'transcriber/modeling_sheetsage2.py').read_text(encoding='utf-8'))
        expected = next(ast.literal_eval(node.value) for node in source.body
                        if isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == 'BASE_CODE_HASHES'
                            for target in node.targets))
        self.assertTrue(expected)
        for name, digest in expected.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((ROOT/'mert'/name).read_bytes()).hexdigest(), digest,
                                 'MERT parent files must retain their exact upstream bytes, including LF line endings.')
