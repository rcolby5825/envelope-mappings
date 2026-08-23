import unittest

import envelope_mappings


class PackageTests(unittest.TestCase):
    def test_package_exposes_version(self):
        self.assertEqual(envelope_mappings.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
