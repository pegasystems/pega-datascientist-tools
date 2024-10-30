"""
Testing the functionality of utils/show_versions functions
"""

import pdstools


def test_show_versions(capsys):
    pdstools.show_versions()
    pdstools.show_versions(False)
