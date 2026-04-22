# python/pdstools/app/impact_analyzer/Home.py
"""Entry script for the Impact Analyzer app.

Thin router on top of ``st.navigation()`` — keeps actual page content
out of the script body so navigation can choose which page to run.
"""

from pdstools.app.impact_analyzer._navigation import build_navigation

build_navigation().run()
