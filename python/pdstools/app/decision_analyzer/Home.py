# python/pdstools/app/decision_analyzer/Home.py
"""Entry script for the Decision Analysis app.

Thin router on top of ``st.navigation()`` — keeps actual page content
out of the script body so navigation can choose which page to run.
"""

from pdstools.app.decision_analyzer._navigation import build_navigation

build_navigation().run()
