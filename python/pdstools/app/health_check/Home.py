# python/pdstools/app/health_check/Home.py
"""Entry script for the ADM Health Check app.

Thin router on top of ``st.navigation()`` — keeps actual page content
out of the script body so navigation can choose which page to run.
"""

from pdstools.app.health_check._navigation import build_navigation

build_navigation().run()
