import os
import sys
import sphinx_rtd_theme

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_ROOT = os.path.dirname(CURRENT_DIR)
XML_OUTPUT_ROOT = os.path.join(DOCS_ROOT, "xml_output")

project = 'Mobile Robot Documentation'
copyright = '2025, Lucas Momesso'
author = 'Lucas Momesso'
release = '1.0'

extensions = [
    'breathe',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'sphinx.ext.viewcode',
]

breathe_projects = {
    "manipulation": os.path.join(XML_OUTPUT_ROOT, "manipulation/xml"),
    "mobile_bringup": os.path.join(XML_OUTPUT_ROOT, "mobile_bringup/xml"),
    "mobile_manipulation_interfaces": os.path.join(XML_OUTPUT_ROOT, "mobile_manipulation_interfaces/xml"),
    "navigation": os.path.join(XML_OUTPUT_ROOT, "navigation/xml"),
    "storage_manager": os.path.join(XML_OUTPUT_ROOT, "storage_manager/xml"),
    "task_planning": os.path.join(XML_OUTPUT_ROOT, "task_planning/xml"),
}
breathe_default_project = "task_planning"

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
}


html_static_path = ['_static']
