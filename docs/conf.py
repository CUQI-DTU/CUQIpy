# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from sphinx_gallery.sorting import FileNameSortKey
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'cuqipy'
copyright = '2022, CUQI Project, Technical University of Denmark (DTU)'
author = 'CUQI'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_panels',
    'sphinx.ext.todo',
    'sphinx_copybutton',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.autosummary',
    ]

# Options for extensions
autodoc_member_order = 'groupwise'
autodoc_default_flags = ['members']
todo_include_todos=True
autosummary_generate = True
autodoc_typehints = "none"
autosummary_imported_members = True # Adds any imported members to api docs automatically
autosummary_ignore_module_all = False # Adds only members from __all__ to the api docs (if present)


# Sphinx-gallery configuration
sphinx_gallery_conf = {
    'filename_pattern': '/*',
    'examples_dirs': ['../demos/tutorials', '../demos/howtos', '../demos/dev'],
    'gallery_dirs': ['user/_auto_tutorials', 'user/_auto_howtos', 'dev/_auto_dev'],
    'download_all_examples': False,
    'within_subsection_order': FileNameSortKey,
}


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The root document.
root_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '_templates']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


html_logo = "../logo.png"

#TODO: The option 'logo' is to be removed once pydata-sphinx-theme 
# makes their theme compatible with sphinx 6.0.0 and later.
html_theme_options = {
    'collapse_navigation': False,
    'navigation_depth': 3,
    "show_prev_next": False,
    "logo": {
      "image_light": "https://github.com/CUQI-DTU/CUQIpy/raw/main/logo.png",
      "image_dark": "https://github.com/CUQI-DTU/CUQIpy/raw/main/logo.png",
   }
}

def setup(app):
    app.add_css_file('custom.css')

