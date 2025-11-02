# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pinball-rt'
copyright = '2025, Patrick Sheehan'
author = 'Patrick Sheehan'

release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_automodapi.automodapi',
    'numpydoc',
    'sphinx.ext.autosectionlabel',
]

numpydoc_show_class_members = False

autoclass_content = 'both'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

html_title = "pinball-rt Documentation"

html_theme_options = {
        "repository_url": "https://github.com/psheehan/pinball-warp",
        "use_repository_button": True,
        "use_edit_page_button": True,
        "use_source_button": True,
        "use_issues_button": True,
        }

html_logo = "_static/pinball_logo.png"
