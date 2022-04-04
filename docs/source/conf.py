# Configuration file for the Sphinx documentation builder.

#sys.path.insert(0, os.path.abspath('../../'))

# -- Project information

project = 'MIMo'
copyright = '2022, Dominik Mattern'
author = 'Dominik Mattern, Francisco Lopez, Markus Ernst, Arthur Aubret'

release = '0.1.0'
version = '0.1.0'

# -- General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
]
numpydoc_class_members_toctree = False
automodapi_toctreedirnm = 'generated'
automodsumm_inherited_members = True
autodoc_mock_imports = ["mujoco-py"]

templates_path = ['_templates']

source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output
#import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
#html_static_path = ['_static']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'mimo': ('https://mimo.readthedocs.io/en/latest/', None),
}
intersphinx_disabled_domains = ['std']

autosummary_generate = True

# -- Options for EPUB output
epub_show_urls = 'footnote'