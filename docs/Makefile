# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docsource
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

github:
	@make html
	@cp -a build/html/. ../docs

githubwin:
	@make html
	@Xcopy build\html\. ..\docs /E/Y

clean-github:
	@make clean
	@find . -type f \( -name '*.html' -or -name '*.inv' -or -name '*.js' \) -delete
	@find . -type d -name '_*' -exec rm -rf {} +
	@find . -maxdepth 1 -type d \( -name modules -or -name api \) -exec rm -rf {} +
