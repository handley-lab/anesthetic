default: pdf

MDFILE=paper.md
BIBFILE=paper.bib
LATEXFILE=paper.tex

# some variables for the paper
REPOSITORY=https://github.com/williamjameshandley/anesthetic
	
	
ARCHIVE=http://dx.doi.org/10.5281/zenodo.2614017
REVIEW=https://github.com/openjournals/joss-reviews/issues/1414
ARCHIVEDOI=10.5281/zenodo.2614017 
FORMATTEDDOI=10.21105/joss.00538
YEAR=2019
TITLE="anesthetic: nested sampling visualisation"
AUTHOR="Handley"
ISSUE=37
VOLUME=4
PAGE=1414
JOURNALNAME="Journal of Open Source Software"
PUBLISHED="12 May 2019"
SUBMITTED="22 April 2019"
PAPERURL=https://www.theoj.org/joss-papers/joss.01414/10.21105.joss.01414.pdf


latex: $(MDFILE) $(BIBFILE)
	# get JOSS template
	curl https://raw.githubusercontent.com/openjournals/whedon/master/resources/latex.template > latex.template
	# get JOSS logo
	curl https://raw.githubusercontent.com/openjournals/whedon/master/resources/joss-logo.png > joss-logo.png
	# use pandoc and pandoc-citeproc to create LaTeX source
	pandoc --filter pandoc-citeproc --bibliography=$(BIBFILE) -V repository="$(REPOSITORY)" -V journal_name=$(JOURNALNAME) -V paper_url="$(PAPERURL)" -V archive_doi="$(ARCHIVEDOI)" -V review_issue_url="$(REVIEW)" -V formatted_doi="$(FORMATTEDDOI)" -V logo_path="joss-logo.png" -V year="$(YEAR)" -V footnote_paper_title=$(TITLE) -V citation_author=$(AUTHOR) -V issue="$(ISSUE)" -V volume="$(VOLUME)" -V page="$(PAGE)" -V submitted=$(SUBMITTED) -V published=$(PUBLISHED) -V graphics="true" -V papersize=a4paper -V geometry:margin=1in -t latex -s $(MDFILE) -o $(LATEXFILE) --template latex.template

pdf: latex
	pdflatex paper.tex
	pdflatex paper.tex

clean:
	rm paper.bcf paper.aux paper.log paper.out paper.run.xml

clean_downloads:
	rm latex.template joss-logo.png
