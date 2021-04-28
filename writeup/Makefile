PAPER := main
BIB := biblio.bib

clear_log : 
	rm tex.out
clean : 
	rm *.log *.aux

build : 
	pdflatex $(PAPER).tex &> tex.out
	# uncomment if using bibliography
	# bibtex $(PAPER) &>> tex.out
	# pdflatex $(PAPER) &>> tex.out
	# pdflatex $(PAPER) &>> tex.out

all : build clean
