lint_local:	
	isort *.py && \
	black -l 125 --exclude "/temp*|/labelImg" . && \
	flake8 .