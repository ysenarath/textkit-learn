build:
	hatch build

clean:
	rm -rf dist/

publish:
	twine upload dist/*

sync:
	bash scratch/bin/sync.sh
