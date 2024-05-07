build:
	maturin sdist
	maturin build --release
	cp -r rust/target/wheels/ dist/
	rm -rf rust/target/

dev:
	rm -rf src/rslib.abi3.so
	maturin develop

clean:
	rm -rf src/rslib.abi3.so
	rm -rf dist/

publish:
	twine upload dist/*

sync:
	bash scripts/sync.sh