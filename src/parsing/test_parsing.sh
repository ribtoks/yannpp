#/bin/bash

echo "Rebuilding test code..."
cd _build
cmake ..
make
cd ..

echo "Running test parsing..."
_build/parse_test ../data/train-images-idx3-ubyte ../data/train-labels-idx1-ubyte

echo "Copying the results..."
rm ~/storage/downloads/mnist/*.bmp
mv *.bmp ~/storage/downloads/mnist/ 2>/dev/null

echo "Done"
