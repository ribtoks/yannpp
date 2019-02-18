#/bin/bash

echo "Rebuilding test code..."
cd _build
cmake ..
make
cd ..

echo "Running training..."
_build/mnist_training ../data/

echo "Done"
