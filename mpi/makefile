all:
	mpicc -D_FILE_OFFSET_BITS=64 -o conv convolution.c
	mpiexec -np 4 ./conv images/im03.ppm kernel/kernel3x3_Edge.txt res.ppm 1