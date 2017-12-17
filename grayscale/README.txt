RGB to Grayscale Conversion using CUDA

This code makes use of the LodePNG Library, which can be downloaded using the following link:
http://lodev.org/lodepng/

The set up is pretty simple:

1) Include the header file and cpp file in your project
2) Rename .cpp file to .c file if necessary
3) Link library to project files when compiling (see Makefile)

Note: This project will only work with .png files

Compile & Execute:

Run make in the terminal to compile executables.

CPU: ./grayscale_seq <input file> <output file> 
GPU: ./grayscale_opt <intput file> <output file> <option> 

<input file> = input image to perform grayscale conversion on
<output file> = the output image file to be written to after conversion is done
<option> = 0, 1 or 2 to specify which grayscale method you would like to see. 

Results and Hardware Used:

An estimated 100x speedup is noticed going from the CPU version to the GPU version. 


