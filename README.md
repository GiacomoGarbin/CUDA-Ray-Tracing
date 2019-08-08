# CUDA-Ray-Tracing

This was my project for the GPU Computing course at the University of Milan.

I parallelized using Nvidia CUDA the algorithm illustrated by Peter Shirley in his book [Ray Tracing in One Weekend](http://in1weekend.blogspot.com/2016/01/ray-tracing-in-one-weekend.html).

In the repository you can find both versions of the algorithm, serial and parallel, and the respective results.

The speedup of the parallel version I found on an Nvidia Tesla M2090 board was remarkable. With the same number of samples (ie, image quality), the parallel version requires less than 2 minutes against the more than 4 hours required by the sequential version to generate the image.
