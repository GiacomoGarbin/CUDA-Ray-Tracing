#include <curand_kernel.h>			// curandState, curand_init, curand_uniform
#include "header.h"

__device__ unsigned count[size];

__device__ vec3 random_in_unit_sphere() {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = j*nx + i;

	curandState state;
	curand_init(0, count[idx]++, 0, &state);

	vec3 p;
	do {
		vec3 temp = {GPU_RAND_FLT, GPU_RAND_FLT, GPU_RAND_FLT};
		// p = temp*2 - 1
		p = ssub(smul(temp, 2.0), 1.0);
	} while (dot(p, p) >= 1.0);
	return p;
}

__device__ vec3 random_in_unit_disk() {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = j*nx + i;

	curandState state;
	curand_init(0, count[idx]++, 0, &state);

	vec3 p;
	do {
		vec3 r = {GPU_RAND_FLT, GPU_RAND_FLT, 0.0};
		vec3 t = {1.0, 1.0, 0.0};
		// p = r*2 - t
		p = sub(smul(r, 2.0), t);
	} while (dot(p, p) >= 1.0);
	return p;
}