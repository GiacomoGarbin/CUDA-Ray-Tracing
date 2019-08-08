#include <stdio.h>				// NULL, printf
#include <float.h>				// FLT_MAX
#include <math.h>				// M_PI, sqrtf, tanf, truncf
#include <curand_kernel.h>		// curandState, curand_init
#include "header.h"

int get_scene(sphere**);
__device__ vec3 color_recursive(ray, sphere*, int, int);
__device__ vec3 color_iterative(ray, sphere*, int, int);
__global__ void kernel(vec3*, sphere*, int);

main(int argc, char* argv[]) {

	int blockDimX = argc > 1 ? atoi(argv[1]) : 32;
	int blockDimY = argc > 2 ? atoi(argv[2]) : 32;
	dim3 block(blockDimX, blockDimY);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	// scene

	sphere* hScene;
	int objs = get_scene(&hScene);

	size_t scene_bytes = objs * sizeof(sphere);

	sphere* dScene;
	cudaMalloc((sphere **) &dScene, scene_bytes);

	cudaMemcpy(dScene, hScene, scene_bytes, cudaMemcpyHostToDevice);

	// color

	size_t col_bytes = size * sizeof(vec3);

	vec3* hCol = (vec3 *) malloc(col_bytes);
	
	vec3* dCol;
	cudaMalloc((vec3 **) &dCol, col_bytes);


	printf("P3\n%d %d\n255\n", nx, ny);

	kernel<<<grid, block>>>(dCol, dScene, objs);
	// cudaMemcpy(hCol, dCol, col_bytes, cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();

	if (error != cudaSuccess) {
		printf("ERROR: %d => %s\n", error, cudaGetErrorString(error));
	} else {
		cudaMemcpy(hCol, dCol, col_bytes, cudaMemcpyDeviceToHost);

		int i = size;
		while (--i >= 0)
			printf("%d %d %d\n", (int) hCol[i].x, (int) hCol[i].y, (int) hCol[i].z);
	}

	cudaFree(dScene);
	free(hScene);

	cudaFree(dCol);
	free(hCol);
}

int get_scene(sphere** scene) {
	// 1 (surface) + 484 (small spheres) + 3 (big spheres)
	int n = 488;

	*scene = (sphere*) malloc(n * sizeof(sphere));

	// surface

	vec3 c0 = {0.0, -1000.0, 0.0};
	vec3 a0 = {0.5, 0.5, 0.5};

	(*scene)[0].center = c0;
	(*scene)[0].radius = 1000.0;
	(*scene)[0].material = LAMBERT;
	(*scene)[0].albedo = a0;

	// big spheres

	vec3 c1 = {-4.0, 1.0, 0.0};
	vec3 a1 = {0.4, 0.2, 0.1};

	(*scene)[1].center = c1;
	(*scene)[1].radius = 1.0;
	(*scene)[1].material = LAMBERT;
	(*scene)[1].albedo = a1;

	vec3 c2 = {+4.0, 1.0, 0.0};
	vec3 a2 = {0.7, 0.6, 0.5};

	(*scene)[2].center = c2;
	(*scene)[2].radius = 1.0;
	(*scene)[2].material = METAL;
	(*scene)[2].albedo = a2;

	vec3 c3 = {+0.0, 1.0, 0.0};

	(*scene)[3].center = c3;
	(*scene)[3].radius = 1.0;
	(*scene)[3].material = GLASS;
	(*scene)[3].refraction = 1.5;

	// small spheres

	int c = 4, i, j;

	for (i = -11; i < 11; i++) {
		for (j = -11; j < 11; j++) {

			float material = CPU_RAND_FLT;
			vec3 center = {(float) i + 0.9*CPU_RAND_FLT, 0.2, (float) j + 0.9*CPU_RAND_FLT};

			if (len(sub(center, c1)) > 0.9 && len(sub(center, c2)) > 0.9 && len(sub(center, c3)) > 0.9) {

				(*scene)[c].center = center;
				(*scene)[c].radius = 0.2;

				if (material < 0.8) {
					// LAMBERT
					vec3 albedo = {CPU_RAND_FLT*CPU_RAND_FLT, CPU_RAND_FLT*CPU_RAND_FLT, CPU_RAND_FLT*CPU_RAND_FLT};
					(*scene)[c].material = LAMBERT;
					(*scene)[c].albedo = albedo;
				} else if (material < 0.95) {
					// METAL
					vec3 albedo = {0.5*(1.0+CPU_RAND_FLT), 0.5*(1.0+CPU_RAND_FLT), 0.5*(1.0+CPU_RAND_FLT)};
					float fuzz = 0.5 * CPU_RAND_FLT;
					(*scene)[c].material = METAL;
					(*scene)[c].albedo = albedo;
					(*scene)[c].fuzz = fuzz;

				} else {
					// GLASS
					(*scene)[c].material = GLASS;
					(*scene)[c].refraction = 1.5;
				}

				c++;
			}

		}
	}

	return c;
}

__device__ vec3 color_recursive(ray r, sphere* scene, int objs, int depth) {
	hit_record rec;

	if (hit_list(scene, objs, r, 0.001, FLT_MAX, &rec)) {
		ray scattered;
		vec3 attenuation;

		bool scatter = false;
		switch (rec.s->material) {
		case LAMBERT:
			scatter = lambert_scatter(r, rec, &attenuation, &scattered);
			break;
		case METAL:
			scatter = metal_scatter(r, rec, &attenuation, &scattered);
			break;
		case GLASS:
			scatter = glass_scatter(r, rec, &attenuation, &scattered);
			break;
		}

		if (depth < 50 && scatter)
			return mul(attenuation, color_recursive(scattered, scene, objs, depth+1));
		else {
			vec3 black = {0.0, 0.0, 0.0};
			return black;
		}
	}

	vec3 unit_direction = unit(get_direction(r));
	float t = 0.5 * (unit_direction.y + 1.0);
	vec3 c1 = {1.0, 1.0, 1.0};
	vec3 c2 = {0.5, 0.7, 1.0};
	// (1.0 - t) * c1 + t * c2;
	return add(smul(c1, 1.0 - t), smul(c2, t));
}

__device__ vec3 color_iterative(ray r, sphere* scene, int objs, int depth) {
	hit_record rec;
	vec3 color = {1.0, 1.0, 1.0};

	while (hit_list(scene, objs, r, 0.001, FLT_MAX, &rec)) {
		ray scattered;
		vec3 attenuation;

		bool scatter = false;
		switch (rec.s->material) {
		case LAMBERT:
			scatter = lambert_scatter(r, rec, &attenuation, &scattered);
			break;
		case METAL:
			scatter = metal_scatter(r, rec, &attenuation, &scattered);
			break;
		case GLASS:
			scatter = glass_scatter(r, rec, &attenuation, &scattered);
			break;
		}

		if (depth < 50 && scatter) {
			r = scattered;
			color = mul(color, attenuation);
			depth++;
		} else {
			color = smul(color, 0.0);
			break;
		}
	}

	vec3 unit_direction = unit(get_direction(r));
	float t = 0.5 * (unit_direction.y + 1.0);
	vec3 c1 = {1.0, 1.0, 1.0};
	vec3 c2 = {0.5, 0.7, 1.0};
	// ((1.0 - t) * c1 + t * c2) * color;
	return mul(add(smul(c1, 1.0 - t), smul(c2, t)), color);
}

__global__ void kernel(vec3 *out, sphere* scene, int objs) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	// check image bounds
	if (i >= nx || j >= ny) return;

	int idx = j*nx + i;
	
	curandState state;
	curand_init(0, idx, 0, &state);

	camera cam = get_camera(lookfrom, lookat, vup, 20.0, aspect, aperture, focus);

	int ns = 100, s;
	vec3 col = {0.0, 0.0, 0.0};
	for (s = 0; s < ns; s++) {
		float u = (float) (i + GPU_RAND_FLT) / nx;
		float v = (float) (j + GPU_RAND_FLT) / ny;
		ray r = get_ray(cam, u, v);
		col = add(col, color_recursive(r, scene, objs, 0));
		// col = add(col, color_iterative(r, scene, objs, 0));
	}
	col = sdiv(col, (float) ns);

	col.x = sqrtf(col.x);
	col.y = sqrtf(col.y);
	col.z = sqrtf(col.z);

	col.x = truncf(255.99*col.x);
	col.y = truncf(255.99*col.y);
	col.z = truncf(255.99*col.z);

	out[idx] = col;
}