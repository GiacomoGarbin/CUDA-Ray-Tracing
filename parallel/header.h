#ifndef HEADER
#define HEADER

#include <stdlib.h>				// rand, RAND_MAX
#include <curand_kernel.h>		// curand_uniform

// typedef enum {false, true} bool;

#define nx 1200
#define ny 800
#define size nx*ny


// ***** vec3 *****

typedef struct {
	float x, y, z;
} vec3;

extern __device__ vec3 add(vec3, vec3);
extern __device__ vec3 sadd(vec3, float);
// this function is compiled for both host and device
extern __host__ __device__ vec3 sub(vec3, vec3);
extern __device__ vec3 ssub(vec3, float);
extern __device__ vec3 mul(vec3, vec3);
extern __device__ vec3 smul(vec3, float);
extern __device__ vec3 div(vec3, vec3);
extern __device__ vec3 sdiv(vec3, float);
// this function is compiled for both host and device
extern __host__ __device__ float len(vec3);
extern __device__ vec3 unit(vec3);
extern __device__ float dot(vec3, vec3);
extern __device__ vec3 cross(vec3, vec3);


// **** ray *****

typedef struct {
	vec3 A, B;
} ray;

extern __device__ vec3 get_origin(ray);
extern __device__ vec3 get_direction(ray);
extern __device__ vec3 point_at_parameter(ray, float);


// ***** sphere *****

typedef enum {LAMBERT, METAL, GLASS} materials;

typedef struct {
	vec3 center;
	float radius;
	materials material;
	vec3 albedo;
	float fuzz;
	float refraction;
} sphere;

typedef struct {
	float t;
	vec3 p;
	vec3 n;
	sphere* s;
} hit_record;

extern __device__ bool hit_sphere(sphere*, ray, float, float, hit_record*);
extern __device__ bool hit_list(sphere*, int, ray, float, float, hit_record*);


// ***** material *****

extern __device__ bool lambert_scatter(ray, hit_record, vec3*, ray*);
extern __device__ vec3 reflect(vec3, vec3);
extern __device__ bool metal_scatter(ray, hit_record, vec3*, ray*);
extern __device__ bool refract(vec3, vec3, float, vec3*);
extern __device__ float schlick(float, float);
extern __device__ bool glass_scatter(ray, hit_record, vec3*, ray*);


// ***** camera *****

typedef struct {
	vec3 origin;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	float lens;
} camera;

extern __device__ vec3 lookfrom;
extern __device__ vec3 lookat;
extern __device__ vec3 vup;
extern __device__ float aspect;
extern __device__ float aperture;
extern __device__ float focus;
extern __device__ camera cam;

extern __device__ camera get_camera(vec3, vec3, vec3, float, float, float, float);
extern __device__ ray get_ray(camera, float, float);


// **** random *****

#define CPU_RAND_FLT ((float) (rand() % RAND_MAX) / RAND_MAX)
#define GPU_RAND_FLT (1.0 - curand_uniform(&state))

extern __device__ unsigned count[size];

extern __device__ vec3 random_in_unit_sphere();
extern __device__ vec3 random_in_unit_disk();


#endif