#ifndef HEADER
#define HEADER

#include <stdlib.h>				// rand, RAND_MAX

typedef enum {false, true} bool;

#define RAND_FLT ((float) (rand() % RAND_MAX) / RAND_MAX)


// ***** vec3 *****

typedef struct {
	float x, y, z;
} vec3;

vec3 add(vec3, vec3);
vec3 sadd(vec3, float);
vec3 sub(vec3, vec3);
vec3 ssub(vec3, float);
vec3 mul(vec3, vec3);
vec3 smul(vec3, float);
vec3 div(vec3, vec3);
vec3 sdiv(vec3, float);
float len(vec3);
vec3 unit(vec3);
float dot(vec3, vec3);
vec3 cross(vec3, vec3);


// **** ray *****

typedef struct {
	vec3 A, B;
} ray;

vec3 get_origin(ray);
vec3 get_direction(ray);
vec3 point_at_parameter(ray, float);


// ***** sphere *****

typedef enum {lambertian, metal, dielectric} materials;

typedef struct {
	vec3 center;
	float radius;
	materials material;
	vec3 albedo;		// lambertian & metal
	float fuzz;			// metal
	float refraction;	// dielectric
} sphere;

typedef struct {
	float t;
	vec3 p;
	vec3 n;
	sphere* s;
} hit_record;

bool hit_sphere(sphere*, ray, float, float, hit_record*);
bool hit_list(sphere**, ray, float, float, hit_record*);


// ***** camera *****

typedef struct {
	vec3 origin;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;

	vec3 u, v, w;
	float lens;
} camera;

vec3 random_in_unit_disk();
camera get_camera(vec3, vec3, vec3, float, float, float, float);
ray get_ray(camera, float, float);


// ***** material *****

vec3 random_in_unit_sphere();
bool lambertian_scatter(ray, hit_record, vec3*, ray*);
vec3 reflect(vec3, vec3);
bool metal_scatter(ray, hit_record, vec3*, ray*);
bool refract(vec3, vec3, float, vec3*);
float schlick(float, float);
bool dielectric_scatter(ray, hit_record, vec3*, ray*);


#endif