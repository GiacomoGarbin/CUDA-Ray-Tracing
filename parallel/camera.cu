#include <math.h>				// M_PI, tanf
#include "header.h"

__device__ vec3 lookfrom = {13.0, 2.0, 3.0};
__device__ vec3 lookat = {0.0, 0.0, 0.0};
__device__ vec3 vup = {0.0, 1.0, 0.0};
__device__ float aspect = (float) nx / (float) ny;
__device__ float aperture = 0.1;
__device__ float focus = 10.0;

__device__ camera get_camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus) {
	vec3 u, v, w;
	float lens = aperture / 2.0;

	float theta = vfov * (float) M_PI / 180.0;
	float half_height = tanf(theta / 2.0);
	float half_width = aspect * half_height;

	vec3 origin = lookfrom;

	w = unit(sub(lookfrom, lookat));
	u = unit(cross(vup, w));
	v = cross(w, u);

	// lower_left_corner = origin - half_width*focus*u - half_height*focus*v - focus*w
	vec3 lower_left_corner = sub(sub(sub(origin, smul(u, half_width*focus)), smul(v, half_height*focus)), smul(w, focus));

	vec3 horizontal = smul(u, 2.0*half_width*focus);
	vec3 vertical = smul(v, 2.0*half_height*focus);

	camera cam = {origin, lower_left_corner, horizontal, vertical, u, v, w, lens};
	return cam;
}

__device__ ray get_ray(camera cam, float s, float t) {
	vec3 rd = smul(random_in_unit_disk(), cam.lens);
	// offset = u*rd.x + v*rd.y
	vec3 offset = add(smul(cam.u, rd.x), smul(cam.v, rd.y));
	// r = {origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset}
	ray r = {add(cam.origin, offset), sub(sub(add(cam.lower_left_corner, add(smul(cam.horizontal, s), smul(cam.vertical, t))), cam.origin), offset)};
	return r;
}