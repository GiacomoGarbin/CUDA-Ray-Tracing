#include "header.h"

__device__ vec3 get_origin(ray r) {
	return r.A;
}

__device__ vec3 get_direction(ray r) {
	return r.B;
}

__device__ vec3 point_at_parameter(ray r, float t) {
	// A + t*B
	return add(r.A, smul(r.B, t));
}