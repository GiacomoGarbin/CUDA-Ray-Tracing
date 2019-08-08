#include "header.h"

vec3 get_origin(ray r) {
	return r.A;
}

vec3 get_direction(ray r) {
	return r.B;
}

vec3 point_at_parameter(ray r, float t) {
	return add(r.A, smul(r.B, t));
}