#include <stdio.h>				// NULL
#include <math.h>				// sqrtf
#include "header.h"

__device__ bool hit_sphere(sphere* s, ray r, float tmin, float tmax, hit_record* rec) {
	vec3 oc = sub(get_origin(r), s->center);
	float a = dot(get_direction(r), get_direction(r));
	float b = 2.0 * dot(oc, get_direction(r));
	float c = dot(oc, oc) - s->radius*s->radius;
	float t, discriminant = b*b - 4.0*a*c;
	if (discriminant > 0) {
		t = (-b - sqrtf(discriminant)) / (2.0*a);
		if (tmin < t && t < tmax) {
			rec->t = t;
			rec->p = point_at_parameter(r, rec->t);
			rec->n = sdiv(sub(rec->p, s->center), s->radius);
			rec->s = s;
			return true;
		}
		t = (-b + sqrtf(discriminant)) / (2.0*a);
		if (tmin < t && t < tmax) {
			rec->t = t;
			rec->p = point_at_parameter(r, rec->t);
			rec->n = sdiv(sub(rec->p, s->center), s->radius);
			rec->s = s;
			return true;
		}
	}
	return false;
}

__device__ bool hit_list(sphere* l, int n, ray r, float tmin, float tmax, hit_record* rec) {
	hit_record temp;
	bool hit_anything = false;
	float closest_so_far = tmax;
	sphere* ptr;
	for (ptr = l; --n >= 0; ptr++) {
		if (hit_sphere(ptr, r, tmin, closest_so_far, &temp)) {
			hit_anything = true;
			closest_so_far = temp.t;
			*rec = temp;
		}
	}
	return hit_anything;
}