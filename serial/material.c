#include <math.h>			// sqrtf, powf
#include "header.h"

vec3 random_in_unit_sphere() {
	vec3 p;
	do {
		vec3 temp = {RAND_FLT, RAND_FLT, RAND_FLT};
		// p = temp*2 - 1
		p = ssub(smul(temp, 2.0), 1.0);
	} while (dot(p, p) >= 1.0);
	return p;
}

bool lambertian_scatter(ray r, hit_record rec, vec3* attenuation, ray* scattered) {
	vec3 target = add(rec.p, add(rec.n, random_in_unit_sphere()));
	ray temp = {rec.p, sub(target, rec.p)};
	*scattered = temp;
	*attenuation = rec.s->albedo;
	return true;
}

vec3 reflect(vec3 v, vec3 n) {
	// v - 2*dot(v,n)*n
	return sub(v, smul(n, 2*dot(v, n)));
}

bool metal_scatter(ray r, hit_record rec, vec3* attenuation, ray* scattered) {
	vec3 reflected = reflect(unit(get_direction(r)), rec.n);
	// scattered = ray{rec.p, reflected + random_in_unit_sphere()*fuzz}
	ray temp = {rec.p, add(reflected, smul(random_in_unit_sphere(), rec.s->fuzz))};
	*scattered = temp;
	*attenuation = rec.s->albedo;
	return dot(get_direction(*scattered), rec.n) > 0;
}

bool refract(vec3 v, vec3 n, float ni_over_nt, vec3* refracted) {
	vec3 uv = unit(v);
	float dt = dot(uv, n);
	float discriminant = 1.0 - ni_over_nt*ni_over_nt * (1.0 - dt*dt);

	if (discriminant > 0.0) {
		// refracted = ni_over_nt * (uv - n*dt) - n*sqrtf(discriminant)
		*refracted = sub(smul(sub(uv, smul(n, dt)), ni_over_nt), smul(n, sqrtf(discriminant)));
		return true;
	}

	return false;
}

float schlick(float cosine, float refraction) {
	float r0 = (1.0 - refraction) / (1.0 + refraction);
	r0 = r0*r0;
	return r0 + (1.0 - r0) * powf(1.0 - cosine, 5.0);
}

bool dielectric_scatter(ray r, hit_record rec, vec3* attenuation, ray* scattered) {
	vec3 outward;
	vec3 reflected = reflect(get_direction(r), rec.n);
	float ni_over_nt;
	vec3 white = {1.0, 1.0, 1.0};
	*attenuation = white;
	vec3 refracted;

	float reflect_prob;
	float cosine;

	if (dot(get_direction(r), rec.n) > 0.0) {
		outward = smul(rec.n, -1.0);
		ni_over_nt = rec.s->refraction;

		cosine = rec.s->refraction * dot(get_direction(r), rec.n) / len(get_direction(r));
	} else {
		outward = rec.n;
		ni_over_nt = 1.0 / rec.s->refraction;

		cosine = (-1.0) * dot(get_direction(r), rec.n) / len(get_direction(r));
	}

	if (refract(get_direction(r), outward, ni_over_nt, &refracted)) {
		reflect_prob = schlick(cosine, rec.s->refraction);
	} else {
		ray temp = {rec.p, reflected};
		*scattered = temp;
		reflect_prob = 1.0;
	}

	if (RAND_FLT < reflect_prob) {
		ray temp = {rec.p, reflected};
		*scattered = temp;
	} else {
		ray temp = {rec.p, refracted};
		*scattered = temp;
	}

	return true;
}