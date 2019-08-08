#include <stdio.h>				// NULL, printf
#include <float.h>				// FLT_MAX
#define _USE_MATH_DEFINES
#include <math.h>				// M_PI, sqrtf, cosf
#include "header.h"

sphere** get_scene();
vec3 color_recursive(ray, sphere**, int);

int main() {
	int nx = 1200;
	int ny = 800;
	int ns = 1;

	vec3 lookfrom = {13.0, 2.0, 3.0};
	vec3 lookat = {0.0, 0.0, 0.0};
	vec3 vup = {0.0, 1.0, 0.0};
	float aspect = (float) nx / (float) ny;
	float aperture = 0.1;
	float focus = 10.0;
	camera cam = get_camera(lookfrom, lookat, vup, 20.0, aspect, aperture, focus);

	sphere** scene = get_scene();

	printf("P3\n%d %d\n255\n", nx, ny);

	int i, j, s;
	for (j = ny-1; j >= 0; j--) {
		for (i = 0; i < nx; i++) {
			
			vec3 col = {0.0, 0.0, 0.0};
			for (s = 0; s < ns; s++) {
				float u = (float) (i + RAND_FLT) / nx;
				float v = (float) (j + RAND_FLT) / ny;
				ray r = get_ray(cam, u, v);
				col = add(col, color_recursive(r, scene, 0));
			}
			col = sdiv(col, (float) ns);

			col.x = sqrtf(col.x);
			col.y = sqrtf(col.y);
			col.z = sqrtf(col.z);
			
			int ir = (int) 255.99*col.x;
			int ig = (int) 255.99*col.y;
			int ib = (int) 255.99*col.z;

			printf("%d %d %d\n", ir, ig, ib);
		}
	}

	sphere** ptr;
	for (ptr = scene; *ptr != NULL; ptr++) {
		free(*ptr);
	}
	free(scene);
}

sphere** get_scene() {
	// 1 (surface) + 3 (big spheres) + 484 (small spheres) + 1 (NULL)
	int n = 489;

	sphere** scene = (sphere**) malloc(n * sizeof(sphere*));
	
	// surface

	sphere* s0 = (sphere*) malloc(sizeof(sphere));

	vec3 c0 = {0.0, -1000.0, 0.0};
	vec3 a0 = {0.5, 0.5, 0.5};

	s0->center = c0;
	s0->radius = 1000.0;
	s0->material = lambertian;
	s0->albedo = a0;	

	scene[0] = s0;

	// big spheres

	sphere* s1 = (sphere*) malloc(sizeof(sphere));

	vec3 c1 = {-4.0, 1.0, 0.0};
	vec3 a1 = {0.4, 0.2, 0.1};

	s1->center = c1;
	s1->radius = 1.0;
	s1->material = lambertian;
	s1->albedo = a1;

	scene[1] = s1;

	sphere* s2 = (sphere*) malloc(sizeof(sphere));

	vec3 c2 = {+4.0, 1.0, 0.0};
	vec3 a2 = {0.7, 0.6, 0.5};

	s2->center = c2;
	s2->radius = 1.0;
	s2->material = metal;
	s2->albedo = a2;

	scene[2] = s2;

	sphere* s3 = (sphere*) malloc(sizeof(sphere));

	vec3 c3 = {0.0, 1.0, 0.0};

	s3->center = c3;
	s3->radius = 1.0;
	s3->material = dielectric;
	s3->refraction = 1.5;

	scene[3] = s3;

	// small spheres

	int c = 4, i, j;

	for (i = -11; i < 11; i++) {
		for (j = -11; j < 11; j++) {

			float material = RAND_FLT;
			vec3 center = {(float) i + 0.9*RAND_FLT, 0.2, (float) j + 0.9*RAND_FLT};

			if (len(sub(center, c1)) > 0.9 && len(sub(center, c2)) > 0.9 && len(sub(center, c3)) > 0.9) {

				sphere* s = (sphere*) malloc(sizeof(sphere));

				s->center = center;
				s->radius = 0.2;

				if (material < 0.8) {
					// lambertian
					s->material = lambertian;
					vec3 albedo = {RAND_FLT*RAND_FLT, RAND_FLT*RAND_FLT, RAND_FLT*RAND_FLT};
					s->albedo = albedo;
				} else if (material < 0.95) {
					// metal
					s->material = metal;
					vec3 albedo = {0.5*(1.0+RAND_FLT), 0.5*(1.0+RAND_FLT), 0.5*(1.0+RAND_FLT)};
					float fuzz = 0.5 * RAND_FLT;
					s->albedo = albedo;
					s->fuzz = fuzz;
				} else {
					// dielectric
					s->material = dielectric;
					s->refraction = 1.5;
				}

				scene[c++] = s;
			}

		}
	}

	scene[c] = NULL;

	return scene;
}

vec3 color_recursive(ray r, sphere** l, int depth) {
	hit_record rec;

	if (hit_list(l, r, 0.001, FLT_MAX, &rec)) {
		ray scattered;
		vec3 attenuation;

		bool scatter = false;
		switch (rec.s->material) {
		case lambertian:
			scatter = lambertian_scatter(r, rec, &attenuation, &scattered);
			break;
		case metal:
			scatter = metal_scatter(r, rec, &attenuation, &scattered);
			break;
		case dielectric:
			scatter = dielectric_scatter(r, rec, &attenuation, &scattered);
			break;
		}

		if (depth < 50 && scatter)
			return mul(attenuation, color_recursive(scattered, l, depth+1));
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