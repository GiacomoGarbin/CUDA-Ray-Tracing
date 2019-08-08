#include <math.h>				// sqrtf
#include "header.h"

vec3 add(vec3 v1, vec3 v2) {
	vec3 temp = {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
	return temp;
}

vec3 sadd(vec3 v, float t) {
	vec3 temp = {v.x + t, v.y + t, v.z + t};
	return temp;
}

vec3 sub(vec3 v1, vec3 v2) {
	vec3 temp = {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
	return temp;
}

vec3 ssub(vec3 v, float t) {
	vec3 temp = {v.x - t, v.y - t, v.z - t};
	return temp;
}

vec3 mul(vec3 v1, vec3 v2) {
	vec3 temp = {v1.x * v2.x, v1.y * v2.y, v1.z * v2.z};
	return temp;
}

vec3 smul(vec3 v, float t) {
	vec3 temp = {v.x * t, v.y * t, v.z * t};
	return temp;
}

vec3 div(vec3 v1, vec3 v2) {
	vec3 temp = {v1.x / v2.x, v1.y / v2.y, v1.z / v2.z};
	return temp;
}

vec3 sdiv(vec3 v, float t) {
	vec3 temp = {v.x / t, v.y / t, v.z / t};
	return temp;
}

float len(vec3 v) {
	return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

vec3 unit(vec3 v) {
	return sdiv(v, len(v));
}

float dot(vec3 v1, vec3 v2) {
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

vec3 cross(vec3 v1, vec3 v2) {
	vec3 temp = {v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x};
	return temp;
}