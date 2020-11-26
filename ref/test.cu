#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

struct Vec
{                   // Usage: time ./smallpt 5000 && xv image.ppm
	float x, y, z; // position, also color (r,g,b)
	__device__ Vec(float x_ = 0, float y_ = 0, float z_ = 0)
	{
		x = x_;
		y = y_;
		z = z_;
	}
	__device__ Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
	__device__ Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
	__device__ Vec operator*(float b) const { return Vec(x * b, y * b, z * b); }
	__device__ Vec mult(const Vec &b) const { return Vec(x * b.x, y * b.y, z * b.z); }
	__device__ Vec &norm() { return *this = *this * (1 / sqrt(x * x + y * y + z * z)); }
	__device__ float dot(const Vec &b) const { return x * b.x + y * b.y + z * b.z; } // cross:
	__device__ Vec operator%(Vec &b) { return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
};
struct Ray
{
	Vec o, d;
	__device__ Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
};
enum Refl_t
{
	DIFF,
	SPEC,
	REFR
}; // material types, used in radiance()
struct Sphere
{
	float rad;  // radius
	Vec p, e, c; // position, emission, color
	Refl_t refl; // reflection type (DIFFuse, SPECular, REFRactive)
	__device__ Sphere(float rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_) : rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
	__device__ float intersect(const Ray &r) const
	{                     // returns distance, 0 if nohit
		Vec op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		float t, eps = 1e-4, b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;
		if (det < 0)
			return 0;
		else
			det = sqrt(det);
		return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
	}
};

__device__ inline float clamp(float x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
__device__ inline int toInt(float x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }
__device__ inline bool intersect(const Ray &r, float &t, int &id, Sphere *spheres)
{
	float n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
	for (int i = int(n); i--;)
		if ((d = spheres[i].intersect(r)) && d < t)
		{
			t = d;
			id = i;
		}
	return t < inf;
}

__device__ Vec radiance(const Ray &r, int depth, Sphere *spheres, curandState *rand_state)
{
	float t;   // distance to intersection
	int id = 0; // id of intersected object
	if (!intersect(r, t, id, spheres))
		return Vec();                // if miss, return black
	const Sphere &obj = spheres[id]; // the hit object
	Vec x = r.o + r.d * t, n = (x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n * -1, f = obj.c;
	float p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl
	if (++depth > 5)
		if (curand_uniform(rand_state) < p)
			f = f * (1 / p);
		else
			return obj.e; //R.R.
	if (obj.refl == DIFF)
	{ // Ideal DIFFUSE reflection
		float r1 = 2 * 3.1415926 * curand_uniform(rand_state), r2 = curand_uniform(rand_state), r2s = sqrt(r2);
		Vec w = nl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w % u;
		Vec d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();
		return obj.e + f.mult(radiance(Ray(x, d), depth, spheres, rand_state));
	}
	else if (obj.refl == SPEC) // Ideal SPECULAR reflection
		return obj.e + f.mult(radiance(Ray(x, r.d - n * 2 * n.dot(r.d)), depth, spheres, rand_state));
	Ray reflRay(x, r.d - n * 2 * n.dot(r.d)); // Ideal dielectric REFRACTION
	bool into = n.dot(nl) > 0;                // Ray from outside going in?
	float nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;
	if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0) // Total internal reflection
		return obj.e + f.mult(radiance(reflRay, depth, spheres, rand_state));
	Vec tdir = (r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
	float a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : tdir.dot(n));
	float Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = .25 + .5 * Re, RP = Re / P, TP = Tr / (1 - P);
	return obj.e + f.mult(depth > 2 ? (curand_uniform(rand_state) < P ? // Russian roulette
		radiance(reflRay, depth, spheres, rand_state) * RP
		: radiance(Ray(x, tdir), depth, spheres, rand_state) * TP)
		: radiance(reflRay, depth, spheres, rand_state) * Re + radiance(Ray(x, tdir), depth, spheres, rand_state) * Tr);
}

__global__ void init_sphere(Sphere *spheres)
{
	spheres[0] = Sphere(1e5, Vec(1e5 + 1, 40.8, 81.6), Vec(), Vec(.75, .25, .25), DIFF);//Left
	spheres[1] = Sphere(1e5, Vec(-1e5 + 99, 40.8, 81.6), Vec(), Vec(.25, .25, .75), DIFF);//Rght
	spheres[2] = Sphere(1e5, Vec(50, 40.8, 1e5), Vec(), Vec(.75, .75, .75), DIFF);//Back
	spheres[3] = Sphere(1e5, Vec(50, 40.8, -1e5 + 170), Vec(), Vec(), DIFF);//Frnt
	spheres[4] = Sphere(1e5, Vec(50, 1e5, 81.6), Vec(), Vec(.75, .75, .75), DIFF);//Botm
	spheres[5] = Sphere(1e5, Vec(50, -1e5 + 81.6, 81.6), Vec(), Vec(.75, .75, .75), DIFF);//Top
	spheres[6] = Sphere(16.5, Vec(27, 16.5, 47), Vec(), Vec(1, 1, 1)*.999, SPEC);//Mirr
	spheres[7] = Sphere(16.5, Vec(73, 16.5, 78), Vec(), Vec(1, 1, 1)*.999, REFR);//Glas
	spheres[8] = Sphere(600, Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12), Vec(), DIFF);//Lite
}

__global__ void init_c(Vec *c,curandState *rand_state)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	c[threadId] = Vec();
	curand_init(clock() + threadId, threadId, 0, &rand_state[threadId]);
}

__global__ void raytrac(Sphere *spheres, Vec *c,curandState *rand_state)
{
	int x = threadIdx.x, y = blockIdx.x;
	int w = 1024, h = 768, samps = 200; // # samples
	Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm());        // cam pos, dir
	Vec cx = Vec(w * .5135 / h), cy = (cx % cam.d).norm() * .5135, r;
	for (int sy = 0, i = y * w + x; sy < 2; sy++)       // 2x2 subpixel rows
		for (int sx = 0; sx < 2; sx++, r = Vec())
		{ // 2x2 subpixel cols
			for (int s = 0; s < samps; s++)
			{
				float r1 = 2 * curand_uniform(rand_state), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
				float r2 = 2 * curand_uniform(rand_state), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
			
				Vec d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
					cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
				r = r + radiance(Ray(cam.o + d * 140, d.norm()), 0, spheres, rand_state) * (1. / samps);
			} // Camera rays are pushed ^^^^^ forward to start in interior
			c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z)) * .25;
		}
}

__global__ void cpyVec2Double(Vec *c, float *d)
{
	int x = threadIdx.x, y = blockIdx.x;
	int w = 1024, h = 768;
	int i = (h - y - 1) * w + x;
	d[i * 3 + 0]=c[i].x;
	d[i * 3 + 1]=c[i].y;
	d[i * 3 + 2]=c[i].z;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

int main(int argc, char *argv[])
{
	int w = 1024, h = 768; // # samples
	float *c_h = (float *)malloc(sizeof(float)*w*h * 3);
	Sphere *spheres;
	Vec *c_d;
	float *v_d;
	curandState *curandStates;

	gpuErrchk(cudaMalloc((void**)&curandStates, sizeof(curandState) * w*h));
	gpuErrchk(cudaMalloc((void**)&spheres, sizeof(Sphere) * 9));
	gpuErrchk(cudaMalloc((void**)&c_d, sizeof(Vec)*w*h));
	gpuErrchk(cudaMalloc((void**)&v_d, sizeof(float)*3*w*h));

	init_sphere << <1, 1 >> >(spheres);
	gpuErrchk(cudaPeekAtLastError());
	init_c << <w, h >> >(c_d,curandStates);
	gpuErrchk(cudaPeekAtLastError());
	raytrac << <w, 20 >> >(spheres, c_d,curandStates);
	gpuErrchk(cudaPeekAtLastError());
	cpyVec2Double << <w, h >> > (c_d, v_d);
	cudaDeviceSynchronize();
	cudaMemcpy(c_h, v_d, sizeof(float)*3*w*h, cudaMemcpyDeviceToHost);
	cudaFree(v_d);
        cudaFree(c_d);
	cudaFree(spheres);
	cudaFree(curandStates);
	FILE *f = fopen("image.ppm", "w"); // Write image to PPM file.
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
	for (int i = 0; i < w * h; i++)
		fprintf(f, "%f %f %f ", (c_h[i*3+0]), (c_h[i*3+1]), (c_h[i*3+2]));
	fclose(f);
	free(c_h);
	return 0;
}
