#include <miner.h>

#include <cuda_helper.h>
//unsigned int atomicAdd(unsigned int* address, unsigned int val) {};


#define saes_data(w) {\
    w(0x63), w(0x7c), w(0x77), w(0x7b), w(0xf2), w(0x6b), w(0x6f), w(0xc5),\
    w(0x30), w(0x01), w(0x67), w(0x2b), w(0xfe), w(0xd7), w(0xab), w(0x76),\
    w(0xca), w(0x82), w(0xc9), w(0x7d), w(0xfa), w(0x59), w(0x47), w(0xf0),\
    w(0xad), w(0xd4), w(0xa2), w(0xaf), w(0x9c), w(0xa4), w(0x72), w(0xc0),\
    w(0xb7), w(0xfd), w(0x93), w(0x26), w(0x36), w(0x3f), w(0xf7), w(0xcc),\
    w(0x34), w(0xa5), w(0xe5), w(0xf1), w(0x71), w(0xd8), w(0x31), w(0x15),\
    w(0x04), w(0xc7), w(0x23), w(0xc3), w(0x18), w(0x96), w(0x05), w(0x9a),\
    w(0x07), w(0x12), w(0x80), w(0xe2), w(0xeb), w(0x27), w(0xb2), w(0x75),\
    w(0x09), w(0x83), w(0x2c), w(0x1a), w(0x1b), w(0x6e), w(0x5a), w(0xa0),\
    w(0x52), w(0x3b), w(0xd6), w(0xb3), w(0x29), w(0xe3), w(0x2f), w(0x84),\
    w(0x53), w(0xd1), w(0x00), w(0xed), w(0x20), w(0xfc), w(0xb1), w(0x5b),\
    w(0x6a), w(0xcb), w(0xbe), w(0x39), w(0x4a), w(0x4c), w(0x58), w(0xcf),\
    w(0xd0), w(0xef), w(0xaa), w(0xfb), w(0x43), w(0x4d), w(0x33), w(0x85),\
    w(0x45), w(0xf9), w(0x02), w(0x7f), w(0x50), w(0x3c), w(0x9f), w(0xa8),\
    w(0x51), w(0xa3), w(0x40), w(0x8f), w(0x92), w(0x9d), w(0x38), w(0xf5),\
    w(0xbc), w(0xb6), w(0xda), w(0x21), w(0x10), w(0xff), w(0xf3), w(0xd2),\
    w(0xcd), w(0x0c), w(0x13), w(0xec), w(0x5f), w(0x97), w(0x44), w(0x17),\
    w(0xc4), w(0xa7), w(0x7e), w(0x3d), w(0x64), w(0x5d), w(0x19), w(0x73),\
    w(0x60), w(0x81), w(0x4f), w(0xdc), w(0x22), w(0x2a), w(0x90), w(0x88),\
    w(0x46), w(0xee), w(0xb8), w(0x14), w(0xde), w(0x5e), w(0x0b), w(0xdb),\
    w(0xe0), w(0x32), w(0x3a), w(0x0a), w(0x49), w(0x06), w(0x24), w(0x5c),\
    w(0xc2), w(0xd3), w(0xac), w(0x62), w(0x91), w(0x95), w(0xe4), w(0x79),\
    w(0xe7), w(0xc8), w(0x37), w(0x6d), w(0x8d), w(0xd5), w(0x4e), w(0xa9),\
    w(0x6c), w(0x56), w(0xf4), w(0xea), w(0x65), w(0x7a), w(0xae), w(0x08),\
    w(0xba), w(0x78), w(0x25), w(0x2e), w(0x1c), w(0xa6), w(0xb4), w(0xc6),\
    w(0xe8), w(0xdd), w(0x74), w(0x1f), w(0x4b), w(0xbd), w(0x8b), w(0x8a),\
    w(0x70), w(0x3e), w(0xb5), w(0x66), w(0x48), w(0x03), w(0xf6), w(0x0e),\
    w(0x61), w(0x35), w(0x57), w(0xb9), w(0x86), w(0xc1), w(0x1d), w(0x9e),\
    w(0xe1), w(0xf8), w(0x98), w(0x11), w(0x69), w(0xd9), w(0x8e), w(0x94),\
    w(0x9b), w(0x1e), w(0x87), w(0xe9), w(0xce), w(0x55), w(0x28), w(0xdf),\
    w(0x8c), w(0xa1), w(0x89), w(0x0d), w(0xbf), w(0xe6), w(0x42), w(0x68),\
    w(0x41), w(0x99), w(0x2d), w(0x0f), w(0xb0), w(0x54), w(0xbb), w(0x16) }

#define SAES_WPOLY           0x011b

#define saes_b2w(b0, b1, b2, b3) (((uint32_t)(b3) << 24) | \
    ((uint32_t)(b2) << 16) | ((uint32_t)(b1) << 8) | (b0))

#define saes_f2(x)   ((x<<1) ^ (((x>>7) & 1) * SAES_WPOLY))
#define saes_f3(x)   (saes_f2(x) ^ x)
#define saes_h0(x)   (x)

#define saes_u0(p)   saes_b2w(saes_f2(p),          p,          p, saes_f3(p))
#define saes_u1(p)   saes_b2w(saes_f3(p), saes_f2(p),          p,          p)
#define saes_u2(p)   saes_b2w(         p, saes_f3(p), saes_f2(p),          p)
#define saes_u3(p)   saes_b2w(         p,          p, saes_f3(p), saes_f2(p))

static  __constant__  uint32_t saes_table[4][256] = { saes_data(saes_u0), saes_data(saes_u1), saes_data(saes_u2), saes_data(saes_u3) };


#define GPU_DEBUG
#define VERUS_KEY_SIZE 8832
#define VERUS_KEY_SIZE128 552
#define THREADS 64


#define AES2_EMU(s0, s1, rci) \
  aesenc(&s0, &rc[rci],sharedMemory1); \
  aesenc(&s1, &rc[rci + 1],sharedMemory1); \
  aesenc(&s0, &rc[rci + 2],sharedMemory1); \
  aesenc(&s1, &rc[rci + 3],sharedMemory1);

#define AES4(s0, s1, s2, s3, rci) \
  aesenc(&s0, &rc[rci],sharedMemory1); \
  aesenc(&s1, &rc[rci + 1],sharedMemory1); \
  aesenc(&s2, &rc[rci + 2],sharedMemory1); \
  aesenc(&s3, &rc[rci + 3],sharedMemory1); \
  aesenc(&s0, &rc[rci + 4], sharedMemory1); \
  aesenc(&s1, &rc[rci + 5], sharedMemory1); \
  aesenc(&s2, &rc[rci + 6], sharedMemory1); \
  aesenc(&s3, &rc[rci + 7], sharedMemory1);


#define AES4_LAST(s3, rci) \
  aesenc(&s3, &rc[rci + 2],sharedMemory1); \
  aesenc(&s3, &rc[rci + 6], sharedMemory1); \


#define TRUNCSTORE(out, s4) \
  *(uint32_t*)(out + 28) = s4.y;

#define MIX2_EMU(s0, s1) \
  tmp = _mm_unpacklo_epi32_emu(s0, s1); \
  s1 = _mm_unpackhi_epi32_emu(s0, s1); \
  s0 = tmp;

#define MIX4(s0, s1, s2, s3) \
  tmp  = _mm_unpacklo_epi32_emu(s0, s1); \
  s0 = _mm_unpackhi_epi32_emu(s0, s1); \
  s1 = _mm_unpacklo_epi32_emu(s2, s3); \
  s2 = _mm_unpackhi_epi32_emu(s2, s3); \
  s3 = _mm_unpacklo_epi32_emu(s0, s2); \
  s0 = _mm_unpackhi_epi32_emu(s0, s2); \
  s2 = _mm_unpackhi_epi32_emu(s1, tmp); \
  s1 = _mm_unpacklo_epi32_emu(s1, tmp);

#define MIX4_LASTBUT1(s0, s1, s2, s3) \
  tmp  = _mm_unpacklo_epi32_emu(s0, s1); \
  s1 = _mm_unpacklo_epi32_emu(s2, s3); \
  s2 = _mm_unpackhi_epi32_emu(s1, tmp); 

typedef struct case_struct {
	uint32_t *case_array[8];
} main_case_s;

__host__ void verus_setBlock(uint8_t *blockf, uint32_t *pTargetIn, uint8_t *lkey, int thr_id);


__global__ void verus_gpu_hash(uint32_t threads, uint32_t startNonce, uint32_t *resNonce, uint4 * d_key_input, uint4 * d_mid, uint32_t *d_fix_r, uint32_t *d_fix_rex);
__global__ void verus_gpu_final(uint32_t threads, uint32_t startNonce, uint32_t *resNonce, uint4 * d_key_input, const  uint4 * d_mid);
__global__ void verus_extra_gpu_prepare(const uint32_t threads, uint4 * d_key_input);
__global__ void verus_extra_gpu_fix(const uint32_t threads, uint4 * d_key_input, uint32_t *d_fix_r, uint32_t *d_fix_rex);
__global__ void verus_case_0(const uint32_t threads, const uint32_t startNonce, uint4 * __restrict__ d_key_input, uint4 * __restrict__ d_mid,
	 uint32_t iteration, main_case_s  d_main_case, uint32_t *totals, main_case_s  d_case_dest);
__global__ void verus_case_4(const uint32_t threads, const uint32_t startNonce, uint4 * __restrict__ d_key_input, uint4 * __restrict__ d_mid,
	 uint32_t iteration, main_case_s  d_main_case, uint32_t *totals, main_case_s  d_case_dest);
__global__ void verus_case_8(const uint32_t threads, const uint32_t startNonce, uint4 * __restrict__ d_key_input, uint4 * __restrict__ d_mid,
	 uint32_t iteration, main_case_s  d_main_case, uint32_t *totals, main_case_s  d_case_dest);
__global__ void verus_case_c(const uint32_t threads, const uint32_t startNonce, uint4 * __restrict__ d_key_input, uint4 * __restrict__ d_mid,
	 uint32_t iteration, main_case_s  d_main_case, uint32_t *totals, main_case_s  d_case_dest);
__global__ void verus_case_10(const uint32_t threads, const uint32_t startNonce, uint4 * __restrict__ d_key_input, uint4 * __restrict__ d_mid,
	 uint32_t iteration, main_case_s  d_main_case, uint32_t *totals, main_case_s  d_case_dest);
__global__ void verus_case_14(const uint32_t threads, const uint32_t startNonce, uint4 * __restrict__ d_key_input, uint4 * __restrict__ d_mid,
	 uint32_t iteration, main_case_s  d_main_case, uint32_t *totals, main_case_s  d_case_dest);
__global__ void verus_case_18(const uint32_t threads, const uint32_t startNonce, uint4 * __restrict__ d_key_input, uint4 * __restrict__ d_mid,
	 uint32_t iteration, main_case_s  d_main_case, uint32_t *totals, main_case_s  d_case_dest);
__global__ void verus_case_1c(const uint32_t threads, const uint32_t startNonce, uint4 * __restrict__ d_key_input, uint4 * __restrict__ d_mid,
	 uint32_t iteration, main_case_s  d_main_case, uint32_t *totals, main_case_s  d_case_dest);


static uint32_t *d_nonces[MAX_GPUS];
static uint4 *d_long_keys[MAX_GPUS];

static uint4 *d_mid[MAX_GPUS];
static main_case_s main_case_struct_p[MAX_GPUS];
static main_case_s main_case_dest[MAX_GPUS];
static uint32_t *d_case_totals[MAX_GPUS];
__device__ __constant__ uint4 vkey[VERUS_KEY_SIZE128];
__device__ __constant__ uint4 blockhash_half[4];
__device__ __constant__ uint4 blockhash_half2[4];
__device__ __constant__ uint32_t ptarget[8];

static cudaStream_t streams[MAX_GPUS][8];
__host__
void verus_init(int thr_id, uint32_t throughput)
{
	//cudaFuncSetCacheConfig(verus_gpu_hash, cudaFuncCachePreferL1);
	//	printf("start of verus_init\n");
	CUDA_SAFE_CALL(cudaMalloc(&d_nonces[thr_id], 1 * sizeof(uint32_t)));

	CUDA_SAFE_CALL(cudaMalloc(&d_long_keys[thr_id], throughput * VERUS_KEY_SIZE));
	CUDA_SAFE_CALL(cudaMalloc(&d_mid[thr_id], throughput * 16));

	for (int i = 0; i < 8; i++)
	{
		CUDA_SAFE_CALL(cudaMalloc(&main_case_struct_p[thr_id].case_array[i], throughput * sizeof(uint32_t)));
	}
	for (int i = 0; i < 8; i++)
	{
		CUDA_SAFE_CALL(cudaMalloc(&main_case_dest[thr_id].case_array[i], throughput * sizeof(uint32_t)));
	}

	CUDA_SAFE_CALL(cudaMalloc(&d_case_totals[thr_id], sizeof(uint32_t) * 8));
	//	printf("send of verus_init\n");
	for (int i = 0; i < 8; i++)
		cudaStreamCreate(&streams[thr_id][i]);
};

__host__
void verus_setBlock(uint8_t blockf[64], uint32_t *pTargetIn, uint8_t *lkey, int thr_id, uint32_t throughput)
{

	//	printf("start of veruse_setblocks\n");
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(ptarget, (void**)pTargetIn, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(blockhash_half2, (void**)blockf, 64 * sizeof(uint8_t), 0, cudaMemcpyHostToDevice));
	
	for(int i=16;i<32;i++)
	blockf[i] = blockf[i] ^ blockf[i+32];
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(blockhash_half, (void**)blockf, 64 * sizeof(uint8_t), 0, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(vkey, (void**)lkey, VERUS_KEY_SIZE * sizeof(uint8_t), 0, cudaMemcpyHostToDevice));
	dim3 grid2(throughput);
	//verus_extra_gpu_prepare << <grid2, 128 >> > (0, d_long_keys[thr_id]); //setup global mem with lots of keys	
	//	printf("end of veruse_setblocks\n");main_case_struct_p[thr_id]
};

#define THREADS2 128

__global__
void copythings(uint32_t threads, main_case_s d_main_case, main_case_s d_main_case_dest)
{

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads) {
#pragma unroll 8
		for (int i = 0; i < 8; i++)
			d_main_case_dest.case_array[i][thread] = d_main_case.case_array[i][thread];

	}


}



__host__
void verus_hash(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *resNonces, uint32_t startcase)
{
	//printf("start of verus)hash\n");

	cudaMemset(d_nonces[thr_id], 0xff, 1 * sizeof(uint32_t));
	const uint32_t threadsperblock = THREADS;
	const uint32_t threadsperblock256 = 256;

	dim3 grid_final((threads + threadsperblock256 - 1) / threadsperblock256);
	dim3 grid_fix(threads);
	dim3 block(threadsperblock);

	uint32_t case_totals[8];
	startcase >>= 2;
	verus_extra_gpu_prepare << <grid_fix, 128 >> > (0, d_long_keys[thr_id]);
	//printf("startcase = %x, \t threads = %d\n", startcase, threads);
	for (int i = 0; i < 8; i++) {

		if (i == startcase)
			case_totals[i] = threads;
		else
			case_totals[i] = 0;


	}
	//CUDA_SAFE_CALL(cudaMemcpy(d_case_totals[thr_id], case_totals, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	main_case_s *case_pointer[2];
	case_pointer[0] = &main_case_struct_p[thr_id];
	case_pointer[1] = &main_case_dest[thr_id];
	int b, c;
	for (int i = 0; i < 32; i++)
	{
		b = i & 1;
		c = !(i & 1);
		CUDA_SAFE_CALL(cudaMemset(d_case_totals[thr_id], 0, 8 * sizeof(uint32_t)));
		//cudaDeviceSynchronize();

		if (case_totals[5] > 0) {

			dim3 grid5((case_totals[5] + threadsperblock - 1) / threadsperblock);
			verus_case_14 << <grid5, block, 0, streams[thr_id][0] >> > (case_totals[5], startNonce, d_long_keys[thr_id], d_mid[thr_id], 
				i, case_pointer[b][0], d_case_totals[thr_id], case_pointer[c][0]);
		}

		if (case_totals[0] > 0) {
			//	printf("got to case 0, totals = %d, startnonce %d \n", case_totals[0], startNonce);

			dim3 grid0((case_totals[0] + threadsperblock - 1) / threadsperblock);
			verus_case_0 << <grid0, block, 0, streams[thr_id][1] >> > (case_totals[0], startNonce, d_long_keys[thr_id], d_mid[thr_id], 
				i, case_pointer[b][0], d_case_totals[thr_id], case_pointer[c][0]);
		}

		if (case_totals[1] > 0) {
			//printf("got to case 1, totals = %d\n", case_totals[1]);
			dim3 grid1((case_totals[1] + threadsperblock - 1) / threadsperblock);
			verus_case_4 << <grid1, block, 0, streams[thr_id][2] >> > (case_totals[1], startNonce, d_long_keys[thr_id], d_mid[thr_id], 
				i, case_pointer[b][0], d_case_totals[thr_id], case_pointer[c][0]);
		}

		if (case_totals[2] > 0) {
			//	printf("got to case 2, totals = %d\n", case_totals[2]);
			dim3 grid2((case_totals[2] + threadsperblock - 1) / threadsperblock);
			verus_case_8 << <grid2, block, 0, streams[thr_id][3] >> > (case_totals[2], startNonce, d_long_keys[thr_id], d_mid[thr_id], 
				i, case_pointer[b][0], d_case_totals[thr_id], case_pointer[c][0]);
		}

		if (case_totals[3] > 0) {

			dim3 grid3((case_totals[3] + threadsperblock - 1) / threadsperblock);
			verus_case_c << <grid3, block, 0, streams[thr_id][4] >> > (case_totals[3], startNonce, d_long_keys[thr_id], d_mid[thr_id], 
				i, case_pointer[b][0], d_case_totals[thr_id], case_pointer[c][0]);
		}

		if (case_totals[4] > 0) {

			dim3 grid4((case_totals[4] + threadsperblock - 1) / threadsperblock);
			verus_case_10 << <grid4, block, 0, streams[thr_id][5] >> > (case_totals[4], startNonce, d_long_keys[thr_id], d_mid[thr_id], 
				i, case_pointer[b][0], d_case_totals[thr_id], case_pointer[c][0]);
		}



		if (case_totals[6] > 0) {

			dim3 grid6((case_totals[6] + threadsperblock - 1) / threadsperblock);
			verus_case_18 << <grid6, block, 0, streams[thr_id][6] >> > (case_totals[6], startNonce, d_long_keys[thr_id], d_mid[thr_id], 
				i, case_pointer[b][0], d_case_totals[thr_id], case_pointer[c][0]);
		}

		if (case_totals[7] > 0) {

			dim3 grid7((case_totals[7] + threadsperblock - 1) / threadsperblock);
			verus_case_1c << <grid7, block, 0, streams[thr_id][7] >> > (case_totals[7], startNonce, d_long_keys[thr_id], d_mid[thr_id], 
				i, case_pointer[b][0], d_case_totals[thr_id], case_pointer[c][0]);
		}


		if (i < 31) 
			CUDA_SAFE_CALL(cudaMemcpy(case_totals, d_case_totals[thr_id], 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	}

	//	exit(0);
	//	verus_gpu_hash << <grid, block >> >(threads, startNonce, d_nonces[thr_id], d_long_keys[thr_id], d_mid[thr_id], d_fix_rand[thr_id], d_fix_randex[thr_id]);
	verus_gpu_final << <grid_final, 256 >> >(threads, startNonce, d_nonces[thr_id], d_long_keys[thr_id], d_mid[thr_id]);
	//verus_extra_gpu_fix << <grid_fix, 32 >> > (0, d_long_keys[thr_id], d_fix_rand[thr_id], d_fix_randex[thr_id]); //setup global mem with lots of keys	
	CUDA_SAFE_CALL(cudaMemcpy(resNonces, d_nonces[thr_id], 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	//exit(0);
};
__device__ __forceinline__
uint32_t xor3x(uint32_t a, uint32_t b, uint32_t c) {
	uint32_t result;
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
	asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result) : "r"(a), "r"(b), "r"(c)); //0x96 = 0xF0 ^ 0xCC ^ 0xAA
#else
	result = a^b^c;
#endif
	return result;
}

__device__  __forceinline__  uint4 _mm_xor_si128_emu(uint4 a, uint4 b)
{
	uint4 result;
	asm("xor.b32 %0, %1, %2; // xor1" : "=r"(result.x) : "r"(a.x), "r"(b.x));
	asm("xor.b32 %0, %1, %2; // xor1" : "=r"(result.y) : "r"(a.y), "r"(b.y));
	asm("xor.b32 %0, %1, %2; // xor1" : "=r"(result.z) : "r"(a.z), "r"(b.z));
	asm("xor.b32 %0, %1, %2; // xor1" : "=r"(result.w) : "r"(a.w), "r"(b.w));
	return result;


}


__device__  __forceinline__  uint32_t XT4(uint32_t b)
{
	uint32_t tmp1, tmp2, tmp3;

	tmp1 = (b << 1) & 0xfefefefe;
	tmp2 = (b >> 7) & 0x1010101;
	asm("mul.lo.u32 %0, %1, 27; ": "=r"(tmp3) : "r"(tmp2));
	asm("xor.b32 %0, %1, %2; // xor1" : "=r"(tmp2) : "r"(tmp1), "r"(tmp3));

	return tmp2;
}

__device__  uint4 _mm_clmulepi64_si128_emu(uint4 ai, uint4 bi)
{
	uint64_t a = ((uint64_t*)&ai)[0]; // (0xffffffffull & ai.x) | ((0x00000000ffffffffull & ai.y) << 32);//+ (imm & 1));

	uint64_t b = ((uint64_t*)&bi)[1]; // (0xffffffffull & bi.z) | ((0x00000000ffffffffull & bi.w) << 32);

									  //	uint8_t  i; 
									  //	uint2 u[8];
	uint64_t r[2]; //uint4 r;
				   //uint2 tmp;

	if (__popcll(a) > __popcll(b)) {

		a = b; b = ((uint64_t*)&ai)[0];
	}
	r[0] = 0; r[1] = 0;

	uint64_t w = a; int counter = 0; int first;

	while ((first = __clzll(w) + 1) != 65) {
		w <<= (first);
		counter += (first);

		r[0] ^= b << (64 - counter);
		r[1] ^= b >> ((counter));
	};

	return ((uint4*)&r)[0];
}

__device__  __forceinline__ uint4 _mm_clmulepi64_si128_emu2(uint4 ai)
{
	uint64_t a = ((uint64_t*)&ai)[1];

	//uint64_t b = 27 ;
	uint8_t  i; //window size s = 4,
				//uint64_t two_s = 16; //2^s
				//uint64_t smask = 15; //s 15 
	uint8_t u[8];
	uint4 r;
	uint64_t tmp;
	//Precomputation

	//#pragma unroll
	u[0] = 0;  //000 x b
	u[1] = 27;  //001 x b
	u[2] = 54; // u[1] << 1; //010 x b
	u[3] = 45;  //011 x b
	u[4] = 108; //100 x b
	u[5] = 119;  //101 x b
	u[6] = 90; //110 x b
	u[7] = 65;  //111 x b
				//Multiply
	((uint64_t*)&r)[0] = u[a & 7]; //first window only affects lower word

	r.z = r.w = 0;
	//#pragma unroll
	for (i = 3; i < 64; i += 3) {
		tmp = u[a >> i & 7];
		r.x ^= (tmp << i) & 0xffffffff;
		r.y ^= ((tmp << i) & 0xffffffff00000000) >> 32;
		//	((uint64_t*)&r)[1] ^= tmp >> (64 - i);
		r.z ^= (tmp >> (64 - i)) & 0xffffffff;
		r.w ^= ((tmp >> (64 - i)) & 0xffffffff00000000) >> 32;
	}

	return r;
}

#define _mm_load_si128_emu(p) (*(uint4*)(p));

#define _mm_cvtsi128_si64_emu(p) (((int64_t *)&p)[0]);

#define _mm_cvtsi128_si32_emu(p) (((int32_t *)&a)[0]);


__device__  void _mm_unpackboth_epi32_emu(uint4 &a, uint4 &b)
{
	uint64_t value;

	asm("mov.b64 %0, {%1, %2}; ": "=l"(value) : "r"(a.z), "r"(a.y));
	asm("mov.b64 {%0, %1}, %2; ": "=r"(a.y), "=r"(a.z) : "l"(value));

	asm("mov.b64 %0, {%1, %2}; ": "=l"(value) : "r"(b.x), "r"(a.y));
	asm("mov.b64 {%0, %1}, %2; ": "=r"(a.y), "=r"(b.x) : "l"(value));

	asm("mov.b64 %0, {%1, %2}; ": "=l"(value) : "r"(b.z), "r"(a.w));
	asm("mov.b64 {%0, %1}, %2; ": "=r"(a.w), "=r"(b.z) : "l"(value));

	asm("mov.b64 %0, {%1, %2}; ": "=l"(value) : "r"(b.y), "r"(a.w));
	asm("mov.b64 {%0, %1}, %2; ": "=r"(a.w), "=r"(b.y) : "l"(value));
}


__device__  __forceinline__ uint4 _mm_unpacklo_epi32_emu(uint4 a, uint4 b)
{

	//uint4 t;

	//	t.x = a.x;
	a.z = a.y;
	a.y = b.x;
	a.w = b.y;
	return a;
}

__device__  __forceinline__ uint4 _nop(uint4 a, uint4 b)
{

	return a;
}

__device__  __forceinline__ uint4 _mm_unpackhi_epi32_emu(uint4 a, uint4 b)
{

	//uint4 t;
	b.x = a.z;
	b.y = b.z;
	b.z = a.w;
	//t.w = b.w;

	return b;
}

__device__ uint32_t get_byte(uint32_t a, uint32_t start_bit)
{
	uint32_t result;
	asm("bfe.u32 %0, %1, %2, 8;" : "=r"(result) : "r"(a), "r"(start_bit));
	return result;
}

__device__   __forceinline__ void aesenc(uint4 * __restrict__ ptr, const uint4 * __restrict__ key, uint32_t * __restrict__ t)
{
	//#define XT(x) (((x) << 1) ^ (((x) >> 7) ? 0x1b : 0))

	//#define XT4(x) ((((x) << 1) & 0xfefefefe) ^ ((((x) >> 31) & 1) ? 0x1b000000 : 0)^ ((((x) >> 23)&1) ? 0x001b0000 : 0)^ ((((x) >> 15)&1) ? 0x00001b00 : 0)^ ((((x) >> 7)&1) ? 0x0000001b : 0))
	uint32_t x0 = ptr[0].x;
	uint32_t x1 = ptr[0].y;
	uint32_t x2 = ptr[0].z;
	uint32_t x3 = ptr[0].w;

	uint32_t y0 = t[get_byte(x0, 0)];
	uint32_t y1 = t[get_byte(x1, 0)];
	uint32_t y2 = t[get_byte(x2, 0)];
	uint32_t y3 = t[get_byte(x3, 0)];
	t += 256;

	y0 ^= t[get_byte(x1, 8)];
	y1 ^= t[get_byte(x2, 8)];
	y2 ^= t[get_byte(x3, 8)];
	y3 ^= t[get_byte(x0, 8)];
	t += 256;

	y0 ^= t[get_byte(x2, 16)];
	y1 ^= t[get_byte(x3, 16)];
	y2 ^= t[get_byte(x0, 16)];
	y3 ^= t[get_byte(x1, 16)];
	t += 256;

	y0 ^= t[get_byte(x3, 24)];
	y1 ^= t[get_byte(x0, 24)];
	y2 ^= t[get_byte(x1, 24)];
	y3 ^= t[get_byte(x2, 24)];

	ptr[0].x = y0 ^ key[0].x;
	ptr[0].y = y1 ^ key[0].y;
	ptr[0].z = y2 ^ key[0].z;
	ptr[0].w = y3 ^ key[0].w;

}


__device__  __forceinline__ uint4 _mm_cvtsi32_si128_emu(uint32_t lo)
{
	uint4 result = { 0 };
	result.x = lo;

	return result;
}
__device__  __forceinline__ uint4 _mm_cvtsi64_si128_emu(uint64_t lo)
{
	uint4 result = { 0 };
	((uint64_t *)&result)[0] = lo;
	//((uint64_t *)&result)[1] = 0;
	return result;
}
__device__  __forceinline__ uint4 _mm_set_epi64x_emu(uint64_t hi, uint64_t lo)
{
	uint4 result;
	((uint64_t *)&result)[0] = lo;
	((uint64_t *)&result)[1] = hi;
	return result;
}

__device__ __forceinline__ uint4 _mm_shuffle_epi8_emu(uint4 b)
{
	uint4 result = { 0 };
	const uint4 M = { 0x2d361b00,0x415a776c,0xf5eec3d8,0x9982afb4 };
	const uint4 Q = { 0x80808080, 0x80808080, 0x80808080, 0x80808080 };
	const uint4 W = b & Q;


#pragma unroll
	for (int i = 0; i < 8; i++)
	{
		if (!((uint8_t *)&W)[i])
		{
			((uint8_t *)&result)[i] = ((uint8_t *)&M)[((uint8_t *)&b)[i] & 0xf];
		}
	}

	return result;
}



__device__  __forceinline__ uint4 _mm_srli_si128_emu(uint4 input, int imm8)
{
	//we can cheat here as its an 8 byte shift just copy the 64bits
	uint4 temp;
	((uint64_t*)&temp)[0] = ((uint64_t*)&input)[1];
	((uint64_t*)&temp)[1] = 0;


	return temp;
}



__device__  uint4 _mm_mulhrs_epi16_emu(uint4 _a, uint4 _b)
{
	int16_t result[8];

	int32_t po;
	int16_t *a = (int16_t*)&_a, *b = (int16_t*)&_b;
#pragma nounroll
	for (int i = 0; i < 8; i++)
	{
		asm("mad.lo.s32 %0, %1, %2, 16384; ": "=r"(po) : "r"((int32_t)a[i]), "r"((int32_t)b[i]));

		result[i] = po >> 15;
		//	result[i] = (int16_t)((((int32_t)(a[i]) * (int32_t)(b[i])) + 0x4000) >> 15);

	}
	return *(uint4 *)result;
}


__device__   __forceinline__  uint32_t haraka512_port_keyed(uint4 * __restrict__  in, uint4 * __restrict__  rc, uint32_t * __restrict__  sharedMemory1)
{
	uint4 s1, s2, s3, s4, tmp;

	s1 = in[0];
	s2 = in[1];
	s3 = in[2];
	s4 = in[3];

	AES4(s1, s2, s3, s4, 0);
	MIX4(s1, s2, s3, s4);

	AES4(s1, s2, s3, s4, 8);
	MIX4(s1, s2, s3, s4);

	AES4(s1, s2, s3, s4, 16);
	MIX4(s1, s2, s3, s4);

	AES4(s1, s2, s3, s4, 24);
	MIX4_LASTBUT1(s1, s2, s3, s4);


	AES4_LAST(s3, 32);

	return s3.z ^ in[3].y;

}

__device__   __forceinline__ uint64_t precompReduction64(uint4 A) {


	//static const uint4 M = { 0x2d361b00,0x415a776c,0xf5eec3d8,0x9982afb4 };
	// const uint4 tmp = { 27 };
	// A.z = 0;
	//tmp.x = 27u;
	uint4 Q2 = _mm_clmulepi64_si128_emu2(A);
	uint4 Q3 = _mm_shuffle_epi8_emu(_mm_srli_si128_emu(Q2, 8));

	//uint4 Q4 = _mm_xor_si128_emu(Q2, A);
	uint4 final;
	final.x = xor3(A.x, Q2.x, Q3.x);
	final.y = xor3(A.y, Q2.y, Q3.y);

	return _mm_cvtsi128_si64_emu(final);/// WARNING: HIGH 64 BITS SHOULD BE ASSUMED TO CONTAIN GARBAGE
}

__global__ __launch_bounds__(THREADS, 1)
void verus_case_0(const uint32_t threads, const uint32_t startNonce, uint4 * __restrict__ d_key_input, uint4 * __restrict__ d_mid,
	 uint32_t iteration, main_case_s  d_main_case, uint32_t *totals, main_case_s  d_case_dest)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {
		uint4 s[4];
		s[0] = blockhash_half[0];
		s[1] = blockhash_half[1];
		s[2] = blockhash_half[2];
		s[3] = blockhash_half[3];

		uint32_t nonce = (iteration == 0) ? thread : d_main_case.case_array[0][thread]; //first iteration nonce is just 0..throughput
		uint4 acc = (iteration == 0) ? vkey[513] : d_mid[nonce];
		uint4 *randomsource = &d_key_input[VERUS_KEY_SIZE128 * nonce];
		s[2].x = nonce + startNonce;
		s[0] = s[0] ^ s[2];
		//s[1] = s[1] ^ s[3];
		// divide key mask by 32 from bytes to uint4
		uint16_t prand_idx, prandex_idx;
		uint64_t selector = _mm_cvtsi128_si64_emu(acc);
		uint4 const *pbuf = s + (selector & 3);
		uint4 prand;
		uint4 prandex;
		prand_idx = ((acc.x >> 5) & 511);
		prandex_idx = ((acc.y) & 511);

		prand = randomsource[prand_idx];
		prandex = randomsource[prandex_idx];
		//printf("iteration %d got past%d acc %08x  %08x  %08x\n",iteration, nonce, randomsource[123].x, prand.x, prandex.x);

		uint4 temp1 = prandex;

		uint4 temp2 = _mm_load_si128_emu(pbuf - (((selector & 1) << 1) - 1));
		uint4 add1 = _mm_xor_si128_emu(temp1, temp2);
		uint4 clprod1 = _mm_clmulepi64_si128_emu(add1, add1);
		acc = _mm_xor_si128_emu(clprod1, acc);

		uint4 tempa1 = _mm_mulhrs_epi16_emu(acc, temp1);
		uint4 tempa2 = _mm_xor_si128_emu(tempa1, temp1);

		uint4 temp12 = prand;
		prand = tempa2;

		uint4 temp22 = _mm_load_si128_emu(pbuf);
		uint4 add12 = _mm_xor_si128_emu(temp12, temp22);
		uint4 clprod12 = _mm_clmulepi64_si128_emu(add12, add12);
		acc = _mm_xor_si128_emu(clprod12, acc);

		uint4 tempb1 = _mm_mulhrs_epi16_emu(acc, temp12);
		uint4 tempb2 = _mm_xor_si128_emu(tempb1, temp12);
		prandex = tempb2;

		//  //d_fix_r[(nonce * 32) + iteration] = prand_idx;
		//d_fix_rex[(nonce * 32) + iteration] = prandex_idx;
		randomsource[prand_idx] = prand;
		randomsource[prandex_idx] = prandex;
		d_mid[nonce] = acc;
		selector = acc.x & 0x1c;
		selector >>= 2;
		uint32_t location = atomicAdd(&totals[selector], 1);
		d_case_dest.case_array[selector][location] = nonce;
	}

}


__global__ __launch_bounds__(THREADS, 1)
void verus_case_4(const uint32_t threads, const uint32_t startNonce, uint4 * __restrict__ d_key_input, uint4 * __restrict__ d_mid,
	 uint32_t iteration, main_case_s  d_main_case, uint32_t *totals, main_case_s  d_case_dest)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {
		uint4 s[4];
		s[0] = blockhash_half[0];
		s[1] = blockhash_half[1];
		s[2] = blockhash_half[2];
		s[3] = blockhash_half[3];


		uint32_t nonce = (iteration == 0) ? thread : d_main_case.case_array[1][thread]; //first iteration nonce is just 0..throughput
		uint4 acc = (iteration == 0) ? vkey[513] : d_mid[nonce];
		uint4 *randomsource = &d_key_input[VERUS_KEY_SIZE128 * nonce];
		s[2].x = nonce + startNonce;
		s[0] = s[0] ^ s[2];
		//s[1] = s[1] ^ s[3];
		// divide key mask by 32 from bytes to uint4

		uint16_t prand_idx, prandex_idx;
		uint64_t selector = _mm_cvtsi128_si64_emu(acc);
		uint4 const *pbuf = s + (selector & 3);
		uint4 prand;
		uint4 prandex;
		prand_idx = ((acc.x >> 5) & 511);
		prandex_idx = ((acc.y) & 511);

		prand = randomsource[prand_idx];
		prandex = randomsource[prandex_idx];

		uint4 temp1 = prand;
		uint4 temp2 = _mm_load_si128_emu(pbuf);
		uint4 add1 = _mm_xor_si128_emu(temp1, temp2);
		uint4 clprod1 = _mm_clmulepi64_si128_emu(add1, add1);
		acc = _mm_xor_si128_emu(clprod1, acc);
		uint4 clprod2 = _mm_clmulepi64_si128_emu(temp2, temp2);
		acc = _mm_xor_si128_emu(clprod2, acc);

		uint4 tempa1 = _mm_mulhrs_epi16_emu(acc, temp1);
		uint4 tempa2 = _mm_xor_si128_emu(tempa1, temp1);

		uint4 temp12 = prandex;
		prandex = tempa2;

		uint4 temp22 = _mm_load_si128_emu(pbuf - (((selector & 1) << 1) - 1));
		uint4 add12 = _mm_xor_si128_emu(temp12, temp22);
		acc = _mm_xor_si128_emu(add12, acc);

		uint4 tempb1 = _mm_mulhrs_epi16_emu(acc, temp12);
		uint4 tempb2 = _mm_xor_si128_emu(tempb1, temp12);
		prand = tempb2;

		//d_fix_r[(nonce * 32) + iteration] = prand_idx;
		//d_fix_rex[(nonce * 32) + iteration] = prandex_idx;
		randomsource[prand_idx] = prand;
		randomsource[prandex_idx] = prandex;
		d_mid[nonce] = acc;
		selector = acc.x & 0x1c;
		selector >>= 2;
		uint32_t location = atomicAdd(&totals[selector], 1);
		d_case_dest.case_array[selector][location] = nonce;
	}

}

__global__ __launch_bounds__(THREADS, 1)
void verus_case_8(const uint32_t threads, const uint32_t startNonce, uint4 * __restrict__ d_key_input, uint4 * __restrict__ d_mid,
	 uint32_t iteration, main_case_s  d_main_case, uint32_t *totals, main_case_s  d_case_dest)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {
		uint4 s[4];
		s[0] = blockhash_half[0];
		s[1] = blockhash_half[1];
		s[2] = blockhash_half[2];
		s[3] = blockhash_half[3];


		uint32_t nonce = (iteration == 0) ? thread : d_main_case.case_array[2][thread]; //first iteration nonce is just 0..throughput
		uint4 acc = (iteration == 0) ? vkey[513] : d_mid[nonce];
		uint4 *randomsource = &d_key_input[VERUS_KEY_SIZE128 * nonce];
		s[2].x = nonce + startNonce;
		s[0] = s[0] ^ s[2];
		//s[1] = s[1] ^ s[3];
		// divide key mask by 32 from bytes to uint4

		uint16_t prand_idx, prandex_idx;
		uint64_t selector = _mm_cvtsi128_si64_emu(acc);
		uint4 const *pbuf = s + (selector & 3);
		uint4 prand;
		uint4 prandex;
		prand_idx = ((acc.x >> 5) & 511);
		prandex_idx = ((acc.y) & 511);

		prand = randomsource[prand_idx];
		prandex = randomsource[prandex_idx];

		uint4 temp1 = prandex;
		uint4 temp2 = _mm_load_si128_emu(pbuf);
		uint4 add1 = _mm_xor_si128_emu(temp1, temp2);
		acc = _mm_xor_si128_emu(add1, acc);

		uint4 tempa1 = _mm_mulhrs_epi16_emu(acc, temp1);
		uint4 tempa2 = _mm_xor_si128_emu(tempa1, temp1);

		uint4 temp12 = prand;
		prand = tempa2;

		uint4 temp22 = _mm_load_si128_emu(pbuf - (((selector & 1) << 1) - 1));
		uint4 add12 = _mm_xor_si128_emu(temp12, temp22);
		uint4 clprod12 = _mm_clmulepi64_si128_emu(add12, add12);
		acc = _mm_xor_si128_emu(clprod12, acc);
		uint4 clprod22 = _mm_clmulepi64_si128_emu(temp22, temp22);
		acc = _mm_xor_si128_emu(clprod22, acc);

		uint4 tempb1 = _mm_mulhrs_epi16_emu(acc, temp12);
		uint4 tempb2 = _mm_xor_si128_emu(tempb1, temp12);
		prandex = tempb2;

		//d_fix_r[(nonce * 32) + iteration] = prand_idx;
		//d_fix_rex[(nonce * 32) + iteration] = prandex_idx;
		randomsource[prand_idx] = prand;
		randomsource[prandex_idx] = prandex;
		d_mid[nonce] = acc;
		selector = acc.x & 0x1c;
		selector >>= 2;
		uint32_t location = atomicAdd(&totals[selector], 1);
		d_case_dest.case_array[selector][location] = nonce;
	}

}

__global__ __launch_bounds__(THREADS, 1)
void verus_case_c(const uint32_t threads, const uint32_t startNonce, uint4 * __restrict__ d_key_input, uint4 * __restrict__ d_mid,
	 uint32_t iteration, main_case_s  d_main_case, uint32_t *totals, main_case_s  d_case_dest)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {
		uint4 s[4];
		s[0] = blockhash_half[0];
		s[1] = blockhash_half[1];
		s[2] = blockhash_half[2];
		s[3] = blockhash_half[3];


		uint32_t nonce = (iteration == 0) ? thread : d_main_case.case_array[3][thread]; //first iteration nonce is just 0..throughput
		uint4 acc = (iteration == 0) ? vkey[513] : d_mid[nonce];
		uint4 *randomsource = &d_key_input[VERUS_KEY_SIZE128 * nonce];
		s[2].x = nonce + startNonce;
		s[0] = s[0] ^ s[2];
		//s[1] = s[1] ^ s[3];
		// divide key mask by 32 from bytes to uint4

		uint16_t prand_idx, prandex_idx;
		uint64_t selector = _mm_cvtsi128_si64_emu(acc);
		uint4 const *pbuf = s + (selector & 3);
		uint4 prand;
		uint4 prandex;
		prand_idx = ((acc.x >> 5) & 511);
		prandex_idx = ((acc.y) & 511);

		prand = randomsource[prand_idx];
		prandex = randomsource[prandex_idx];
		//if (nonce == 1)
		//	printf("[i]=%d \t acc = %08x, prand_idx = %d\t, prandex_idx = %d \t prand %08x, prandex %08x case 3\n", iteration, acc.x, prand_idx, prandex_idx, prand.x, prandex.x);

		const uint4 temp1 = prand;
		const uint4 temp2 = _mm_load_si128_emu(pbuf - (((selector & 1) << 1) - 1));
		const uint4 add1 = _mm_xor_si128_emu(temp1, temp2);

		// cannot be zero here
		const int32_t divisor = ((uint32_t*)&selector)[0];

		acc = _mm_xor_si128_emu(add1, acc);

		int64_t dividend = _mm_cvtsi128_si64_emu(acc);
		int64_t tmpmod = dividend % divisor;
		const uint4 modulo = _mm_cvtsi32_si128_emu(tmpmod);
		acc = _mm_xor_si128_emu(modulo, acc);

		const uint4 tempa1 = _mm_mulhrs_epi16_emu(acc, temp1);
		const uint4 tempa2 = _mm_xor_si128_emu(tempa1, temp1);
		dividend &= 1;
		if (dividend)
		{
			const uint4 temp12 = prandex;
			prandex = tempa2;

			const uint4 temp22 = _mm_load_si128_emu(pbuf);
			const uint4 add12 = _mm_xor_si128_emu(temp12, temp22);
			const uint4 clprod12 = _mm_clmulepi64_si128_emu(add12, add12);
			acc = _mm_xor_si128_emu(clprod12, acc);
			const uint4 clprod22 = _mm_clmulepi64_si128_emu(temp22, temp22);
			acc = _mm_xor_si128_emu(clprod22, acc);

			const uint4 tempb1 = _mm_mulhrs_epi16_emu(acc, temp12);
			const uint4 tempb2 = _mm_xor_si128_emu(tempb1, temp12);
			prand = tempb2;
		}
		else
		{
			const uint4 tempb3 = prandex;
			prandex = tempa2;
			prand = tempb3;
		}

		//d_fix_r[(nonce * 32) + iteration] = prand_idx;
		//d_fix_rex[(nonce * 32) + iteration] = prandex_idx;
		randomsource[prand_idx] = prand;
		randomsource[prandex_idx] = prandex;
		d_mid[nonce] = acc;
		selector = acc.x & 0x1c;
		selector >>= 2;
		uint32_t location = atomicAdd(&totals[selector], 1);
		d_case_dest.case_array[selector][location] = nonce;

		//	if (nonce == 0)
		//		printf("[i]=%d \t acc = %08x, prand_idx = %d\t, prandex_idx = %d \t prand %08x, prandex %08x case 3\n", iteration, acc.x, prand_idx, prandex_idx, prand.x, prandex.x);


	}

}

__global__ __launch_bounds__(THREADS, 1)
void verus_case_10(const uint32_t threads, const uint32_t startNonce, uint4 * __restrict__ d_key_input, uint4 * __restrict__ d_mid,
	 uint32_t iteration, main_case_s d_main_case, uint32_t *totals, main_case_s  d_case_dest)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	__shared__ uint32_t shared_m[4][256];

	for (int i = threadIdx.x; i < 256; i += blockDim.x) {

		shared_m[0][i] = saes_table[0][i];
		shared_m[1][i] = saes_table[1][i];
		shared_m[2][i] = saes_table[2][i];
		shared_m[3][i] = saes_table[3][i];
	}
	__syncthreads();

	uint32_t *sharedMemory1 = shared_m[0];

	if (thread < threads) {
		uint4 s[4];
		s[0] = blockhash_half[0];
		s[1] = blockhash_half[1];
		s[2] = blockhash_half[2];
		s[3] = blockhash_half[3];


		uint32_t nonce = (iteration == 0) ? thread : d_main_case.case_array[4][thread]; //first iteration nonce is just 0..throughput
		uint4 acc = (iteration == 0) ? vkey[513] : d_mid[nonce];
		uint4 *randomsource = &d_key_input[VERUS_KEY_SIZE128 * nonce];
		s[2].x = nonce + startNonce;
		s[0] = s[0] ^ s[2];
		//s[1] = s[1] ^ s[3];
		// divide key mask by 32 from bytes to uint4

		uint16_t prand_idx, prandex_idx;
		uint64_t selector = _mm_cvtsi128_si64_emu(acc);
		uint4 const *pbuf = s + (selector & 3);
		uint4 prand;
		uint4 prandex;
		prand_idx = ((acc.x >> 5) & 511);
		prandex_idx = ((acc.y) & 511);

		prand = randomsource[prand_idx];
		prandex = randomsource[prandex_idx];

		uint4 *rc = &randomsource[prand_idx];

		uint4 tmp;

		uint4 temp1 = _mm_load_si128_emu(pbuf - (((selector & 1) << 1) - 1));
		uint4 temp2 = _mm_load_si128_emu(pbuf);

		AES2_EMU(temp1, temp2, 0);
		MIX2_EMU(temp1, temp2);


		AES2_EMU(temp1, temp2, 4);
		MIX2_EMU(temp1, temp2);

		AES2_EMU(temp1, temp2, 8);
		MIX2_EMU(temp1, temp2);


		acc = _mm_xor_si128_emu(temp1, acc);
		acc = _mm_xor_si128_emu(temp2, acc);

		const uint4 tempa1 = prand;
		const uint4 tempa2 = _mm_mulhrs_epi16_emu(acc, tempa1); selector = _mm_cvtsi128_si64_emu(acc);
		const uint4 tempa3 = _mm_xor_si128_emu(tempa1, tempa2);

		const uint4 tempa4 = prandex;
		prandex = tempa3;
		prand = tempa4;

		//d_fix_r[(nonce * 32) + iteration] = prand_idx;
		//d_fix_rex[(nonce * 32) + iteration] = prandex_idx;
		randomsource[prand_idx] = prand;
		randomsource[prandex_idx] = prandex;
		d_mid[nonce] = acc;
		selector = acc.x & 0x1c;
		selector >>= 2;
		uint32_t location = atomicAdd(&totals[selector], 1);
		d_case_dest.case_array[selector][location] = nonce;
	}

}

__global__ __launch_bounds__(THREADS, 1)
void verus_case_14(const uint32_t threads, const uint32_t startNonce, uint4 * __restrict__ d_key_input, uint4 * __restrict__ d_mid,
	 uint32_t iteration, main_case_s  d_main_case, uint32_t *totals, main_case_s  d_case_dest)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	__shared__ uint32_t shared_m[4][256];

	for (int i = threadIdx.x; i < 256; i += blockDim.x) {

		shared_m[0][i] = saes_table[0][i];
		shared_m[1][i] = saes_table[1][i];
		shared_m[2][i] = saes_table[2][i];
		shared_m[3][i] = saes_table[3][i];
	}
	__syncthreads();

	uint32_t *sharedMemory1 = shared_m[0];

	if (thread < threads) {
		uint4 s[4];
		s[0] = blockhash_half[0];
		s[1] = blockhash_half[1];
		s[2] = blockhash_half[2];
		s[3] = blockhash_half[3];


		uint32_t nonce = (iteration == 0) ? thread : d_main_case.case_array[5][thread]; //first iteration nonce is just 0..throughput
		uint4 acc = (iteration == 0) ? vkey[513] : d_mid[nonce];
		uint4 *randomsource = &d_key_input[VERUS_KEY_SIZE128 * nonce];
		s[2].x = nonce + startNonce;
		s[0] = s[0] ^ s[2];
		//s[1] = s[1] ^ s[3];
		// divide key mask by 32 from bytes to uint4

		uint16_t prand_idx, prandex_idx;
		uint64_t selector = _mm_cvtsi128_si64_emu(acc);
		uint4 const *pbuf = s + (selector & 3);
		uint4 prand;
		uint4 prandex;
		prand_idx = ((acc.x >> 5) & 511);
		prandex_idx = ((acc.y) & 511);

		prand = randomsource[prand_idx];
		prandex = randomsource[prandex_idx];

		// we'll just call this one the monkins loop, inspired by Chris
		const uint4 *buftmp = pbuf - (((selector & 1) << 1) - 1);
		uint4 tmp; // used by MIX2

		uint64_t rounds = selector >> 61; // loop randomly between 1 and 8 times
		uint4 *rc = &randomsource[prand_idx];


		uint64_t aesround = 0;
		uint4 onekey;
		uint64_t loop_c;


		for (int i = 0; i<8; i++)
		{
			if (rounds <= 8) {
				loop_c = selector & ((uint64_t)0x10000000 << rounds);
				if (loop_c)
				{
					onekey = _mm_load_si128_emu(rc++);
					const uint4 temp2 = _mm_load_si128_emu(rounds & 1 ? pbuf : buftmp);
					const uint4 add1 = _mm_xor_si128_emu(onekey, temp2);
					const uint4 clprod1 = _mm_clmulepi64_si128_emu(add1, add1);
					acc = _mm_xor_si128_emu(clprod1, acc);
				}
				else
				{
					onekey = _mm_load_si128_emu(rc++);
					uint4 temp2 = _mm_load_si128_emu(rounds & 1 ? buftmp : pbuf);

					const uint64_t roundidx = aesround++ << 2;
					AES2_EMU(onekey, temp2, roundidx);

					MIX2_EMU(onekey, temp2);

					acc = _mm_xor_si128_emu(onekey, acc);
					acc = _mm_xor_si128_emu(temp2, acc);

				}

				(rounds--);
			}
		}

		const uint4 tempa1 = (prand);
		const uint4 tempa2 = _mm_mulhrs_epi16_emu(acc, tempa1);
		const uint4 tempa3 = _mm_xor_si128_emu(tempa1, tempa2);

		const uint4 tempa4 = (prandex);
		prandex = tempa3;
		prand = tempa4;

		//d_fix_r[(nonce * 32) + iteration] = prand_idx;
		//d_fix_rex[(nonce * 32) + iteration] = prandex_idx;
		randomsource[prand_idx] = prand;
		randomsource[prandex_idx] = prandex;
		d_mid[nonce] = acc;
		selector = acc.x & 0x1c;
		selector >>= 2;
		uint32_t location = atomicAdd(&totals[selector], 1);
		d_case_dest.case_array[selector][location] = nonce;
	}

}

__global__ __launch_bounds__(THREADS, 1)
void verus_case_18(const uint32_t threads, const uint32_t startNonce, uint4 * __restrict__ d_key_input, uint4 * __restrict__ d_mid,
	 uint32_t iteration, main_case_s  d_main_case, uint32_t *totals, main_case_s  d_case_dest)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {
		uint4 s[4];
		s[0] = blockhash_half[0];
		s[1] = blockhash_half[1];
		s[2] = blockhash_half[2];
		s[3] = blockhash_half[3];


		uint32_t nonce = (iteration == 0) ? thread : d_main_case.case_array[6][thread]; //first iteration nonce is just 0..throughput
		uint4 acc = (iteration == 0) ? vkey[513] : d_mid[nonce];
		uint4 *randomsource = &d_key_input[VERUS_KEY_SIZE128 * nonce];
		s[2].x = nonce + startNonce;
		s[0] = s[0] ^ s[2];
		//s[1] = s[1] ^ s[3];
		// divide key mask by 32 from bytes to uint4

		uint16_t prand_idx, prandex_idx;
		uint64_t selector = _mm_cvtsi128_si64_emu(acc);
		uint4 const *pbuf = s + (selector & 3);
		uint4 prand;
		uint4 prandex;
		prand_idx = ((acc.x >> 5) & 511);
		prandex_idx = ((acc.y) & 511);

		prand = randomsource[prand_idx];
		prandex = randomsource[prandex_idx];

		// we'll just call this one the monkins loop, inspired by Chris
		const uint4 *buftmp = pbuf - (((selector & 1) << 1) - 1);


		uint64_t rounds = selector >> 61; // loop randomly between 1 and 8 times
		uint4 *rc = &randomsource[prand_idx];

		uint4 onekey;
		uint64_t loop_c;

		for (int i = 0; i<8; i++)
		{
			if (rounds <= 8) {
				loop_c = selector & ((uint64_t)0x10000000 << rounds);
				if (loop_c)
				{
					onekey = _mm_load_si128_emu(rc++);
					const uint4 temp2 = _mm_load_si128_emu(rounds & 1 ? pbuf : buftmp);
					const uint4 add1 = _mm_xor_si128_emu(onekey, temp2);

					const int32_t divisor = (uint32_t)selector;
					const int64_t dividend = ((int64_t*)&add1)[0];
					uint4 modulo = { 0 }; ((int32_t*)&modulo)[0] = (dividend % divisor);
					acc = modulo ^ acc;

				}
				else
				{
					onekey = _mm_load_si128_emu(rc++);
					uint4 temp2 = _mm_load_si128_emu(rounds & 1 ? buftmp : pbuf);
					uint4 add1 = (onekey^ temp2);
					uint4 clprod1 = _mm_clmulepi64_si128_emu(add1, add1);
					uint4 clprod2 = _mm_mulhrs_epi16_emu(acc, clprod1);
					acc = clprod2^ acc;
				}
			}
			(rounds--);
		}

		const uint4 tempa3 = (prandex);
		const uint4 tempa4 = _mm_xor_si128_emu(tempa3, acc);
		prandex = tempa4;
		prand = onekey;

		//d_fix_r[(nonce * 32) + iteration] = prand_idx;
		//d_fix_rex[(nonce * 32) + iteration] = prandex_idx;
		randomsource[prand_idx] = prand;
		randomsource[prandex_idx] = prandex;
		d_mid[nonce] = acc;
		selector = acc.x & 0x1c;
		selector >>= 2;
		uint32_t location = atomicAdd(&totals[selector], 1);
		d_case_dest.case_array[selector][location] = nonce;
	}

}


__global__ __launch_bounds__(THREADS, 1)
void verus_case_1c(const uint32_t threads, const uint32_t startNonce, uint4 * __restrict__ d_key_input, uint4 * __restrict__ d_mid,
	 uint32_t iteration, main_case_s  d_main_case, uint32_t *totals, main_case_s  d_case_dest)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {
		uint4 s[4];
		s[0] = blockhash_half[0];
		s[1] = blockhash_half[1];
		s[2] = blockhash_half[2];
		s[3] = blockhash_half[3];

		uint32_t nonce = (iteration == 0) ? thread : d_main_case.case_array[7][thread]; //first iteration nonce is just 0..throughput
		uint4 acc = (iteration == 0) ? vkey[513] : d_mid[nonce];
		uint4 *randomsource = &d_key_input[VERUS_KEY_SIZE128 * nonce];
		s[2].x = nonce + startNonce;
		s[0] = s[0] ^ s[2];
		//s[1] = s[1] ^ s[3];
		// divide key mask by 32 from bytes to uint4



		uint16_t prand_idx, prandex_idx;
		uint64_t selector = _mm_cvtsi128_si64_emu(acc);
		uint4 const *pbuf = s + (selector & 3);
		uint4 prand;
		uint4 prandex;
		prand_idx = ((acc.x >> 5) & 511);
		prandex_idx = ((acc.y) & 511);

		prand = randomsource[prand_idx];
		prandex = randomsource[prandex_idx];
		//	if(nonce ==1)
		//		printf("[i]=%d \t acc = %08x, prand_idx = %d\t, prandex_idx = %d \t prand %08x, prandex %08x case 7\n", iteration, acc.x, prand_idx, prandex_idx, prand.x, prandex.x);

		//	if (nonce == 0)
		//		printf("pbuf %08x%08x%08x%08x\n",s[0].x, s[1].x, s[2].x, s[3].x);

		const uint4 temp1 = _mm_load_si128_emu(pbuf);
		const uint4 temp2 = (prandex);
		const uint4 add1 = _mm_xor_si128_emu(temp1, temp2);
		const uint4 clprod1 = _mm_clmulepi64_si128_emu(add1, add1);
		acc = _mm_xor_si128_emu(clprod1, acc);
		//	if (nonce == 0)
		//		printf("acc = %08x",acc.x);


		const uint4 tempa1 = _mm_mulhrs_epi16_emu(acc, temp2);
		const uint4 tempa2 = _mm_xor_si128_emu(tempa1, temp2);
		const uint4 tempa3 = (prand);


		prand = tempa2;

		acc = _mm_xor_si128_emu(tempa3, acc);
		//	if (nonce == 0)
		//		printf("[i]=%d \t acc = %08x, prand_idx = %d\t, prandex_idx = %d \t prand %08x, prandex %08x\n", iteration, acc.x, prand_idx, prandex_idx, prand.x, prandex.x);

		const uint4 tempb1 = _mm_mulhrs_epi16_emu(acc, tempa3);
		const uint4 tempb2 = _mm_xor_si128_emu(tempb1, tempa3);
		prandex = tempb2;

		//d_fix_r[(nonce * 32) + iteration] = prand_idx;
		//d_fix_rex[(nonce * 32) + iteration] = prandex_idx;
		randomsource[prand_idx] = prand;
		randomsource[prandex_idx] = prandex;
		d_mid[nonce] = acc;
		selector = acc.x & 0x1c;
		selector >>= 2;
		uint32_t location = atomicAdd(&totals[selector], 1);
		d_case_dest.case_array[selector][location] = nonce;

		//	if (nonce == 0)
		//	printf("[i]=%d \t acc = %08x, prand_idx = %d\t, prandex_idx = %d \t prand %08x, prandex %08x\n", iteration, acc.x, prand_idx, prandex_idx, prand.x, prandex.x);

	}

}

__global__ __launch_bounds__(256, 1)
void verus_gpu_final(const uint32_t threads, const uint32_t startNonce, uint32_t * __restrict__ resNonce,
	uint4 * __restrict__  d_key_input, const uint4 * __restrict__ d_mid)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint4 mid = d_mid[thread];
	mid.x ^= 0x00010000;

	uint64_t acc = precompReduction64(mid);
	//uint4 wizz = d_mid[thread];

	const uint32_t nounce = startNonce + thread;
	uint32_t hash;

	uint4 s[4];
	__shared__ uint32_t sharedMemory1[4][256];
	//	sharedMemory1[threadIdx.x] = sbox[threadIdx.x];// copy sbox to shared mem
	sharedMemory1[0][threadIdx.x] = saes_table[0][threadIdx.x];// copy sbox to shared mem

	sharedMemory1[1][threadIdx.x] = saes_table[1][threadIdx.x];// copy sbox to shared mem

	sharedMemory1[2][threadIdx.x] = saes_table[2][threadIdx.x];// copy sbox to shared mem


	sharedMemory1[3][threadIdx.x] = saes_table[3][threadIdx.x];// copy sbox to shared mem



	s[0] = blockhash_half2[0];
	s[1] = blockhash_half2[1];
	s[2] = blockhash_half2[2];
	s[3] = blockhash_half2[3];
	__syncthreads();

	((uint32_t *)&s)[8] = nounce;
	memcpy(((uint8_t*)&s) + 47, &acc, 8);
	memcpy(((uint8_t*)&s) + 55, &acc, 8);
	memcpy(((uint8_t*)&s) + 63, &acc, 1);
	//uint64_t mask = 8191 >> 4;
	acc &= 511;


	hash = haraka512_port_keyed(s, (&d_key_input[VERUS_KEY_SIZE128 * thread] + acc), sharedMemory1[0]);
	if (hash < ptarget[7]) {

		resNonce[0] = nounce;

	}


};

__global__ __launch_bounds__(128, 1)
void verus_extra_gpu_prepare(const uint32_t threads, uint4 * d_key_input)
{

	d_key_input[(blockIdx.x * VERUS_KEY_SIZE128) + threadIdx.x] = vkey[threadIdx.x];
	d_key_input[(blockIdx.x * VERUS_KEY_SIZE128) + threadIdx.x + 128] = vkey[threadIdx.x + 128];
	d_key_input[(blockIdx.x * VERUS_KEY_SIZE128) + threadIdx.x + 256] = vkey[threadIdx.x + 256];
	d_key_input[(blockIdx.x * VERUS_KEY_SIZE128) + threadIdx.x + 384] = vkey[threadIdx.x + 384];
	if (threadIdx.x < 40)
		d_key_input[(blockIdx.x * VERUS_KEY_SIZE128) + threadIdx.x + 512] = vkey[threadIdx.x + 512];

}

