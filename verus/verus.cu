
#include <miner.h>
extern "C" {
#include <stdint.h>
#include <memory.h>
}
#define HARAKAS_RATE 32
#include <cuda_helper.h>
#define NPT 2
#define NBN 2
__device__  uint32_t sbox[64] =
{ 0x7b777c63, 0xc56f6bf2, 0x2b670130, 0x76abd7fe, 0x7dc982ca, 0xf04759fa, 0xafa2d4ad, 0xc072a49c, 0x2693fdb7, 0xccf73f36, 0xf1e5a534, 0x1531d871, 0xc323c704, 0x9a059618, 0xe2801207, 0x75b227eb, 0x1a2c8309, 0xa05a6e1b, 0xb3d63b52, 0x842fe329, 0xed00d153, 0x5bb1fc20, 0x39becb6a, 0xcf584c4a, 0xfbaaefd0, 0x85334d43, 0x7f02f945, 0xa89f3c50, 0x8f40a351, 0xf5389d92, 0x21dab6bc, 0xd2f3ff10, 0xec130ccd, 0x1744975f, 0x3d7ea7c4, 0x73195d64, 0xdc4f8160, 0x88902a22, 0x14b8ee46, 0xdb0b5ede, 0x0a3a32e0, 0x5c240649, 0x62acd3c2, 0x79e49591, 0x6d37c8e7, 0xa94ed58d, 0xeaf4566c, 0x08ae7a65, 0x2e2578ba, 0xc6b4a61c, 0x1f74dde8, 0x8a8bbd4b, 0x66b53e70, 0x0ef60348, 0xb9573561, 0x9e1dc186, 0x1198f8e1, 0x948ed969, 0xe9871e9b, 0xdf2855ce, 0x0d89a18c, 0x6842e6bf, 0x0f2d9941, 0x16bb54b0 };
#define XT(x) (((x) << 1) ^ ((((x) >> 7) & 1) * 0x1b))
__global__ void verus_gpu_hash(uint32_t threads, uint32_t startNonce, uint32_t *resNonce);
__device__ void haraka512_perm(unsigned char *out, unsigned char *in);
static uint32_t *d_nonces[MAX_GPUS];
__constant__ uint8_t blockhash_half[128];
__constant__ uint32_t ptarget[8];

__device__   void memcpy_decker(unsigned char *dst, unsigned char *src, int len) {
	int i;
	for (i = 0; i< len; i++) { dst[i] = src[i]; }
}

__host__
void verus_init(int thr_id)
{
CUDA_SAFE_CALL(cudaMalloc(&d_nonces[thr_id], 2 * sizeof(uint32_t)));
};
void verus_setBlock(void *blockf, const void *pTargetIn)
{
CUDA_SAFE_CALL(cudaMemcpyToSymbol(ptarget, pTargetIn, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
CUDA_SAFE_CALL(cudaMemcpyToSymbol(blockhash_half, blockf, 64 * sizeof(uint8_t), 0, cudaMemcpyHostToDevice));
};

__host__
void verus_hash(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *resNonces)
{
	cudaMemset(d_nonces[thr_id], 0xff, 2 * sizeof(uint32_t));
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	verus_gpu_hash << <grid, block >> >(threads, startNonce, d_nonces[thr_id]);
	//cudaThreadSynchronize();
	cudaMemcpy(resNonces, d_nonces[thr_id], NBN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	//memcpy(resNonces, h_nonces[thr_id], NBN * sizeof(uint32_t));

};



//__constant__ static const

// Simulate _mm_aesenc_si128 instructions from AESNI
__device__   void aesenc(unsigned char *s,uint32_t *sharedMemory1)
{
	uint32_t i, t, u;
	register uint32_t  v[4][4];
	
	for (i = 0; i < 16; ++i) {
		v[((i >> 2) + 4 - (i & 3)) & 3][i & 3] = ((uint8_t*)&sharedMemory1[0])[s[i]];
	}

	for (i = 0; i < 4; ++i) {
		t = v[i][0];
		u = v[i][0] ^ v[i][1] ^ v[i][2] ^ v[i][3];
		v[i][0] = v[i][0] ^ u ^ XT(v[i][0] ^ v[i][1]);
		v[i][1] = v[i][1] ^ u ^ XT(v[i][1] ^ v[i][2]);
		v[i][2] = v[i][2] ^ u ^ XT(v[i][2] ^ v[i][3]);
		v[i][3] = v[i][3] ^ u ^ XT(v[i][3] ^ t);
	}
	for (i = 0; i < 16; ++i) {
		s[i] = (unsigned char)v[i >> 2][i & 3]; // VerusHash have 0 rc vector
	}
}

// Simulate _mm_unpacklo_epi32
__device__ __forceinline__   void unpacklo32(unsigned char *t, unsigned char *a, unsigned char *b)
{
	unsigned char tmp[16];
	memcpy_decker(tmp, a, 4);
	memcpy_decker(tmp + 4, b, 4);
	memcpy_decker(tmp + 8, a + 4, 4);
	memcpy_decker(tmp + 12, b + 4, 4);
	memcpy_decker(t, tmp, 16);
}

// Simulate _mm_unpackhi_epi32
__device__  __forceinline__  void unpackhi32(unsigned char *t, unsigned char *a, unsigned char *b)
{
	unsigned char tmp[16];
	memcpy_decker(tmp, a + 8, 4);
	memcpy_decker(tmp + 4, b + 8, 4);
	memcpy_decker(tmp + 8, a + 12, 4);
	memcpy_decker(tmp + 12, b + 12, 4);
	memcpy_decker(t, tmp, 16);

}


__global__ __launch_bounds__(256, 2)
void verus_gpu_hash(uint32_t threads, uint32_t startNonce, uint32_t *resNonce)
{
	uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread > threads)
		asm("exit;");
	int i, j; unsigned char s[64], tmp[16];
		__shared__ uint32_t sharedMemory1[64];
	if (threadIdx.x < 64)
		sharedMemory1[threadIdx.x] = sbox[threadIdx.x];//	for (i = 0; i < 64; ++i)
					

		uint32_t nounce = startNonce + thread;
		unsigned char in[64];
		
		uint64_t blockhash[4];
		memcpy(s, blockhash_half, 32);
		memset(s + 32, 0x0, 32);
		((uint32_t *)&s)[8] = nounce;
		memcpy(in +48, s + 48, 8);
		//memcpy_decker(s, in, 64);
    #pragma unroll 
		for (i = 0; i < 5; ++i) {
			// aes round(s)
			//__syncthreads();
			for (j = 0; j < 2; ++j) {

				aesenc(s, sharedMemory1);
				aesenc(s + 16, sharedMemory1);
				aesenc(s + 32, sharedMemory1);
				aesenc(s + 48, sharedMemory1);
			}
			unpacklo32(tmp, s, s + 16);
			unpackhi32(s, s, s + 16);
			unpacklo32(s + 16, s + 32, s + 48);
			unpackhi32(s + 32, s + 32, s + 48);
			unpacklo32(s + 48, s, s + 32);
			unpackhi32(s, s, s + 32);
			unpackhi32(s + 32, s + 16, tmp);
			unpacklo32(s + 16, s + 16, tmp);

		}
		for (i = 48; i < 56; i++) {
			s[i] = s[i] ^ in[i];
		}

		memcpy_decker((unsigned char*)blockhash + 24, s + 48, 8);
		
		

		if (blockhash[3] < ((uint64_t*)&ptarget)[3]) { resNonce[0] = nounce; }
	
};


