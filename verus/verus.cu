
#include <miner.h>
extern "C" {
#include <stdint.h>
#include <memory.h>
}
#define HARAKAS_RATE 32

#include <cuda_helper.h>

#define NPT 2
#define NBN 2


__global__ void verus_gpu_hash(uint32_t threads, uint32_t startNonce, uint32_t *resNonce);

__device__ void haraka512_full(unsigned char *out, const unsigned char *in);
__device__ void haraka512_perm(unsigned char *out, const unsigned char *in);
	

static uint32_t *d_nonces[MAX_GPUS];

__constant__ uint8_t blockhash_half[128];
__constant__ uint32_t ptarget[8];

__host__
void verus_init(int thr_id)
{
	
	CUDA_SAFE_CALL(cudaMalloc(&d_nonces[thr_id], 2*sizeof(uint32_t)));
   
};


void verus_setBlock(void *blockf,const void *pTargetIn) 
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(ptarget, pTargetIn, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
 	CUDA_SAFE_CALL(cudaMemcpyToSymbol(blockhash_half, blockf, 64*sizeof(uint8_t), 0, cudaMemcpyHostToDevice));
	
        
};
__host__ 
void verus_hash(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *resNonces)
{
	cudaMemset(d_nonces[thr_id], 0xff, 2 * sizeof(uint32_t));
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	verus_gpu_hash<<<grid, block>>>(threads, startNonce, d_nonces[thr_id]);
	cudaThreadSynchronize();
	cudaMemcpy(resNonces, d_nonces[thr_id], 2*sizeof(uint32_t), cudaMemcpyDeviceToHost);

	
 
};
__global__ 
void verus_gpu_hash(uint32_t threads, uint32_t startNonce, uint32_t *resNonce)
{
	

	uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread < threads)
	{uint32_t nounce = startNonce + thread;

    uint8_t hash_buf[64];
    uint8_t blockhash[64];
    
    memcpy(hash_buf,blockhash_half,64);
    memset(hash_buf + 32, 0x0,32);
    //memcpy(hash_buf + 32, (unsigned char *)&full_data + 1486 - 14, 15);
    ((uint32_t *)&hash_buf)[8] = nounce;
  
    
    haraka512_full((unsigned char*)blockhash, (unsigned char*)hash_buf); // ( out, in)

		if (((uint64_t*)&blockhash)[3] < ((uint64_t*)&ptarget)[3]) { resNonce[0] = nounce;}   
    }
};

__device__ void memcpy_decker(unsigned char *dst, unsigned char *src, int len) {
    int i;
    for (i=0; i<len; i++) { dst[i] = src[i]; }
}




//__constant__ static const
__device__  unsigned char sbox[256] =
{ 0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe,
  0xd7, 0xab, 0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4,
  0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7,
  0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, 0x04, 0xc7, 0x23, 0xc3,
  0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, 0x09,
  0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3,
  0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe,
  0x39, 0x4a, 0x4c, 0x58, 0xcf, 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
  0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92,
  0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c,
  0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19,
  0x73, 0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14,
  0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2,
  0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5,
  0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, 0xba, 0x78, 0x25,
  0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
  0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86,
  0xc1, 0x1d, 0x9e, 0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e,
  0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf, 0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42,
  0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16 };

#define XT(x) (((x) << 1) ^ ((((x) >> 7) & 1) * 0x1b))

// Simulate _mm_aesenc_si128 instructions from AESNI
__device__  void aesenc(unsigned char *s,const unsigned char sharedMemory1[256])
{
    unsigned char i, t, u, v[4][4];
    for (i = 0; i < 16; ++i) {
        v[((i / 4) + 4 - (i%4) ) % 4][i % 4] = sharedMemory1[s[i]];
    }
    for (i = 0; i < 4; ++i) {
        t = v[i][0];
        u = v[i][0] ^ v[i][1] ^ v[i][2] ^ v[i][3];
        v[i][0] ^= u ^ XT(v[i][0] ^ v[i][1]);
        v[i][1] ^= u ^ XT(v[i][1] ^ v[i][2]);
        v[i][2] ^= u ^ XT(v[i][2] ^ v[i][3]);
        v[i][3] ^= u ^ XT(v[i][3] ^ t);
    }
    for (i = 0; i < 16; ++i) {
        s[i] = v[i / 4][i % 4]; // VerusHash have 0 rc vector
    }
}

// Simulate _mm_unpacklo_epi32
__device__ __forceinline__ void unpacklo32(unsigned char *t, unsigned char *a, unsigned char *b)
{
    unsigned char tmp[16];
    memcpy_decker(tmp, a, 4);
    memcpy_decker(tmp + 4, b, 4);
    memcpy_decker(tmp + 8, a + 4, 4);
    memcpy_decker(tmp + 12, b + 4, 4);
    memcpy_decker(t, tmp, 16);
}

// Simulate _mm_unpackhi_epi32
__device__ __forceinline__ void unpackhi32(unsigned char *t, unsigned char *a, unsigned char *b)
{
    unsigned char tmp[16];
    memcpy_decker(tmp, a + 8, 4);
    memcpy_decker(tmp + 4, b + 8, 4);
    memcpy_decker(tmp + 8, a + 12, 4);
    memcpy_decker(tmp + 12, b + 12, 4);
    memcpy_decker(t, tmp, 16);
}



__device__ void haraka512_perm(unsigned char *out, const unsigned char *in) 
{
    int i, j;
	__align__(4) __shared__ unsigned char sharedMemory1[256];
	if (threadIdx.x < 256)
		sharedMemory1[threadIdx.x] = sbox[threadIdx.x];
    unsigned char s[64], tmp[16];
    memcpy_decker(s, (unsigned char *)in, 64);
#pragma unroll
    for (i = 0; i < 5; ++i) {
        // aes round(s)
		
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
if(i<4)
        unpacklo32(s + 16, s + 16, tmp);
    }

    memcpy_decker(out, s, 64);
}

__device__ void haraka512_full(unsigned char *out, const unsigned char *in)
{
    int i;

    //unsigned char out[64];
    haraka512_perm(out, in);

    for (i = 32; i < 40; i++) {
        out[i-16] = out[i] ^ in[i];
    }

     for (i = 48; i < 56; i++) {
        out[i-24] = out[i] ^ in[i];
    }



    /* Truncated */
    //memcpy_decker(out,      out + 8, 8);
    //memcpy_decker(out + 8,  out + 24, 8);
   // memcpy_decker(out + 16, out + 32, 8);
    //memcpy_decker(out + 24, out + 48, 8);
}
