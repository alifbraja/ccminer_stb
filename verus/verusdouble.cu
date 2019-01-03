
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
__device__  uint32_t sbox[256] =
{0x63636363, 0x7c7c7c7c, 0x77777777, 0x7b7b7b7b, 0xf2f2f2f2, 0x6b6b6b6b, 0x6f6f6f6f, 0xc5c5c5c5, 0x30303030, 0x01010101, 0x67676767, 0x2b2b2b2b, 0xfefefefe, 0xd7d7d7d7, 0xabababab, 0x76767676, 0xcacacaca, 0x82828282, 0xc9c9c9c9, 0x7d7d7d7d, 0xfafafafa, 0x59595959, 0x47474747, 0xf0f0f0f0, 0xadadadad, 0xd4d4d4d4, 0xa2a2a2a2, 0xafafafaf, 0x9c9c9c9c, 0xa4a4a4a4, 0x72727272, 0xc0c0c0c0, 0xb7b7b7b7, 0xfdfdfdfd, 0x93939393, 0x26262626, 0x36363636, 0x3f3f3f3f, 0xf7f7f7f7, 0xcccccccc, 0x34343434, 0xa5a5a5a5, 0xe5e5e5e5, 0xf1f1f1f1, 0x71717171, 0xd8d8d8d8, 0x31313131, 0x15151515, 0x04040404, 0xc7c7c7c7, 0x23232323, 0xc3c3c3c3, 0x18181818, 0x96969696, 0x05050505, 0x9a9a9a9a, 0x07070707, 0x12121212, 0x80808080, 0xe2e2e2e2, 0xebebebeb, 0x27272727, 0xb2b2b2b2, 0x75757575, 0x09090909, 0x83838383, 0x2c2c2c2c, 0x1a1a1a1a, 0x1b1b1b1b, 0x6e6e6e6e, 0x5a5a5a5a, 0xa0a0a0a0, 0x52525252, 0x3b3b3b3b, 0xd6d6d6d6, 0xb3b3b3b3, 0x29292929, 0xe3e3e3e3, 0x2f2f2f2f, 0x84848484, 0x53535353, 0xd1d1d1d1, 0x00000000, 0xedededed, 0x20202020, 0xfcfcfcfc, 0xb1b1b1b1, 0x5b5b5b5b, 0x6a6a6a6a, 0xcbcbcbcb, 0xbebebebe, 0x39393939, 0x4a4a4a4a, 0x4c4c4c4c, 0x58585858, 0xcfcfcfcf, 0xd0d0d0d0, 0xefefefef, 0xaaaaaaaa, 0xfbfbfbfb, 0x43434343, 0x4d4d4d4d, 0x33333333, 0x85858585, 0x45454545, 0xf9f9f9f9, 0x02020202, 0x7f7f7f7f, 0x50505050, 0x3c3c3c3c, 0x9f9f9f9f, 0xa8a8a8a8, 0x51515151, 0xa3a3a3a3, 0x40404040, 0x8f8f8f8f, 0x92929292, 0x9d9d9d9d, 0x38383838, 0xf5f5f5f5, 0xbcbcbcbc, 0xb6b6b6b6, 0xdadadada, 0x21212121, 0x10101010, 0xffffffff, 0xf3f3f3f3, 0xd2d2d2d2, 0xcdcdcdcd, 0x0c0c0c0c, 0x13131313, 0xecececec, 0x5f5f5f5f, 0x97979797, 0x44444444, 0x17171717, 0xc4c4c4c4, 0xa7a7a7a7, 0x7e7e7e7e, 0x3d3d3d3d, 0x64646464, 0x5d5d5d5d, 0x19191919, 0x73737373, 0x60606060, 0x81818181, 0x4f4f4f4f, 0xdcdcdcdc, 0x22222222, 0x2a2a2a2a, 0x90909090, 0x88888888, 0x46464646, 0xeeeeeeee, 0xb8b8b8b8, 0x14141414, 0xdededede, 0x5e5e5e5e, 0x0b0b0b0b, 0xdbdbdbdb, 0xe0e0e0e0, 0x32323232, 0x3a3a3a3a, 0x0a0a0a0a, 0x49494949, 0x06060606, 0x24242424, 0x5c5c5c5c, 0xc2c2c2c2, 0xd3d3d3d3, 0xacacacac, 0x62626262, 0x91919191, 0x95959595, 0xe4e4e4e4, 0x79797979, 0xe7e7e7e7, 0xc8c8c8c8, 0x37373737, 0x6d6d6d6d, 0x8d8d8d8d, 0xd5d5d5d5, 0x4e4e4e4e, 0xa9a9a9a9, 0x6c6c6c6c, 0x56565656, 0xf4f4f4f4, 0xeaeaeaea, 0x65656565, 0x7a7a7a7a, 0xaeaeaeae, 0x08080808, 0xbabababa, 0x78787878, 0x25252525, 0x2e2e2e2e, 0x1c1c1c1c, 0xa6a6a6a6, 0xb4b4b4b4, 0xc6c6c6c6, 0xe8e8e8e8, 0xdddddddd, 0x74747474, 0x1f1f1f1f, 0x4b4b4b4b, 0xbdbdbdbd, 0x8b8b8b8b, 0x8a8a8a8a, 0x70707070, 0x3e3e3e3e, 0xb5b5b5b5, 0x66666666, 0x48484848, 0x03030303, 0xf6f6f6f6, 0x0e0e0e0e, 0x61616161, 0x35353535, 0x57575757, 0xb9b9b9b9, 0x86868686, 0xc1c1c1c1, 0x1d1d1d1d, 0x9e9e9e9e, 0xe1e1e1e1, 0xf8f8f8f8, 0x98989898, 0x11111111, 0x69696969, 0xd9d9d9d9, 0x8e8e8e8e, 0x94949494, 0x9b9b9b9b, 0x1e1e1e1e, 0x87878787, 0xe9e9e9e9, 0xcececece, 0x55555555, 0x28282828, 0xdfdfdfdf, 0x8c8c8c8c, 0xa1a1a1a1, 0x89898989, 0x0d0d0d0d, 0xbfbfbfbf, 0xe6e6e6e6, 0x42424242, 0x68686868, 0x41414141, 0x99999999, 0x2d2d2d2d, 0x0f0f0f0f, 0xb0b0b0b0, 0x54545454, 0xbbbbbbbb, 0x16161616}

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
#define XT(x) (((x) << 1) ^ ((((x) >> 7) & 1) * 0x1b))

// Simulate _mm_aesenc_si128 instructions from AESNI
__device__   void aesenc(uint32_t *s,uint32_t *sharedMemory1)
{
	uint32_t i, t, u;
	register uint32_t  v[4][4];
	
	for (i = 0; i < 16; ++i) {
		v[((i >> 2) + 4 - (i & 3)) & 3][i & 3] = sharedMemory1[s[i]];
	
	}

	for (i = 0; i < 4; ++i) {
		t = v[i][0];
		u = v[i][0] ^ v[i][1] ^ v[i][2] ^ v[i][3];
		v[i][0] = v[i][0] ^ u ^ XT(v[i][0] ^ v[i][1]);


		v[i][0] = v[i][0] ^ u ^ XT(v[i][0] ^ v[i][1]);

		v[i][1] = v[i][1] ^ u ^ XT(v[i][1] ^ v[i][2]);
		v[i][2] = v[i][2] ^ u ^ XT(v[i][2] ^ v[i][3]);
		v[i][3] = v[i][3] ^ u ^ XT(v[i][3] ^ t);
	}
	for (i = 0; i < 16; ++i) {
		s[i] = v[i >> 2][i & 3]; // VerusHash have 0 rc vector
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
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
	
	int i, j; 
	uint32_t s[64] = { 0 };
	uint32_t tmp[16];
		__shared__ uint32_t sharedMemory1[64];
	if (threadIdx.x < 64)
		sharedMemory1[threadIdx.x] = sbox[threadIdx.x];//	for (i = 0; i < 64; ++i)
					

	uint32_t nounce[4];
	uint64_t in[4];
	nounce[0] = startNonce + thread;
	nounce[1] = startNonce + thread + 1;
	nounce[2] = startNonce + thread + 2;
	nounce[3] = startNonce + thread + 3;
		
		
		uint64_t blockhash[4];
		for (i = 0; i < 32; ++i) {
			((uint8_t*)&s[i])[0] = blockhash_half[i];
			((uint8_t*)&s[i])[1] = blockhash_half[i];
			((uint8_t*)&s[i])[2] = blockhash_half[i];
			((uint8_t*)&s[i])[3] = blockhash_half[i];
		}
		for (i = 32; i < 64; ++i) {
			((uint8_t*)&s[i])[0] = 0;
			((uint8_t*)&s[i])[1] = 0;
			((uint8_t*)&s[i])[2] = 0;
			((uint8_t*)&s[i])[3] = 0;
		}
		for (i = 32; i < 40; ++i) {
			((uint8_t*)&s[i])[0] = ((uint8_t*)&nounce)[i];
			((uint8_t*)&s[i])[1] = ((uint8_t*)&nounce)[i];
			((uint8_t*)&s[i])[2] = ((uint8_t*)&nounce)[i];
			((uint8_t*)&s[i])[3] = ((uint8_t*)&nounce)[i];
		}
		//memcpy(s, blockhash_half, 32);
		//memset(s + 32, 0x0, 32);
		//((uint32_t *)&s)[8] = startNonce + thread;
		//memcpy(in +48, s + 48, 8);
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


