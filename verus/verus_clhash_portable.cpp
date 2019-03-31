/*
 * This uses veriations of the clhash algorithm for Verus Coin, licensed
 * with the Apache-2.0 open source license.
 * 
 * Copyright (c) 2018 Michael Toutonghi
 * Distributed under the Apache 2.0 software license, available in the original form for clhash
 * here: https://github.com/lemire/clhash/commit/934da700a2a54d8202929a826e2763831bd43cf7#diff-9879d6db96fd29134fc802214163b95a
 * 
 * Original CLHash code and any portions herein, (C) 2017, 2018 Daniel Lemire and Owen Kaser
 * Faster 64-bit universal hashing
 * using carry-less multiplications, Journal of Cryptographic Engineering (to appear)
 *
 * Best used on recent x64 processors (Haswell or better).
 * 
 * This implements an intermediate step in the last part of a Verus block hash. The intent of this step
 * is to more effectively equalize FPGAs over GPUs and CPUs.
 *
 **/


#include "haraka_portable.h"

#include <assert.h>
#include <string.h>




#ifdef __APPLE__
#include <sys/types.h>
#endif// APPLE

#ifdef _WIN32
#pragma warning (disable : 4146)
#include <intrin.h>
#else
//#include <x86intrin.h>
//#include "arm_neon.h"

#   include "SSE2NEON.h"
#include "softaesnc.h"
typedef int32x4_t __m128i;

#endif //WIN32

static const unsigned char sbox[256] =
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
void aesenc(unsigned char *s, unsigned char *rk) 
{
 uint8_t result[16];

  aesb_single_round((const unsigned char*)s,result, rk);
    
    memcpy(s,result,16);
}

void aesenc2(unsigned char *s, const unsigned char *rk) 
{
    unsigned char i, t, u, v[4][4];
    for (i = 0; i < 16; ++i) {
        v[((i / 4) + 4 - (i%4) ) % 4][i % 4] = sbox[s[i]];
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
        s[i] = v[i / 4][i % 4] ^ rk[i];
    }
}

void clmul64(uint64_t a, uint64_t b, uint64_t* r)
{
	uint8_t s = 4, i; //window size
	uint64_t two_s = 1 << s; //2^s
	uint64_t smask = two_s - 1; //s 1 bits
	uint64_t u[16];
	uint64_t tmp;
	uint64_t ifmask;
	//Precomputation
	u[0] = 0;
	u[1] = b;
	for (i = 2; i < two_s; i += 2) {
		u[i] = u[i >> 1] << 1; //even indices: left shift
		u[i + 1] = u[i] ^ b; //odd indices: xor b
	}
	//Multiply
	r[0] = u[a & smask]; //first window only affects lower word
	r[1] = 0;
	for (i = s; i < 64; i += s) {
		tmp = u[a >> i & smask];
		r[0] ^= tmp << i;
		r[1] ^= tmp >> (64 - i);
	}
	//Repair
	uint64_t m = 0xEEEEEEEEEEEEEEEE; //s=4 => 16 times 1110
	for (i = 1; i < s; i++) {
		tmp = ((a & m) >> i);
		m &= m << 1; //shift mask to exclude all bit j': j' mod s = i
		ifmask = -((b >> (64 - i)) & 1); //if the (64-i)th bit of b is 1
		r[1] ^= (tmp & ifmask);
	}
}

 __m128i _mm_clmulepi64_si128_emu(const __m128i &a, const __m128i &b, int imm)
{
	uint64_t result[2];
   
 // uint64x2_t aa = *(uint64x2_t*)&a;
 // uint64x2_t bb = *(uint64x2_t*)&b;
  
 //result = (__m128i)vmull_p64(vgetq_lane_u64(aa, 1), vgetq_lane_u64(bb,0)); 
  
 
	clmul64(*((uint64_t*)&a + (imm & 1)), *((uint64_t*)&b + ((imm & 0x10) >> 4)), result);
//  result = vmull_p64(aa, bb);
	/*
	// TEST
	const __m128i tmp1 = _mm_load_si128(&a);
	const __m128i tmp2 = _mm_load_si128(&b);
	imm = imm & 0x11;
	const __m128i testresult = (imm == 0x10) ? _mm_clmulepi64_si128(tmp1, tmp2, 0x10) : ((imm == 0x01) ? _mm_clmulepi64_si128(tmp1, tmp2, 0x01) : ((imm == 0x00) ? _mm_clmulepi64_si128(tmp1, tmp2, 0x00) : _mm_clmulepi64_si128(tmp1, tmp2, 0x11)));
	if (!memcmp(&testresult, &result, 16))
	{
	printf("_mm_clmulepi64_si128_emu: Portable version passed!\n");
	}
	else
	{
	printf("_mm_clmulepi64_si128_emu: Portable version failed! a: %lxh %lxl, b: %lxh %lxl, imm: %x, emu: %lxh %lxl, intrin: %lxh %lxl\n",
	*((uint64_t *)&a + 1), *(uint64_t *)&a,
	*((uint64_t *)&b + 1), *(uint64_t *)&b,
	imm,
	*((uint64_t *)result + 1), *(uint64_t *)result,
	*((uint64_t *)&testresult + 1), *(uint64_t *)&testresult);
	return testresult;
	}
	*/

	return *(__m128i *)result;
}

__m128i _mm_mulhrs_epi16_emu(__m128i _a, __m128i _b)
{
	int16_t result[8];
	int16_t *a = (int16_t*)&_a, *b = (int16_t*)&_b;
	for (int i = 0; i < 8; i++)
	{
		result[i] = (int16_t)((((int32_t)(a[i]) * (int32_t)(b[i])) + 0x4000) >> 15);
	}

	/*
	const __m128i testresult = _mm_mulhrs_epi16(_a, _b);
	if (!memcmp(&testresult, &result, 16))
	{
	printf("_mm_mulhrs_epi16_emu: Portable version passed!\n");
	}
	else
	{
	printf("_mm_mulhrs_epi16_emu: Portable version failed! a: %lxh %lxl, b: %lxh %lxl, emu: %lxh %lxl, intrin: %lxh %lxl\n",
	*((uint64_t *)&a + 1), *(uint64_t *)&a,
	*((uint64_t *)&b + 1), *(uint64_t *)&b,
	*((uint64_t *)result + 1), *(uint64_t *)result,
	*((uint64_t *)&testresult + 1), *(uint64_t *)&testresult);
	}
	*/

	return *(__m128i *)result;
}

 __m128i _mm_set_epi64x_emu(uint64_t hi, uint64_t lo)
{
	__m128i result;
	((uint64_t *)&result)[0] = lo;
	((uint64_t *)&result)[1] = hi;
	return result;
}

 __m128i _mm_cvtsi64_si128_emu(uint64_t lo)
{
	__m128i result;
	((uint64_t *)&result)[0] = lo;
	((uint64_t *)&result)[1] = 0;
	return result;
}

 int64_t _mm_cvtsi128_si64_emu(__m128i &a)
{
	return *(int64_t *)&a;
}

 int32_t _mm_cvtsi128_si32_emu(__m128i &a)
{
	return *(int32_t *)&a;
}

 __m128i _mm_cvtsi32_si128_emu(uint32_t lo)
{
//	__m128i result;
//	((uint32_t *)&result)[0] = lo;
//	((uint32_t *)&result)[1] = 0;
//	((uint64_t *)&result)[1] = 0;
return vreinterpretq_m128i_s32(vsetq_lane_s32(lo, vdupq_n_s32(0), 0));
	/*
	const __m128i testresult = _mm_cvtsi32_si128(lo);
	if (!memcmp(&testresult, &result, 16))
	{
	printf("_mm_cvtsi32_si128_emu: Portable version passed!\n");
	}
	else
	{
	printf("_mm_cvtsi32_si128_emu: Portable version failed!\n");
	}
	*/

//	return result;
}
typedef uint8_t u_char;

__m128i _mm_setr_epi8_emu(u_char c0, u_char c1, u_char c2, u_char c3, u_char c4, u_char c5, u_char c6, u_char c7, u_char c8, u_char c9, u_char c10, u_char c11, u_char c12, u_char c13, u_char c14, u_char c15)
{
	__m128i result;
	((uint8_t *)&result)[0] = c0;
	((uint8_t *)&result)[1] = c1;
	((uint8_t *)&result)[2] = c2;
	((uint8_t *)&result)[3] = c3;
	((uint8_t *)&result)[4] = c4;
	((uint8_t *)&result)[5] = c5;
	((uint8_t *)&result)[6] = c6;
	((uint8_t *)&result)[7] = c7;
	((uint8_t *)&result)[8] = c8;
	((uint8_t *)&result)[9] = c9;
	((uint8_t *)&result)[10] = c10;
	((uint8_t *)&result)[11] = c11;
	((uint8_t *)&result)[12] = c12;
	((uint8_t *)&result)[13] = c13;
	((uint8_t *)&result)[14] = c14;
	((uint8_t *)&result)[15] = c15;

	/*
	const __m128i testresult = _mm_setr_epi8(c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15);
	if (!memcmp(&testresult, &result, 16))
	{
	printf("_mm_setr_epi8_emu: Portable version passed!\n");
	}
	else
	{
	printf("_mm_setr_epi8_emu: Portable version failed!\n");
	}
	*/

	return result;
}

#define _mm_srli_si128_emu(a, imm) \
({ \
	__m128i ret; \
	if ((imm) <= 0) { \
		ret = a; \
	} \
	else if ((imm) > 15) { \
		ret = _mm_setzero_si128(); \
	} \
	else { \
		ret = vreinterpretq_m128i_s8(vextq_s8(vreinterpretq_s8_m128i(a), vdupq_n_s8(0), (imm))); \
	} \
	ret; \
})

 __m128i _mm_srli_si128_emu_old(__m128i a, int imm8)
{
	unsigned char result[16];
	uint8_t shift = imm8 & 0xff;
	if (shift > 15) shift = 16;

	int i;
	for (i = 0; i < (16 - shift); i++)
	{
		result[i] = ((unsigned char *)&a)[shift + i];
	}
	for (; i < 16; i++)
	{
		result[i] = 0;
	}

	/*
	const __m128i tmp1 = _mm_load_si128(&a);
	__m128i testresult = _mm_srli_si128(tmp1, imm8);
	if (!memcmp(&testresult, result, 16))
	{
	printf("_mm_srli_si128_emu: Portable version passed!\n");
	}
	else
	{
	printf("_mm_srli_si128_emu: Portable version failed! val: %lx%lx imm: %x emu: %lx%lx, intrin: %lx%lx\n",
	*((uint64_t *)&a + 1), *(uint64_t *)&a,
	imm8,
	*((uint64_t *)result + 1), *(uint64_t *)result,
	*((uint64_t *)&testresult + 1), *(uint64_t *)&testresult);
	}
	*/

	return *(__m128i *)result;
}

 __m128i _mm_xor_si128_emu(__m128i a, __m128i b)
{
	return vreinterpretq_m128i_s32( veorq_s32(vreinterpretq_s32_m128i(a), vreinterpretq_s32_m128i(b)) );

}

 __m128i _mm_load_si128_emu(const void *p)
{

return vreinterpretq_m128i_s32(vld1q_s32((int32_t *)p));
//	return *(__m128i *)p;
}

 void _mm_store_si128_emu(void *p, __m128i val)
{
	
  vst1q_s32((int32_t*) p, vreinterpretq_s32_m128i(val));
 // *(__m128i *)p = val;
}

__m128i _mm_shuffle_epi8_emu(__m128i a, __m128i b)
{
	__m128i result;
	for (int i = 0; i < 16; i++)
	{
		if (((uint8_t *)&b)[i] & 0x80)
		{
			((uint8_t *)&result)[i] = 0;
		}
		else
		{
			((uint8_t *)&result)[i] = ((uint8_t *)&a)[((uint8_t *)&b)[i] & 0xf];
		}
	}

	/*
	const __m128i tmp1 = _mm_load_si128(&a);
	const __m128i tmp2 = _mm_load_si128(&b);
	__m128i testresult = _mm_shuffle_epi8(tmp1, tmp2);
	if (!memcmp(&testresult, &result, 16))
	{
	printf("_mm_shuffle_epi8_emu: Portable version passed!\n");
	}
	else
	{
	printf("_mm_shuffle_epi8_emu: Portable version failed!\n");
	}
	*/

	return result;
}

// portable
__m128i lazyLengthHash_port(uint64_t keylength, uint64_t length) {
	const __m128i lengthvector = _mm_set_epi64x_emu(keylength, length);
	const __m128i clprod1 = _mm_clmulepi64_si128_emu(lengthvector, lengthvector, 0x10);
	return clprod1;
}

// modulo reduction to 64-bit value. The high 64 bits contain garbage, see precompReduction64
__m128i precompReduction64_si128_port(__m128i A) {

	//const __m128i C = _mm_set_epi64x(1U,(1U<<4)+(1U<<3)+(1U<<1)+(1U<<0)); // C is the irreducible poly. (64,4,3,1,0)
	const __m128i C = _mm_cvtsi64_si128_emu((1U << 4) + (1U << 3) + (1U << 1) + (1U << 0));
	__m128i Q2 = _mm_clmulepi64_si128_emu(A, C, 0x01);
	__m128i Q3 = _mm_shuffle_epi8_emu(_mm_setr_epi8_emu(0, 27, 54, 45, 108, 119, 90, 65, (char)216, (char)195, (char)238, (char)245, (char)180, (char)175, (char)130, (char)153),
		_mm_srli_si128_emu(Q2, 8));
	__m128i Q4 = _mm_xor_si128_emu(Q2, A);
	const __m128i final = _mm_xor_si128_emu(Q3, Q4);
	return final;/// WARNING: HIGH 64 BITS SHOULD BE ASSUMED TO CONTAIN GARBAGE
}

uint64_t precompReduction64_port(__m128i A) {
	__m128i tmp = precompReduction64_si128_port(A);
	return _mm_cvtsi128_si64_emu(tmp);
}

// verus intermediate hash extra
__m128i __verusclmulwithoutreduction64alignedrepeat_port(__m128i *randomsource, const __m128i buf[4], uint64_t keyMask, uint32_t * __restrict fixrand, uint32_t * __restrict fixrandex)
{
	__m128i const *pbuf;

	/*
	std::cout << "Random key start: ";
	std::cout << LEToHex(*randomsource) << ", ";
	std::cout << LEToHex(*(randomsource + 1));
	std::cout << std::endl;
	*/

	// divide key mask by 16 from bytes to __m128i
	keyMask >>= 4;

	// the random buffer must have at least 32 16 byte dwords after the keymask to work with this
	// algorithm. we take the value from the last element inside the keyMask + 2, as that will never
	// be used to xor into the accumulator before it is hashed with other values first
	__m128i acc = _mm_load_si128_emu(randomsource + (keyMask + 2));

	for (int64_t i = 0; i < 32; i++)
	{
		//std::cout << "LOOP " << i << " acc: " << LEToHex(acc) << std::endl;

		const uint64_t selector = _mm_cvtsi128_si64_emu(acc);

		// get two random locations in the key, which will be mutated and swapped
		__m128i *prand = randomsource + ((selector >> 5) & keyMask);
		__m128i *prandex = randomsource + ((selector >> 32) & keyMask);

	

		// select random start and order of pbuf processing
		pbuf = buf + (selector & 3);
	uint32_t prand_idx = (selector >> 5) & keyMask;
	uint32_t prandex_idx = (selector >>32) & keyMask;
  


		switch (selector & 0x1c)
		{
		case 0:
		{
			const __m128i temp1 = _mm_load_si128_emu(prandex);
			const __m128i temp2 = _mm_load_si128_emu(pbuf - (((selector & 1) << 1) - 1));
			const __m128i add1 = _mm_xor_si128_emu(temp1, temp2);
			const __m128i clprod1 = _mm_clmulepi64_si128_emu(add1, add1, 0x10);
			acc = _mm_xor_si128_emu(clprod1, acc);

			/*
			std::cout << "temp1: " << LEToHex(temp1) << std::endl;
			std::cout << "temp2: " << LEToHex(temp2) << std::endl;
			std::cout << "add1: " << LEToHex(add1) << std::endl;
			std::cout << "clprod1: " << LEToHex(clprod1) << std::endl;
			std::cout << "acc: " << LEToHex(acc) << std::endl;
			*/

			const __m128i tempa1 = _mm_mulhrs_epi16_emu(acc, temp1);
			const __m128i tempa2 = _mm_xor_si128_emu(tempa1, temp1);

			const __m128i temp12 = _mm_load_si128_emu(prand);
			_mm_store_si128_emu(prand, tempa2);

			const __m128i temp22 = _mm_load_si128_emu(pbuf);
			const __m128i add12 = _mm_xor_si128_emu(temp12, temp22);
			const __m128i clprod12 = _mm_clmulepi64_si128_emu(add12, add12, 0x10);
			acc = _mm_xor_si128_emu(clprod12, acc);

			const __m128i tempb1 = _mm_mulhrs_epi16_emu(acc, temp12);
			const __m128i tempb2 = _mm_xor_si128_emu(tempb1, temp12);
			_mm_store_si128_emu(prandex, tempb2);
			break;
		}
		case 4:
		{
			const __m128i temp1 = _mm_load_si128_emu(prand);
			const __m128i temp2 = _mm_load_si128_emu(pbuf);
			const __m128i add1 = _mm_xor_si128_emu(temp1, temp2);
			const __m128i clprod1 = _mm_clmulepi64_si128_emu(add1, add1, 0x10);
			acc = _mm_xor_si128_emu(clprod1, acc);
			const __m128i clprod2 = _mm_clmulepi64_si128_emu(temp2, temp2, 0x10);
			acc = _mm_xor_si128_emu(clprod2, acc);

			const __m128i tempa1 = _mm_mulhrs_epi16_emu(acc, temp1);
			const __m128i tempa2 = _mm_xor_si128_emu(tempa1, temp1);

			const __m128i temp12 = _mm_load_si128_emu(prandex);
			_mm_store_si128_emu(prandex, tempa2);

			const __m128i temp22 = _mm_load_si128_emu(pbuf - (((selector & 1) << 1) - 1));
			const __m128i add12 = _mm_xor_si128_emu(temp12, temp22);
			acc = _mm_xor_si128_emu(add12, acc);

			const __m128i tempb1 = _mm_mulhrs_epi16_emu(acc, temp12);
			const __m128i tempb2 = _mm_xor_si128_emu(tempb1, temp12);
			_mm_store_si128_emu(prand, tempb2);
			break;
		}
		case 8:
		{
			const __m128i temp1 = _mm_load_si128_emu(prandex);
			const __m128i temp2 = _mm_load_si128_emu(pbuf);
			const __m128i add1 = _mm_xor_si128_emu(temp1, temp2);
			acc = _mm_xor_si128_emu(add1, acc);

			const __m128i tempa1 = _mm_mulhrs_epi16_emu(acc, temp1);
			const __m128i tempa2 = _mm_xor_si128_emu(tempa1, temp1);

			const __m128i temp12 = _mm_load_si128_emu(prand);
			_mm_store_si128_emu(prand, tempa2);

			const __m128i temp22 = _mm_load_si128_emu(pbuf - (((selector & 1) << 1) - 1));
			const __m128i add12 = _mm_xor_si128_emu(temp12, temp22);
			const __m128i clprod12 = _mm_clmulepi64_si128_emu(add12, add12, 0x10);
			acc = _mm_xor_si128_emu(clprod12, acc);
			const __m128i clprod22 = _mm_clmulepi64_si128_emu(temp22, temp22, 0x10);
			acc = _mm_xor_si128_emu(clprod22, acc);

			const __m128i tempb1 = _mm_mulhrs_epi16_emu(acc, temp12);
			const __m128i tempb2 = _mm_xor_si128_emu(tempb1, temp12);
			_mm_store_si128_emu(prandex, tempb2);
			break;
		}
		case 0xc:
		{
			const __m128i temp1 = _mm_load_si128_emu(prand);
			const __m128i temp2 = _mm_load_si128_emu(pbuf - (((selector & 1) << 1) - 1));
			const __m128i add1 = _mm_xor_si128_emu(temp1, temp2);

			// cannot be zero here
			const int32_t divisor = (uint32_t)selector;

			acc = _mm_xor_si128_emu(add1, acc);

			const int64_t dividend = _mm_cvtsi128_si64_emu(acc);
			const __m128i modulo = _mm_cvtsi32_si128_emu(dividend % divisor);
			acc = _mm_xor_si128_emu(modulo, acc);

			const __m128i tempa1 = _mm_mulhrs_epi16_emu(acc, temp1);
			const __m128i tempa2 = _mm_xor_si128_emu(tempa1, temp1);

			if (dividend & 1)
			{
				const __m128i temp12 = _mm_load_si128_emu(prandex);
				_mm_store_si128_emu(prandex, tempa2);

				const __m128i temp22 = _mm_load_si128_emu(pbuf);
				const __m128i add12 = _mm_xor_si128_emu(temp12, temp22);
				const __m128i clprod12 = _mm_clmulepi64_si128_emu(add12, add12, 0x10);
				acc = _mm_xor_si128_emu(clprod12, acc);
				const __m128i clprod22 = _mm_clmulepi64_si128_emu(temp22, temp22, 0x10);
				acc = _mm_xor_si128_emu(clprod22, acc);

				const __m128i tempb1 = _mm_mulhrs_epi16_emu(acc, temp12);
				const __m128i tempb2 = _mm_xor_si128_emu(tempb1, temp12);
				_mm_store_si128_emu(prand, tempb2);
			}
			else
			{
				const __m128i tempb3 = _mm_load_si128_emu(prandex);
				_mm_store_si128_emu(prandex, tempa2);
				_mm_store_si128_emu(prand, tempb3);
			}
			break;
		}
		case 0x10:
		{
			// a few AES operations
			const __m128i *rc = prand;
			__m128i tmp;

			__m128i temp1 = _mm_load_si128_emu(pbuf - (((selector & 1) << 1) - 1));
			__m128i temp2 = _mm_load_si128_emu(pbuf);

			AES2_EMU(temp1, temp2, 0);
			MIX2_EMU(temp1, temp2);

			AES2_EMU(temp1, temp2, 4);
			MIX2_EMU(temp1, temp2);

			AES2_EMU(temp1, temp2, 8);
			MIX2_EMU(temp1, temp2);

			acc = _mm_xor_si128_emu(temp1, acc);
			acc = _mm_xor_si128_emu(temp2, acc);

			const __m128i tempa1 = _mm_load_si128_emu(prand);
			const __m128i tempa2 = _mm_mulhrs_epi16_emu(acc, tempa1);
			const __m128i tempa3 = _mm_xor_si128_emu(tempa1, tempa2);

			const __m128i tempa4 = _mm_load_si128_emu(prandex);
			_mm_store_si128_emu(prandex, tempa3);
			_mm_store_si128_emu(prand, tempa4);
			break;
		}
		case 0x14:
		{
			// we'll just call this one the monkins loop, inspired by Chris
			const __m128i *buftmp = pbuf - (((selector & 1) << 1) - 1);
			__m128i tmp; // used by MIX2

			uint64_t rounds = selector >> 61; // loop randomly between 1 and 8 times
			__m128i *rc = prand;
			uint64_t aesround = 0;
			__m128i onekey;

			do
			{
				//std::cout << "acc: " << LEToHex(acc) << ", round check: " << LEToHex((selector & (0x10000000 << rounds))) << std::endl;

				// note that due to compiler and CPUs, we expect this to do:
				// if (selector & ((0x10000000 << rounds) & 0xffffffff) if rounds != 3 else selector & 0xffffffff80000000):
				if (selector & (0x10000000 << rounds))
				{
					onekey = _mm_load_si128_emu(rc++);
					const __m128i temp2 = _mm_load_si128_emu(rounds & 1 ? pbuf : buftmp);
					const __m128i add1 = _mm_xor_si128_emu(onekey, temp2);
					const __m128i clprod1 = _mm_clmulepi64_si128_emu(add1, add1, 0x10);
					acc = _mm_xor_si128_emu(clprod1, acc);
				}
				else
				{
					onekey = _mm_load_si128_emu(rc++);
					__m128i temp2 = _mm_load_si128_emu(rounds & 1 ? buftmp : pbuf);
					const uint64_t roundidx = aesround++ << 2;
					AES2_EMU(onekey, temp2, roundidx);

					/*
					std::cout << " onekey1: " << LEToHex(onekey) << std::endl;
					std::cout << "  temp21: " << LEToHex(temp2) << std::endl;
					std::cout << "roundkey: " << LEToHex(rc[roundidx]) << std::endl;

					aesenc((unsigned char *)&onekey, (unsigned char *)&(rc[roundidx]));

					std::cout << "onekey2: " << LEToHex(onekey) << std::endl;
					std::cout << "roundkey: " << LEToHex(rc[roundidx + 1]) << std::endl;

					aesenc((unsigned char *)&temp2, (unsigned char *)&(rc[roundidx + 1]));

					std::cout << " temp22: " << LEToHex(temp2) << std::endl;
					std::cout << "roundkey: " << LEToHex(rc[roundidx + 2]) << std::endl;

					aesenc((unsigned char *)&onekey, (unsigned char *)&(rc[roundidx + 2]));

					std::cout << "onekey2: " << LEToHex(onekey) << std::endl;

					aesenc((unsigned char *)&temp2, (unsigned char *)&(rc[roundidx + 3]));

					std::cout << " temp22: " << LEToHex(temp2) << std::endl;
					*/

					MIX2_EMU(onekey, temp2);

					/*
					std::cout << "onekey3: " << LEToHex(onekey) << std::endl;
					*/

					acc = _mm_xor_si128_emu(onekey, acc);
					acc = _mm_xor_si128_emu(temp2, acc);
				}
			} while (rounds--);

			const __m128i tempa1 = _mm_load_si128_emu(prand);
			const __m128i tempa2 = _mm_mulhrs_epi16_emu(acc, tempa1);
			const __m128i tempa3 = _mm_xor_si128_emu(tempa1, tempa2);

			const __m128i tempa4 = _mm_load_si128_emu(prandex);
			_mm_store_si128_emu(prandex, tempa3);
			_mm_store_si128_emu(prand, tempa4);
			break;
		}
		case 0x18:
		{
			const __m128i temp1 = _mm_load_si128_emu(pbuf - (((selector & 1) << 1) - 1));
			const __m128i temp2 = _mm_load_si128_emu(prand);
			const __m128i add1 = _mm_xor_si128_emu(temp1, temp2);
			const __m128i clprod1 = _mm_clmulepi64_si128_emu(add1, add1, 0x10);
			acc = _mm_xor_si128_emu(clprod1, acc);

			const __m128i tempa1 = _mm_mulhrs_epi16_emu(acc, temp2);
			const __m128i tempa2 = _mm_xor_si128_emu(tempa1, temp2);

			const __m128i tempb3 = _mm_load_si128_emu(prandex);
			_mm_store_si128_emu(prandex, tempa2);
			_mm_store_si128_emu(prand, tempb3);
			break;
		}
		case 0x1c:
		{
			const __m128i temp1 = _mm_load_si128_emu(pbuf);
			const __m128i temp2 = _mm_load_si128_emu(prandex);
			const __m128i add1 = _mm_xor_si128_emu(temp1, temp2);
			const __m128i clprod1 = _mm_clmulepi64_si128_emu(add1, add1, 0x10);
			acc = _mm_xor_si128_emu(clprod1, acc);

			const __m128i tempa1 = _mm_mulhrs_epi16_emu(acc, temp2);
			const __m128i tempa2 = _mm_xor_si128_emu(tempa1, temp2);

			const __m128i tempa3 = _mm_load_si128_emu(prand);
			_mm_store_si128_emu(prand, tempa2);

			acc = _mm_xor_si128_emu(tempa3, acc);

			const __m128i tempb1 = _mm_mulhrs_epi16_emu(acc, tempa3);
			const __m128i tempb2 = _mm_xor_si128_emu(tempb1, tempa3);
			_mm_store_si128_emu(prandex, tempb2);
			break;
		}
		}
   fixrand[i] = prand_idx;
		fixrandex[i] = prandex_idx;
   
	}
	return acc;
}

// hashes 64 bytes only by doing a carryless multiplication and reduction of the repeated 64 byte sequence 16 times, 
// returning a 64 bit hash value
uint64_t verusclhash_port(void * random, const unsigned char buf[64], uint64_t keyMask, uint32_t *  __restrict fixrand, uint32_t * __restrict fixrandex) {
    const unsigned int  m = 128;// we process the data in chunks of 16 cache lines
    __m128i * rs64 = (__m128i *)random;
    const __m128i * string = (const __m128i *) buf;

    __m128i  acc = __verusclmulwithoutreduction64alignedrepeat_port(rs64, string, keyMask, fixrand, fixrandex);
    acc = _mm_xor_si128_emu(acc, lazyLengthHash_port(1024, 64));
    return precompReduction64_port(acc);
}
