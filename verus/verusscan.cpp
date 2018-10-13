/**
 * Equihash solver interface for ccminer (compatible with linux and windows)
 * Solver taken from nheqminer, by djeZo (and NiceHash)
 * tpruvot - 2017 (GPL v3)
 */
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <stdexcept>
#include <vector>



//#include "eqcuda.hpp"
//#include "equihash.h" // equi_verify()

#include <miner.h>
extern "C"
{
#include "./verus/haraka.h"
}

// input here is 140 for the header and 1344 for the solution (equi.cpp)


#define EQNONCE_OFFSET 30 /* 27:34 */
#define NONCE_OFT EQNONCE_OFFSET

static bool init[MAX_GPUS] = { 0 };
static int valid_sols[MAX_GPUS] = { 0 };
static uint8_t _ALIGN(64) data_sols[MAX_GPUS][10][1536] = { 0 }; // 140+3+1344 required
extern void verus_hash(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t* resNonces);


#ifndef htobe32
#define htobe32(x) swab32(x)
#endif

extern "C" void VerusHashHalf(uint8_t *result, uint8_t *data, size_t len)
{
    unsigned char buf[128];
    unsigned char *bufPtr = buf;
    int pos = 0, nextOffset = 64;
    unsigned char *bufPtr2 = bufPtr + nextOffset;
    unsigned char *ptr = (unsigned char *)data;
    uint32_t count = 0;

    // put our last result or zero at beginning of buffer each time
    memset(bufPtr, 0, 32);

    // digest up to 32 bytes at a time
    for ( ; pos < len; pos += 32)
    {
        if (len - pos >= 32)
        {
            memcpy(bufPtr + 32, ptr + pos, 32);
        }
        else
        {
            int i = (int)(len - pos);
            memcpy(bufPtr + 32, ptr + pos, i);
            memset(bufPtr + 32 + i, 0, 32 - i);
        }

        count++;

        if (count == 47) break; // exit from cycle before last iteration

        //printf("[%02d.1] ", count); for (int z=0; z<64; z++) printf("%02x", bufPtr[z]); printf("\n");
		haraka512_zero(bufPtr2, bufPtr); // ( out, in)
        bufPtr2 = bufPtr;
        bufPtr += nextOffset;
        //printf("[%02d.2] ", count); for (int z=0; z<64; z++) printf("%02x", bufPtr[z]); printf("\n");


        nextOffset *= -1;
    }
    memcpy(result, bufPtr, 32);
};



extern "C" int scanhash_verus(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[35];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
    int dev_id = device_map[thr_id];
	uint32_t throughput = 0x4;
	struct timeval tv_start, tv_end, diff;
	double secs, solps;
	
	_ALIGN(64) uint8_t blockhash_half[256];
	uint32_t nonce_buf = 0;
	
    unsigned char block_41970[] = {0xfd, 0x40, 0x05}; // solution
	uint8_t _ALIGN(64) full_data[140+3+1344] = { 0 };
    uint8_t* sol_data = &full_data[140];
	uint32_t intensity = 25;
	
	
	
	
	memcpy(endiandata, pdata, 140);
	memcpy(full_data, endiandata, 140);  //pdata
    memcpy(full_data +140, block_41970, 3);
  
	
	VerusHashHalf(blockhash_half, full_data, 1487);	
		
	gettimeofday(&tv_start, NULL);  //get millisecond timer val for cal of h
	
	work->valid_nonces = 0;
	//verus_setBlock(blockhash_half, work->target, thr_id); //set data to gpu kernel
	memset(blockhash_half + 32, 0x00, 32);
	memset(blockhash_half + 96, 0x00, 32);
	memset(blockhash_half + 160, 0x00, 32);
	memset(blockhash_half + 224, 0x00, 32);


	//memcpy(blockhash_half + 32, full_data + 1486 - 14, 15);
	memcpy(blockhash_half + 64, blockhash_half, 64);
	memcpy(blockhash_half + 128, blockhash_half, 64);
	memcpy(blockhash_half + 192, blockhash_half, 64);
	//for (int i = 0; i < 256; i++) printf("%02x", ((uint8_t*)(&blockhash_half))[i]);
	//printf("\n");
	

     uint32_t _ALIGN(64) vhash[32];   const uint32_t Htarg = ptarget[7];
uint32_t _ALIGN(64) vhash2[8];
	do {
		
		*hashes_done = nonce_buf;
		((uint32_t *)&blockhash_half)[8] = nonce_buf;
		((uint32_t *)&blockhash_half)[24] = nonce_buf + 1;
		((uint32_t *)&blockhash_half)[40] = nonce_buf + 2;
		((uint32_t *)&blockhash_half)[56] = nonce_buf + 3;

		haraka512_4x((unsigned char*)vhash, (unsigned char*)blockhash_half);
			
		if (vhash[7] <= Htarg || vhash[15] <= Htarg || vhash[23] <= Htarg || vhash[31] <= Htarg )
		   {		
				if (vhash[7] <= Htarg) {
					*((uint32_t *)full_data + 368) = nonce_buf;
					for (int i = 0; i < 8; i++)
						vhash2[i] = vhash[i];
				}
				if (vhash[15] <= Htarg) {
					*((uint32_t *)full_data + 368) = nonce_buf + 1;
					for (int i = 0; i<8; i++)
						vhash2[i] = vhash[i+8];
				}
				if (vhash[23] <= Htarg) {
					*((uint32_t *)full_data + 368) = nonce_buf + 2;
					for (int i = 0; i<8; i++)
						vhash2[i] = vhash[i+16];
				}
				if (vhash[31] <= Htarg) {
					*((uint32_t *)full_data + 368) = nonce_buf + 3;
					for (int i = 0; i<8; i++)
						vhash2[i] = vhash[i+24];

				}
                //memset(blockhash_half + 32, 0x0, 32);
                memcpy(blockhash_half + 32, full_data + 1486 - 14, 15);
			//	for (int i = 0; i < 32; i++) printf("", blockhash_half[i]);
				//Sleep(2);
                //haraka512_zero((unsigned char*)vhash, (unsigned char*)blockhash_half);
				
				//Sleep(2);
				
				
					
					    work->valid_nonces++;
					
                        memcpy(work->data, endiandata, 140);
                        int nonce = work->valid_nonces-1;
                        memcpy(work->extra, sol_data, 1347);
                        bn_store_hash_target_ratio(vhash2, work->target, work, nonce);
                                    
						work->nonces[work->valid_nonces - 1] = endiandata[NONCE_OFT];
                        pdata[NONCE_OFT] = endiandata[NONCE_OFT] + 1;
						goto out; 
					
						
			}
			if ((uint64_t)throughput + (uint64_t)nonce_buf >= (uint64_t)UINT32_MAX) {
				
				break;
			}
		nonce_buf += throughput;

	} while (!work_restart[thr_id].restart);
        
        
out:
	gettimeofday(&tv_end, NULL);
	timeval_subtract(&diff, &tv_end, &tv_start);
	secs = (1.0 * diff.tv_sec) + (0.000001 * diff.tv_usec);
	solps = (double)nonce_buf / secs;
	//gpulog(LOG_INFO, thr_id, "%u K/hashes in %.2f s (%.2f MH/s)", nonce_buf/1000, secs, solps / 1000000);

	return work->valid_nonces;
}


