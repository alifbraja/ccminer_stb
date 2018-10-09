/**
 * Equihash solver interface for ccminer (compatible with linux and windows)
 * Solver taken from nheqminer, by djeZo (and NiceHash)
 * tpruvot - 2017 (GPL v3)
 */
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

#include <stdexcept>
#include <vector>

#include <sph/sph_sha2.h>

//#include "eqcuda.hpp"
//#include "equihash.h" // equi_verify()

#include <miner.h>
extern "C"
{
#include "./verus/haraka.h"
}

// input here is 140 for the header and 1344 for the solution (equi.cpp)


#include <cuda_helper.h>

#define EQNONCE_OFFSET 30 /* 27:34 */
#define NONCE_OFT EQNONCE_OFFSET

static bool init[MAX_GPUS] = { 0 };
static int valid_sols[MAX_GPUS] = { 0 };
static uint8_t _ALIGN(64) data_sols[MAX_GPUS][10][1536] = { 0 }; // 140+3+1344 required
extern void verus_hash(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t* resNonces);
extern void verus_setBlock(void *blockf,const void *pTargetIn);
extern void verus_init(int thr_id);

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
		haraka512_port_zero(bufPtr2, bufPtr); // ( out, in)
        bufPtr2 = bufPtr;
        bufPtr += nextOffset;
        //printf("[%02d.2] ", count); for (int z=0; z<64; z++) printf("%02x", bufPtr[z]); printf("\n");


        nextOffset *= -1;
    }
    memcpy(result, bufPtr, 32);
};

static void cb_hashdone(int thr_id) {
	if (!valid_sols[thr_id]) valid_sols[thr_id] = -1;
}
static bool cb_cancel(int thr_id) {
	if (work_restart[thr_id].restart)
		valid_sols[thr_id] = -1;
	return work_restart[thr_id].restart;
}

extern "C" int scanhash_verus(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[35];
	int i;
	
	
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
    int dev_id = device_map[thr_id];
	uint32_t throughput;
	struct timeval tv_start, tv_end, diff;
	double secs, solps;
	
    uint8_t blockhash_half[64];
	uint32_t nonce_buf = 0;
	
    unsigned char block_41970[] = {0xfd, 0x40, 0x05}; // solution
	uint8_t _ALIGN(64) full_data[140+3+1344] = { 0 };
    uint8_t* sol_data = &full_data[140];
	uint32_t intensity = 25;
	
		
	throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, 0x8000000);
	
	
	memcpy(endiandata, pdata, 140);
	memcpy(full_data, endiandata, 140);  //pdata
    memcpy(full_data +140, block_41970, 3);
  
	
	if (opt_benchmark)
		ptarget[7] = 0x000f;
	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		cuda_get_arch(thr_id);
	//	api_set_throughput(thr_id, throughput);
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);
		verus_init(thr_id);
		init[thr_id] = true;
	}
	
	VerusHashHalf(blockhash_half, full_data, 1487);	
	memset(blockhash_half + 32, 0x00, 32);
	
	gettimeofday(&tv_start, NULL);  //get millisecond timer val for cal of h
	
	work->valid_nonces = 0;
	verus_setBlock(blockhash_half, work->target); //set data to gpu kernel
	
        
	do {
		
            *hashes_done = (uint64_t)nonce_buf + (uint64_t)throughput;
		     verus_hash(thr_id, throughput, nonce_buf , work->nonces);
			
			if (work->nonces[0] != UINT32_MAX )
		   {
				const uint32_t Htarg = ptarget[7];
				uint32_t _ALIGN(64) vhash[8];
				
                *((uint32_t *)full_data + 368) = work->nonces[0];
                                       
                memset(blockhash_half + 32, 0x0, 32);
                memcpy(blockhash_half + 32, full_data + 1486 - 14, 15);
			//	for (int i = 0; i < 32; i++) printf("", blockhash_half[i]);
				//Sleep(2);
                haraka512_port_zero((unsigned char*)vhash, (unsigned char*)blockhash_half);
				//for (int i = 0; i < 32; i++) printf("", ((uint8_t*)(&vhash))[i]);
				//Sleep(2);
				
				if (vhash[7] <= Htarg && fulltest(vhash, ptarget))
					{
					
					    work->valid_nonces++;
					
                        memcpy(work->data, endiandata, 140);
                        int nonce = work->valid_nonces-1;
                        memcpy(work->extra, sol_data, 1347);
                        bn_store_hash_target_ratio(vhash, work->target, work, nonce);
                                    
						work->nonces[work->valid_nonces - 1] = endiandata[NONCE_OFT];
                        pdata[NONCE_OFT] = endiandata[NONCE_OFT] + 1;
						goto out; 
					}
						
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

// cleanup
void free_verushash(int thr_id)
{
	if (!init[thr_id])
		return;

	

	init[thr_id] = false;
}


