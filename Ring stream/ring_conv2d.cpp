#include <iostream>
#include "ap_int.h"
#include "hls_stream.h"

#define CONV_IN_HEIGHT    8
#define CONV_IN_WIDTH     8
#define CONV_IN_CHANNELS  3
#define CONV_OUT_CHANNELS 1
#define KERNEL_SIZE       3
#define CONV_STRIDE       1


// Calculate output dimensions
#define CONV_OUT_HEIGHT ((CONV_IN_HEIGHT - KERNEL_SIZE) / CONV_STRIDE + 1)  // 6
#define CONV_OUT_WIDTH ((CONV_IN_WIDTH - KERNEL_SIZE) / CONV_STRIDE + 1)    // 6

void ring_conv2d(
float IFM[CONV_IN_HEIGHT][CONV_IN_WIDTH][CONV_IN_CHANNELS],
float F[KERNEL_SIZE][KERNEL_SIZE][CONV_OUT_CHANNELS],
float bias[CONV_OUT_CHANNELS],
float OFM[CONV_OUT_HEIGHT][CONV_OUT_WIDTH][CONV_OUT_CHANNELS]

){

  // Initialize output feature map to zero
	init_output:
	for (int h = 0; h < CONV_OUT_HEIGHT; h++) {

		for (int w = 0; w < CONV_OUT_WIDTH; w++) {

			#pragma HLS PIPELINE II=1

			for (int oc = 0; oc < CONV_OUT_CHANNELS; oc++) {
				OFM[h][w][oc] = bias[oc];
			}
		}
	}


	ring_conv:
	int W=0;
	for(int H=2; H<CONV_IN_HEIGHT; H=H+CONV_STRIDE){

		if(H%2==0){
			W+=KERNEL_SIZE-1;     // for changing left shifting to right shifting
			for(W; W<CONV_IN_WIDTH; W=W+CONV_STRIDE){

				for(int C=0; C<CONV_OUT_CHANNELS; C=C+1){

					for(int i=0; i<KERNEL_SIZE;i=i+1){

						for(int j=0; j<KERNEL_SIZE; j=j+1){

							OFM[H+1-KERNEL_SIZE][W+1-KERNEL_SIZE][C]+=IFM[H+1+i-KERNEL_SIZE][W+j-KERNEL_SIZE+1][C]*F[i][j][C];

						}

					}
					// if want to add bias finally
					OFM[H+1-KERNEL_SIZE][W+1-KERNEL_SIZE][C]+=bias[C];

				}



			}


		}

		else if(H%2!=0){
					W-=KERNEL_SIZE-1;     // for changing right shifting to left shifting
					for(W; W>=0; W=W-CONV_STRIDE){

						for(int C=0; C<CONV_OUT_CHANNELS; C=C+1){

							for(int i=0; i<KERNEL_SIZE;i=i+1){

								for(int j=0; j<KERNEL_SIZE; j=j+1){

									OFM[H+1-KERNEL_SIZE][W][C]+=IFM[H+1+i-KERNEL_SIZE][W+j][C]*F[i][j][C];

								}

							}
							// if want to add bias finally
							OFM[H+1-KERNEL_SIZE][W][C]+=bias[C];

						}



					}


				}


	}



};

