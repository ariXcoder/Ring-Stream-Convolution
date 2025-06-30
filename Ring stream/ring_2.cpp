#include <iostream>
#include "hls_stream.h"

#define CONV_IN_HEIGHT    8
#define CONV_IN_WIDTH     8
#define CONV_IN_CHANNELS_2  2
#define CONV_OUT_CHANNELS_COM 1
#define KERNEL_SIZE       3
#define CONV_STRIDE       1
#define KERNEL_CHANNELS_COM   1     // same as the no. of CONV_OUT_CHANNELS_2
#define padding           0     // user defined, not an input argument

// Calculate output dimensions
#define CONV_OUT_HEIGHT ((CONV_IN_HEIGHT - KERNEL_SIZE+2*padding) / CONV_STRIDE + 1)  // 6
#define CONV_OUT_WIDTH ((CONV_IN_WIDTH - KERNEL_SIZE+2*padding) / CONV_STRIDE + 1)    // 6

/*void ring_2_conv2d(
float IFM[CONV_IN_CHANNELS_2][CONV_IN_HEIGHT][CONV_IN_WIDTH],
float F[KERNEL_CHANNELS_COM][CONV_IN_CHANNELS_2][KERNEL_SIZE][KERNEL_SIZE],       //3x3x2x4 - 4 2 3 3
float bias[KERNEL_CHANNELS_COM],
float OFM[CONV_OUT_CHANNELS_COM][CONV_OUT_HEIGHT][CONV_OUT_WIDTH]
)*/

void ring_2_conv2d(
float IFM[2][8][8],
float F[1][2][3][3],       //3x3x2x4 - 4 2 3 3
float OFM[1][6][6]
){

  // Initialize output feature map to zero
	init_output:
	for (int h = 0; h < CONV_OUT_HEIGHT; h++) {

		for (int w = 0; w < CONV_OUT_WIDTH; w++) {

			#pragma HLS PIPELINE II=1

			for (int oc = 0; oc < CONV_OUT_CHANNELS_COM; oc++) {
				OFM[oc][h][w] = 0;
			}
		}
	}


	ring_conv:

	for(int H=KERNEL_SIZE-1; H<CONV_IN_HEIGHT; H=H+CONV_STRIDE){

		if(H%2==0){
			                 //W=KERNEL_SIZE-1;     // for changing left shifting to right shifting
			for(int W=KERNEL_SIZE-1; W<CONV_IN_WIDTH; W=W+CONV_STRIDE){ // HERE considering the STRIDE

				for(int C=0; C<CONV_OUT_CHANNELS_COM; C=C+1){

					//OFM[C][H+1-KERNEL_SIZE][W+1-KERNEL_SIZE]+=bias[C];

					// this loop is for multiple channels of IFM and single 3D kernel
					for(int ks=0;ks<CONV_IN_CHANNELS_2;ks++){

						//single channel OFM local value , all channels value added in end
						//float local_OFM[CONV_OUT_HEIGHT][CONV_OUT_WIDTH]={0.0f};
						  float local=0.0f;

						for(int i=0; i<KERNEL_SIZE;i=i+1){

							for(int j=0; j<KERNEL_SIZE; j=j+1){


								//local_OFM[H+1-KERNEL_SIZE][W+1-KERNEL_SIZE]+=IFM[ks][H+1+i-KERNEL_SIZE][W+j-KERNEL_SIZE+1]*F[C][ks][i][j];
								  local+=IFM[ks][H+1+i-KERNEL_SIZE][W+j-KERNEL_SIZE+1]*F[C][ks][i][j];

							}

						}
						// add all the single-channel values to main OFM
						//OFM[C][H+1-KERNEL_SIZE][W+1-KERNEL_SIZE]+=local_OFM[H+1-KERNEL_SIZE][W+1-KERNEL_SIZE];
						OFM[C][H+1-KERNEL_SIZE][W+1-KERNEL_SIZE]+=local;
				   }
				}
			}
		}

		else {
			//W=conv_in_width-KERNEL_SIZE;     // for changing right shifting to left shifting
			for(int W=CONV_IN_WIDTH-KERNEL_SIZE; W>=0; W=W-CONV_STRIDE){// here considering the stride

				for(int C=0; C<CONV_OUT_CHANNELS_COM; C=C+1){

					//OFM[C][H+1-KERNEL_SIZE][W+1-KERNEL_SIZE]+=bias[C];

					// this loop is for multiple channels of IFM and single 3D kernel
					for(int ks=0;ks<CONV_IN_CHANNELS_2;ks++){

						//single channel OFM local value , all channels value added in end
						//float local_OFM2[CONV_OUT_HEIGHT][CONV_OUT_WIDTH]={0.0f};
						  float local2=0.0f;

						for(int i=0; i<KERNEL_SIZE;i=i+1){

							for(int j=0; j<KERNEL_SIZE; j=j+1){


								//local_OFM2[H+1-KERNEL_SIZE][W]+=IFM[ks][H+1+i-KERNEL_SIZE][W+j]*F[C][ks][i][j];
								  local2+=IFM[ks][H+1+i-KERNEL_SIZE][W+j]*F[C][ks][i][j];

							}

						}
						// add all the single-channel values to main OFM
						//OFM[C][H+1-KERNEL_SIZE][W]+=local_OFM2[H+1-KERNEL_SIZE][W+1-KERNEL_SIZE];
						OFM[C][H+1-KERNEL_SIZE][W]+=local2;
					}


				}



			}


		}


	}

//OFM

// end
};

