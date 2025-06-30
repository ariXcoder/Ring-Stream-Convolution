
#include <iostream>
#include "hls_stream.h"

#define CONV_IN_HEIGHT    8
#define CONV_IN_WIDTH     8
#define CONV_IN_CHANNELS  6
#define CONV_OUT_CHANNELS 6
#define KERNEL_SIZE       3
#define CONV_STRIDE       1
#define KERNEL_CHANNELS   6     // same as the no. of CONV_OUT_CHANNELS

#define padding           0     // user defined, not an input argument
// Calculate output dimensions
#define CONV_OUT_HEIGHT ((CONV_IN_HEIGHT - KERNEL_SIZE+2*padding) / CONV_STRIDE + 1)  // same as CONV_INPUT
#define CONV_OUT_WIDTH ((CONV_IN_WIDTH - KERNEL_SIZE+2*padding) / CONV_STRIDE + 1)

void padded_ring_conv2d(
float IFM[CONV_IN_CHANNELS][CONV_IN_HEIGHT][CONV_IN_WIDTH],
float F[KERNEL_CHANNELS][CONV_IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
/*float bias[KERNEL_CHANNELS],*/
float OFM[CONV_OUT_CHANNELS][CONV_OUT_HEIGHT][CONV_OUT_WIDTH]
){

	  /// Array partitioning for input feature maps - complete partition for parallel access
	#pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
	#pragma HLS ARRAY_PARTITION variable=IFM cyclic factor=2 dim=2
	#pragma HLS ARRAY_PARTITION variable=IFM cyclic factor=2 dim=3

	// Array partitioning for filters - complete partition for kernel dimensions
	#pragma HLS ARRAY_PARTITION variable=F complete dim=1
	#pragma HLS ARRAY_PARTITION variable=F complete dim=2
	#pragma HLS ARRAY_PARTITION variable=F complete dim=3
	#pragma HLS ARRAY_PARTITION variable=F complete dim=4

	// Array partitioning for output feature maps
	#pragma HLS ARRAY_PARTITION variable=OFM complete dim=1
	#pragma HLS ARRAY_PARTITION variable=OFM cyclic factor=2 dim=2
	#pragma HLS ARRAY_PARTITION variable=OFM cyclic factor=2 dim=3

	// Initialize output feature map to zero
	init_output:
	for (int h = 0; h < CONV_OUT_HEIGHT; h++) {

		for (int w = 0; w < CONV_OUT_WIDTH; w++) {

			#pragma HLS PIPELINE II=1

			for (int oc = 0; oc < CONV_OUT_CHANNELS; oc++) {
				OFM[oc][h][w] = 0;
			}
		}
	}


	ring_conv:

	for(int H=KERNEL_SIZE-1; H<CONV_IN_HEIGHT+2*padding; H=H+CONV_STRIDE){

		if(H%2==0){
							 //W=KERNEL_SIZE-1;     // for changing left shifting to right shifting
			for(int W=KERNEL_SIZE-1; W<CONV_IN_WIDTH+2*padding; W=W+CONV_STRIDE){ // HERE considering the STRIDE

				for(int C=0; C<CONV_OUT_CHANNELS; C=C+1){

					//OFM[C][H+1-KERNEL_SIZE][W+1-KERNEL_SIZE]+=bias[C];

					// this loop is for multiple channels of IFM and single 3D kernel
					for(int ks=0;ks<CONV_IN_CHANNELS;ks++){
	#pragma HLS PIPELINE II=1
						//single channel OFM local value , all channels value added in end
						//float local_OFM[CONV_OUT_HEIGHT][CONV_OUT_WIDTH]={0.0f};
						  float local=0.0f;

						for(int i=0; i<KERNEL_SIZE;i=i+1){
	#pragma HLS UNROLL
							for(int j=0; j<KERNEL_SIZE; j=j+1){
	#pragma HLS UNROLL
								int wtemp=W+j-KERNEL_SIZE+1; // index for width of ifm
								int htemp=H+1+i-KERNEL_SIZE; // index for hieght of ifm

								if(wtemp<=padding-1 || wtemp >=CONV_IN_WIDTH+padding){
									local+=0.0f;
								}
								else if(htemp<=padding-1 || htemp >=CONV_IN_HEIGHT+padding){
									local+=0.0f;
								}
								else{
									local+=IFM[ks][H+1+i-KERNEL_SIZE-padding][W+j-KERNEL_SIZE+1-padding]*F[C][ks][i][j];
								}
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

			///change Padding W wala in this *********************

			//W=conv_in_width-KERNEL_SIZE;     // for changing right shifting to left shifting
			for(int W=CONV_IN_WIDTH-KERNEL_SIZE+padding+1; W>=0; W=W-CONV_STRIDE){// here considering the stride

				for(int C=0; C<CONV_OUT_CHANNELS; C=C+1){

					//OFM[C][H+1-KERNEL_SIZE][W+1-KERNEL_SIZE]+=bias[C];

					// this loop is for multiple channels of IFM and single 3D kernel
					for(int ks=0;ks<CONV_IN_CHANNELS;ks++){
	#pragma HLS PIPELINE II=1
						//single channel OFM local value , all channels value added in end
						//float local_OFM2[CONV_OUT_HEIGHT][CONV_OUT_WIDTH]={0.0f};
						  float local2=0.0f;

						for(int i=0; i<KERNEL_SIZE;i=i+1){
	#pragma HLS UNROLL
							for(int j=0; j<KERNEL_SIZE; j=j+1){
	#pragma HLS UNROLL
								int wtemp1=W+j; // index for width of ifm
								int htemp1=H+1+i-KERNEL_SIZE; // index for hieght of ifm

								if(wtemp1<=padding-1 || wtemp1 >=CONV_IN_WIDTH+padding){
									local2+=0.0f;
								}
								else if(htemp1<=padding-1 || htemp1 >=CONV_IN_HEIGHT+padding){
									local2+=0.0f;
								}
								else{
									local2+=IFM[ks][H+1+i-KERNEL_SIZE-padding][W+j-padding]*F[C][ks][i][j];

								}



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


};

