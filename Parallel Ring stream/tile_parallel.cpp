#include <iostream>
#include "hls_stream.h"
using namespace std;

#define CONV_IN_HEIGHT    224
#define CONV_IN_WIDTH     224
#define CONV_IN_CHANNELS  64
#define CONV_OUT_CHANNELS 64
#define KERNEL_SIZE       3
#define CONV_STRIDE       1
#define KERNEL_CHANNELS   CONV_OUT_CHANNELS
#define padding           1     // user defined, not an input argument

// Calculate output dimensions
#define CONV_OUT_HEIGHT ((CONV_IN_HEIGHT - KERNEL_SIZE+2*padding) / CONV_STRIDE + 1)
#define CONV_OUT_WIDTH ((CONV_IN_WIDTH - KERNEL_SIZE+2*padding) / CONV_STRIDE + 1)

// ring convolution function calling

#define N  CONV_IN_CHANNELS
#define M  CONV_OUT_CHANNELS
#define Ri CONV_IN_HEIGHT
#define Ci CONV_IN_WIDTH
#define Ro CONV_OUT_HEIGHT
#define Co CONV_OUT_WIDTH
#define K  KERNEL_SIZE

// tile parameter
#define Tn 2
#define Tm 64

#define Pn CONV_IN_CHANNELS/Tn
#define Qn CONV_IN_CHANNELS-Pn*Tn

#define Pm CONV_OUT_CHANNELS/Tm
#define Qm CONV_OUT_CHANNELS-Pm*Tm
#define KERNEL_CHANNELS_COM   1
#define CONV_OUT_CHANNELS_COM 1


#define CONV_IN_CHANNELS_1 2 // same as Tn

#define CONV_IN_CHANNELS_2 1  // same as N-Pn*Tn=Qn


void parallel_1_conv2d(
float IFM[CONV_IN_CHANNELS_1][CONV_IN_HEIGHT][CONV_IN_WIDTH],
float F[KERNEL_CHANNELS_COM][CONV_IN_CHANNELS_1][KERNEL_SIZE][KERNEL_SIZE],
float OFM[CONV_OUT_CHANNELS_COM][CONV_OUT_HEIGHT][CONV_OUT_WIDTH]
);

/*
void parallel_2_conv2d(
float IFM[1][CONV_IN_HEIGHT][CONV_IN_WIDTH],
float F[KERNEL_CHANNELS_COM][1][KERNEL_SIZE][KERNEL_SIZE],
float OFM[CONV_OUT_CHANNELS_COM][CONV_OUT_HEIGHT][CONV_OUT_WIDTH]
);
*/




// ****** MAIN FUNCTION *******

void tile_parallel_conv2d(
float IFM[CONV_IN_CHANNELS][CONV_IN_HEIGHT][CONV_IN_WIDTH],
float F[KERNEL_CHANNELS][CONV_IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],       //3x3x2x4 - 4 2 3 3
float OFM[CONV_OUT_CHANNELS][CONV_OUT_HEIGHT][CONV_OUT_WIDTH]
){



// 37 K ke liye , not copied from padding wala

	//Array partitioning for input feature maps - complete partition for parallel access



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




			// OUTPUT INITIALIZE
			OUTPUT_INIT_R: for(int r=0;r<Ro;r++){
		#pragma HLS PIPELINE II=4
				OUTPUT_INIT_C: for(int c=0;c<Co;c++){
					OUTPUT_INIT_OUT: for(int out=0;out<CONV_OUT_CHANNELS;out++){
		#pragma HLS UNROLL
						OFM[out][r][c]=0.0f;
					}
				}
			}
			//cout<<"output initialised "<<endl;

			MAIN_TILE_LOOP: for(int i=0;i<Pm;i++){
		#pragma HLS LOOP_TRIPCOUNT min=1 max=3
				//cout<<endl<<" !! repeating tile iteration- "<<i+1<<endl;

				int tm_d=(Tm)*i;

				//This loop selects which Tm no. of kernels are to be used ( here 2 , hencei=0-> 1&2 i=1->3&4.i=2->3&6),total kernel=6

				KERNEL_LOOP: for(tm_d;tm_d<Tm+i*(Tm);tm_d++){ /// this loop runs for Tm times, generating 1 OFM at a time
		//#pragma HLS LOOP_TRIPCOUNT min=2 max=2

					CHANNEL_TILE_LOOP: for(int p=0;p<Pn;p++){   // this loop runs for Pn times, generating
		//#pragma HLS LOOP_TRIPCOUNT min=1 max=2
		//#pragma HLS PIPELINE II=1

						float partial_IFM[2][224][224]; // Tn Ri Ci
						float partial_F[1][2][3][3];  // always 1 , then K x K
						float partial_OFM[1][224][224]; // 1 Ro Co ( 1 always)




		#pragma HLS ARRAY_PARTITION variable=partial_IFM complete dim=1
		#pragma HLS ARRAY_PARTITION variable=partial_F complete dim=1
		#pragma HLS ARRAY_PARTITION variable=partial_F complete dim=2
		#pragma HLS ARRAY_PARTITION variable=partial_OFM complete dim=1



						int tn_d=(Tn)*p;
						//cout<<"copying Pn*Tn partial IFM/F for"<<p<<"th iteration"<<endl;
						COPY_PARTIAL_DATA: for(tn_d;tn_d<Tn+(Tn)*p;tn_d++){
//#pragma HLS UNROLL
							//partial IFM
							COPY_IFM_R: for(int rr=0;rr<Ri;rr++){
#pragma HLS PIPELINE II=1
								COPY_IFM_C: for(int cc=0;cc<Ci;cc++){
#pragma HLS UNROLL factor=15
									partial_IFM[tn_d-(Tn)*p][rr][cc]=IFM[tn_d][rr][cc];
								}
							}
							//partial Filter
							COPY_F_R: for(int rr=0;rr<K;rr++){
#pragma HLS PIPELINE II=1
								COPY_F_C: for(int cc=0;cc<K;cc++){
		//#pragma HLS UNROLL //this one ***
									partial_F[0][tn_d-(Tn)*p][rr][cc]=F[tm_d][tn_d][rr][cc];
								}
							}
						}
						//cout<<"function calling -1 for "<<p<<" iteration "<<endl;

						// function call for getting Tn valued Single channel OFM (partial) // function call-1
						parallel_1_conv2d(partial_IFM,partial_F,partial_OFM);
						//cout<<endl<<"1st function calling done *******"<<p<<" iteration "<<endl;

						ACCUMULATE_OFM: for(int h=0;h<Ro;h++){
#pragma HLS PIPELINE II=1
							for(int w=0;w<Co;w++){
#pragma HLS UNROLL factor=15
								OFM[tm_d][h][w]+=partial_OFM[0][h][w];
							}
						} // copied the Tn valued single channel partial OFM to original OFM , upto N valued

				    }//p=0->Pn
					//cout<<"single channel ofm repetition done for kernel channel-"<<tm_d+1<<endl;

					//***********************************************************************************************

					// to calculate for remaining N-Pn*Tn channels of IFM and single 3D kernel ( Qn part)
	/*				if(Qn!=0){
						float extra_par_IFM[1][224][224];  // Qn Ri Ci
						float extra_par_F[1][1][3][3];
						float extra_partial_OFM[1][10][10];



		#pragma HLS ARRAY_PARTITION variable=extra_par_IFM complete dim=1
		#pragma HLS ARRAY_PARTITION variable=extra_par_F complete dim=1
		#pragma HLS ARRAY_PARTITION variable=extra_par_F complete dim=2
		#pragma HLS ARRAY_PARTITION variable=extra_partial_OFM complete dim=1



						EXTRA_COPY_LOOP: for(int q=0;q<Qn;q++){
		#pragma HLS UNROLL

							//partial IFM
							EXTRA_IFM_R: for(int rr=0;rr<Ri;rr++){
		#pragma HLS PIPELINE II=1
								EXTRA_IFM_C: for(int cc=0;cc<Ci;cc++){
		#pragma HLS UNROLL factor=2
									extra_par_IFM[q][rr][cc]=IFM[q+Pn*Tn][rr][cc];
								}
							}
							//partial Filter
							EXTRA_F_R: for(int rr=0;rr<K;rr++){
		#pragma HLS PIPELINE II=1
								EXTRA_F_C: for(int cc=0;cc<K;cc++){
		//#pragma HLS UNROLL // this one***
									extra_par_F[0][q][rr][cc]=F[0][q+Pn*Tn][rr][cc];
								}
							}
						}
						//function calll*************** function -2
						parallel_2_conv2d(extra_par_IFM,extra_par_F,extra_partial_OFM);
						//cout<<"function calling-2 done *******" <<endl;
						//copy the N-Pn*Tn channels of IFM and F , completing the single OFM
						EXTRA_ACCUMULATE: for(int h=0;h<Ro;h++){
		#pragma HLS PIPELINE II=1
							for(int w=0;w<Co;w++){
		#pragma HLS UNROLL
								OFM[tm_d][h][w]+=extra_partial_OFM[0][h][w];
							}
						}// after this we get a complete single channeled OFM made from N depth IFM and kernel
					}*/

				}//t= 0->Tm-1 (i=0), Tm->2Tm (i=1) ......(Pm-1)*Tm-1-> Pm*Tm-1

				//cout<<endl<<"received single OFM tile of repetition for iteration-"<<i<<endl;
			}// i=0->Pm

			///****************************************************************************************


			//for remaining Qm channels of OFM
			// using same logic as above, instead of repeating Tm times , we repeat for Qm times

			//This loop will give Tm channels of OFM
/*
			REMAINING_OFM_LOOP: for(int qm=0;qm<Qm;qm++){
		//#pragma HLS LOOP_TRIPCOUNT min=0 max=2

				// this loop(along with Qn part) will deduce convolution of 1 IFM and 1 3D Kernel and give 1 channel of OFM
				REMAINING_CHANNEL_LOOP: for(int p2=0;p2<Pn;p2++){
		//#pragma HLS LOOP_TRIPCOUNT min=1 max=2
		//#pragma HLS PIPELINE II=1

					float partial_IFM2[2][224][224];
					float partial_F2[1][2][3][3];
					float partial_OFM2[1][10][10];


		#pragma HLS ARRAY_PARTITION variable=partial_IFM2 complete dim=1
		#pragma HLS ARRAY_PARTITION variable=partial_F2 complete dim=1
		#pragma HLS ARRAY_PARTITION variable=partial_F2 complete dim=2
		#pragma HLS ARRAY_PARTITION variable=partial_OFM2 complete dim=1


					int tn_d2=(Tn+1)*p2;
					REMAINING_COPY_LOOP: for(tn_d2;tn_d2<Tn+(Tn)*p2;tn_d2++){
		//#pragma HLS UNROLL
						//partial IFM
						REMAINING_IFM_R: for(int rr=0;rr<Ri;rr++){
		#pragma HLS PIPELINE II=1
							REMAINING_IFM_C: for(int cc=0;cc<Ci;cc++){
		#pragma HLS UNROLL factor=2
								partial_IFM2[tn_d2-(Tn)*p2][rr][cc]=IFM[tn_d2][rr][cc];
							}
						}
						//partial Filter
						REMAINING_F_R: for(int rr=0;rr<K;rr++){
		#pragma HLS PIPELINE II=1
							REMAINING_F_C: for(int cc=0;cc<K;cc++){
		//#pragma HLS UNROLL // this one***
								partial_F2[0][tn_d2-(Tn)*p2][rr][cc]=F[qm][tn_d2][rr][cc];
							}
						}
					}
					// function call for getting Tn valued Single channel OFM (partial) - function -3***********
					parallel_1_conv2d(partial_IFM2,partial_F2,partial_OFM2);

					REMAINING_ACCUMULATE: for(int h=0;h<Ro;h++){
		#pragma HLS PIPELINE II=1
						for(int w=0;w<Co;w++){
		#pragma HLS UNROLL
							OFM[qm+Pm*Tm][h][w]+=partial_OFM2[0][h][w];
						}
					} // copied the Tn valued single channel partial OFM to original OFM , upto N valued

				}//p=0->Pn

				// to calculate for remaining N-Pn*Tn=Qn channels of IFM and single 3D kernel ( Qn part)
				float extra_par_IFM2[1][224][224];
				float extra_par_F2[1][1][3][3];
				float extra_partial_OFM2[1][10][10];


		#pragma HLS ARRAY_PARTITION variable=extra_par_IFM2 complete dim=1
		#pragma HLS ARRAY_PARTITION variable=extra_par_F2 complete dim=1
		#pragma HLS ARRAY_PARTITION variable=extra_par_F2 complete dim=2
		#pragma HLS ARRAY_PARTITION variable=extra_partial_OFM2 complete dim=1


				FINAL_EXTRA_LOOP: for(int q2=0;q2<Qn;q2++){
		//#pragma HLS UNROLL

					//partial IFM
					FINAL_IFM_R: for(int rr=0;rr<Ri;rr++){
		#pragma HLS PIPELINE II=1
						FINAL_IFM_C: for(int cc=0;cc<Ci;cc++){
		#pragma HLS UNROLL factor=2
							extra_par_IFM2[q2][rr][cc]=IFM[q2+Pn*Tn][rr][cc];
						}
					}
					//partial Filter
					FINAL_F_R: for(int rr=0;rr<K;rr++){
		#pragma HLS PIPELINE II=1
						FINAL_F_C: for(int cc=0;cc<K;cc++){
		//#pragma HLS UNROLL // this one **
							extra_par_F2[0][q2][rr][cc]=F[0][q2+Pn*Tn][rr][cc];
						}
					}
				}
				// function call-4
				parallel_2_conv2d(extra_par_IFM2,extra_par_F2,extra_partial_OFM2);

				//copy the N-Pn*Tn channels of IFM and F , completing the single OFM
				FINAL_ACCUMULATE: for(int h=0;h<Ro;h++){
		#pragma HLS PIPELINE II=1
					for(int w=0;w<Co;w++){
		#pragma HLS UNROLL
						OFM[qm][h][w]+=extra_partial_OFM2[0][h][w];
					}
				}// after this we get a complete single channeled OFM made from N depth IFM and kernel

			}//qm=0->Qm
*/

		}
