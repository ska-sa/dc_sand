
// Server side implementation of UDP client-server model 
#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h> 
#include <string.h> 
#include <sys/types.h> 
#include <sys/types.h>  //For networking
#include <sys/socket.h> //For networking
#include <arpa/inet.h>  //For networking
#include <netinet/in.h>  //For networking
#include <sys/time.h> //For timing functions

#include "network_packets.h"
  
#define MAXLINE 1024 
  
// Driver code 
int main() { 
    int sockfd; 
    char buffer[MAXLINE]; 
    char *hello = "Hello from server"; 
    struct sockaddr_in servaddr, cliaddr; 

    int iTotalTransmitBytes = NUMBER_OF_PACKETS*sizeof(struct UdpTestingPacket);
    struct UdpTestingPacket * psReceiveBuffer = malloc(iTotalTransmitBytes);

    struct timeval * psRxTimes = malloc(NUMBER_OF_PACKETS*sizeof(struct UdpTestingPacket));
      
    // Creating socket file descriptor 
    if ( (sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0 ) { 
        perror("socket creation failed"); 
        exit(EXIT_FAILURE); 
    } 
      
    memset(&servaddr, 0, sizeof(servaddr)); 
    memset(&cliaddr, 0, sizeof(cliaddr)); 
      
    // Filling server information 
    servaddr.sin_family    = AF_INET; // IPv4 
    servaddr.sin_addr.s_addr = INADDR_ANY; 
    servaddr.sin_port = htons(UDP_TEST_PORT); 
      
    // Bind the socket with the server address 
    if ( bind(sockfd, (const struct sockaddr *)&servaddr,  
            sizeof(servaddr)) < 0 ) 
    { 
        perror("bind failed"); 
        exit(EXIT_FAILURE); 
    } 
      
    int len, n; 
  
    len = sizeof(cliaddr);  //len is value/resuslt 
  
    // n = recvfrom(sockfd, (char *)buffer, MAXLINE,  
    //             MSG_WAITALL, ( struct sockaddr *) &cliaddr, 
    //             &len); 
    // buffer[n] = '\0'; 

    //printf("Original Message Received\n");

    for (size_t i = 0; i < 100000; i++)
    {
        printf("Waiting for stream\n");
        
        struct timeval stop, start;
        gettimeofday(&start, NULL);
        for (size_t i = 0; i < NUMBER_OF_PACKETS; i++)
        {
            n = recvfrom(sockfd, (const char *)&psReceiveBuffer[i], sizeof(struct UdpTestingPacket)*2,  
                    MSG_WAITALL, ( struct sockaddr *) &cliaddr, 
                    &len); 
            if(n != sizeof(struct UdpTestingPacket)){
                printf("******More than a single packet was received: %d *****",n);
            }
            if(i == 0){
                gettimeofday(&start, NULL);
            }
            gettimeofday(&psRxTimes[i], NULL);
            //printf("Received Packet %d %d.\n",i,psReceiveBuffer[i].header.i32PacketIndex); 
        }
        gettimeofday(&stop, NULL);
        printf("All Messages Received\n");

        float fTimeTaken_s = (stop.tv_sec - start.tv_sec) + ((float)(stop.tv_usec - start.tv_usec))/1000000;
        double fDataRate_Gibps = ((double)iTotalTransmitBytes-sizeof(struct UdpTestingPacket))*8.0/fTimeTaken_s/1024.0/1024.0/1024.0;
        printf("It took %f seconds to receive %d bytes of data (%d packets)\n", fTimeTaken_s,iTotalTransmitBytes,NUMBER_OF_PACKETS-1);
        printf("Data Rate: %f Gibps\n",fDataRate_Gibps); 

        double dRxTime_prev = (double)psRxTimes[0].tv_sec + ((double)psRxTimes[0].tv_usec)/1000000.0;
        double dTxTime_prev = (double)psReceiveBuffer[0].header.cTransmitTime.tv_sec + ((double)psReceiveBuffer[0].header.cTransmitTime.tv_usec)/1000000.0;
        double dMinTxRxDiff=1,dMinTxTxDiff=1,dMinRxRxDiff=1;
        double dMaxTxRxDiff=-1,dMaxTxTxDiff=-1,dMaxRxRxDiff=-1;
        double dAvgTxRxDiff=0,dAvgTxTxDiff=0,dAvgRxRxDiff=0;

        for (size_t i = 0; i < NUMBER_OF_PACKETS; i++)
        {
            if(i != psReceiveBuffer[i].header.i32PacketIndex){
                printf("Data received out of order\n");
            }
            double dTxTime = (double)psReceiveBuffer[i].header.cTransmitTime.tv_sec + ((double)psReceiveBuffer[i].header.cTransmitTime.tv_usec)/1000000.0;
            double dRxTime = (double)psRxTimes[i].tv_sec + ((double)psRxTimes[i].tv_usec)/1000000.0;

            double dDiffRxTx = dRxTime-dTxTime;
            dAvgTxRxDiff+=dDiffRxTx;
            if(dDiffRxTx < dMinTxRxDiff && dDiffRxTx != 0){
                dMinTxRxDiff = dDiffRxTx;
            }
            if(dDiffRxTx > dMaxTxRxDiff){
                dMaxTxRxDiff = dDiffRxTx;
            }

            double dDiffRxRx = dRxTime-dRxTime_prev;
            dAvgRxRxDiff+=dDiffRxRx;
            if(dDiffRxRx < dMinRxRxDiff && dDiffRxRx != 0){
                dMinRxRxDiff = dDiffRxRx;
            }
            if(dDiffRxRx > dMaxRxRxDiff){
                dMaxRxRxDiff = dDiffRxRx;
            }

            double dDiffTxTx = dTxTime-dTxTime_prev;
            dAvgTxTxDiff+=dDiffTxTx;
            if(dDiffTxTx < dMinTxTxDiff && dDiffTxTx != 0){
                dMinTxTxDiff = dDiffTxTx;
            }
            if(dDiffTxTx > dMaxTxTxDiff){
                dMaxTxTxDiff = dDiffTxTx;
            }

            printf("Packet %d TX %fs, RX %fs, Diff RX/TX %fs, Diff TX/TX %fs, Diff RX/RX %fs\n",i,dTxTime,dRxTime,dDiffRxTx,dDiffTxTx,dDiffRxRx);
            dRxTime_prev = dRxTime;
            dTxTime_prev = dTxTime;
        }
        dAvgTxRxDiff = dAvgTxRxDiff/(NUMBER_OF_PACKETS-1);

        printf("TX/RX|%f|%f|%f|\n",dAvgTxRxDiff,dMinTxRxDiff,dMaxTxRxDiff);
        printf("TX/TX|%f|%f|%f|\n",dAvgTxTxDiff,dMinTxTxDiff,dMaxTxTxDiff);
        printf("RX/RX|%f|%f|%f|\n",dAvgRxRxDiff,dMinRxRxDiff,dMaxRxRxDiff);
        


        sendto(sockfd, (const char *)hello, strlen(hello),  
        MSG_CONFIRM, (const struct sockaddr *) &cliaddr, 
            len); 
        printf("Hello message sent.\n");  

        printf("\n");
    }

    return 0; 
} 
