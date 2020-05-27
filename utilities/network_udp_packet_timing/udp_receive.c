
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
    printf("Funnel In Test Server Started\n");
    int sockfd; 
    char buffer[MAXLINE]; 
    char *hello = "Hello from server"; 
    struct sockaddr_in servaddr, cliaddr; 

    int iTotalTransmitBytes = MAXIMUM_NUMBER_OF_PACKETS*sizeof(struct UdpTestingPacket);
    struct UdpTestingPacket * psReceiveBuffer = malloc(iTotalTransmitBytes);

    struct timeval * psRxTimes = malloc(MAXIMUM_NUMBER_OF_PACKETS*sizeof(struct UdpTestingPacket));
    

    //***** Creating socket file descriptor *****
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

    //Loop is here so that we can write multiple client tests
    for (size_t k = 0; k < 100000; k++)
    {
        
        //***** Waiting for initial hello message from client *****
        printf("Waiting For Hello Message From Client\n");
        struct MetadataPacketClient sHelloPacket = {CLIENT_MESSAGE_EMPTY,0};
        while(sHelloPacket.u32MetadataPacketCode != CLIENT_MESSAGE_HELLO){
            n = recvfrom(sockfd, (struct MetadataPacketClient *)&sHelloPacket, sizeof(struct MetadataPacketClient),  
                        MSG_WAITALL, ( struct sockaddr *) &cliaddr, 
                        &len); 
            printf("Message Received\n");
        }
        printf("Hello Message Received\n");
        
        //***** Determine and send Configuration Information to client *****
        printf("Sending Configuration Message to client\n");
        struct timeval sCurrentTime;
        gettimeofday(&sCurrentTime,NULL);

        struct MetadataPacketMaster sConfigurationPacket;
        sConfigurationPacket.u32MetadataPacketCode = SERVER_MESSAGE_CONFIGURATION;
        sConfigurationPacket.sSpecifiedTransmitStartTime.tv_sec = sCurrentTime.tv_sec + 1;
        sConfigurationPacket.sSpecifiedTransmitStartTime.tv_usec = 0;
        sConfigurationPacket.sSpecifiedTransmitStopTime.tv_sec = sCurrentTime.tv_sec + 1;
        sConfigurationPacket.sSpecifiedTransmitStopTime.tv_usec = 1000;
        sConfigurationPacket.fWaitAfterStreamTransmitted_s = 1;

        sendto(sockfd, (const struct MetadataPacketMaster *)&sConfigurationPacket, sizeof(struct MetadataPacketMaster),  
            MSG_CONFIRM, (const struct sockaddr *) &cliaddr, 
                len); 

        //***** Wait For Data stream messages from client *****
        printf("Waiting for stream\n");
        int32_t i32ReceivedPacketsCount = 0;
        struct timeval stop, start;
        gettimeofday(&start, NULL);
        for (;;)//For loop has been removed, trailing packets will instead indicate that the receiver must stop
        {
            n = recvfrom(sockfd, (const char *)&psReceiveBuffer[i32ReceivedPacketsCount], 
                    sizeof(struct UdpTestingPacket)*2, MSG_WAITALL, ( struct sockaddr *) &cliaddr, 
                    &len); 
            if(n != sizeof(struct UdpTestingPacket)){
                printf("******More than a single packet was received: %d *****",n);
                return 1;
            }
            if(psReceiveBuffer[i32ReceivedPacketsCount].sHeader.i32TrailingPacket != 0){
                printf("Trailing packet received indicating that some packets were dropped\n");
                
                break;
            }
            if(i32ReceivedPacketsCount == 0){
                gettimeofday(&start, NULL);
            }
            gettimeofday(&psRxTimes[i32ReceivedPacketsCount], NULL);
            i32ReceivedPacketsCount++;//Not counted in the case of a trailing packet
        }
        int i32TotalSentPackets = psReceiveBuffer[i32ReceivedPacketsCount].sHeader.i32PacketsSent;//Takes us to the \
        trailing packet which is not part of timing
        printf("All Messages Received\n");
        stop = psRxTimes[i32ReceivedPacketsCount-1]; //Set stop time equal to last received packet - not simply getting system time here as \
        trailing packets can take quite a while to arrive

        //***** Analyse data, and calculate and display performance metrics *****
        float fTimeTaken_s = (stop.tv_sec - start.tv_sec) + ((float)(stop.tv_usec - start.tv_usec))/1000000;
        double fDataRate_Gibps = ((i32ReceivedPacketsCount)*sizeof(struct UdpTestingPacket))
                *8.0/fTimeTaken_s/1024.0/1024.0/1024.0;

        double dRxTime_prev = (double)psRxTimes[0].tv_sec + ((double)psRxTimes[0].tv_usec)/1000000.0;
        double dTxTime_prev = (double)psReceiveBuffer[0].sHeader.sTransmitTime.tv_sec + 
                ((double)psReceiveBuffer[0].sHeader.sTransmitTime.tv_usec)/1000000.0;

        double dMinTxRxDiff=1,dMinTxTxDiff=1,dMinRxRxDiff=1;
        double dMaxTxRxDiff=-1,dMaxTxTxDiff=-1,dMaxRxRxDiff=-1;
        double dAvgTxRxDiff=0,dAvgTxTxDiff=0,dAvgRxRxDiff=0;

        uint8_t u8OutOfOrder = 0;
        for (size_t i = 0; i < i32ReceivedPacketsCount; i++)
        {
            if(i != 0 && psReceiveBuffer[i-1].sHeader.i32PacketIndex > psReceiveBuffer[i].sHeader.i32PacketIndex){
                printf("Data received out of order\n");
                u8OutOfOrder = 1;
            }
            double dTxTime = (double)psReceiveBuffer[i].sHeader.sTransmitTime.tv_sec + ((double)psReceiveBuffer[i].sHeader.sTransmitTime.tv_usec)/1000000.0;
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
        dAvgTxRxDiff = dAvgTxRxDiff/(i32ReceivedPacketsCount-1);
        dAvgTxTxDiff = dAvgTxTxDiff/(i32ReceivedPacketsCount-1);
        dAvgRxRxDiff = dAvgRxRxDiff/(i32ReceivedPacketsCount-1);

        printf("\n Average Time Between Packets\n");
        printf("     |  Avg(s) |  Min(s) |  Max(s) |\n");
        printf("TX/RX|%9.6f|%9.6f|%9.6f|\n",dAvgTxRxDiff,dMinTxRxDiff,dMaxTxRxDiff);
        printf("TX/TX|%9.6f|%9.6f|%9.6f|\n",dAvgTxTxDiff,dMinTxTxDiff,dMaxTxTxDiff);
        printf("RX/RX|%9.6f|%9.6f|%9.6f|\n",dAvgRxRxDiff,dMinRxRxDiff,dMaxRxRxDiff);
        printf("\n");
        printf("It took %f seconds to receive %d bytes of data (%d packets)\n", fTimeTaken_s,iTotalTransmitBytes,i32ReceivedPacketsCount-1);
        printf("Data Rate: %f Gibps\n",fDataRate_Gibps); 
        printf("\n");

        if(u8OutOfOrder != 0){
            printf("\n");
            printf("*********Data Received out of order********");
            printf("\n");
        }else{
            printf("\n");
            printf("Data Received in order");
            printf("\n");
        }

        printf("\n");
        printf("%d of %d packets received. Drop rate = %.2f\%\n",i32ReceivedPacketsCount,i32TotalSentPackets,(1-((double)i32ReceivedPacketsCount)/((double)i32TotalSentPackets))*100);
        printf("\n");

        double fDataRateAvg2_Gibps = ((double)sizeof(struct UdpTestingPacket))/dAvgTxTxDiff/1024.0/1024.0/1024.0*8;//*8 is for bit to byte conversion
        printf("Data Rate According to Average Packet Tx Time Difference: %f Gibps\n",fDataRateAvg2_Gibps);

        sendto(sockfd, (const char *)hello, strlen(hello),  
        MSG_CONFIRM, (const struct sockaddr *) &cliaddr, 
            len); 
        printf("Hello message sent.\n");  

        printf("\n");
    }

    close(sockfd);
    return 0; 
} 
