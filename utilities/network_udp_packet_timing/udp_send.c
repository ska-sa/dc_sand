
// Client side implementation of UDP client-server model 


#define _GNU_SOURCE //This gives access to the GNU source, specifically needed for the sendmmsg function

#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h> 
#include <string.h> 
#include <sys/types.h>  //For networking
#include <sys/socket.h> //For networking
#include <arpa/inet.h>  //For networking
#include <netinet/in.h>  //For networking
#include <sys/time.h> //For timing functions

#include "network_packets.h"

#define SERVER_ADDRESS  "10.100.101.1"//TODO: Make a parameter
#define LOCAL_ADDRESS  "127.0.0.1"//TODO: Make default value when now paramter is provided.
  
// Driver code 
int main() { 
    int sockfd; 
    struct sockaddr_in     servaddr; 

    //***** Create sample data to be sent *****
    int iMaximumTransmitBytes = MAXIMUM_NUMBER_OF_PACKETS*sizeof(struct UdpTestingPacket);
    struct UdpTestingPacket * psSendBuffer = malloc(iMaximumTransmitBytes);
    for (size_t i = 0; i < MAXIMUM_NUMBER_OF_PACKETS; i++)
    {
        psSendBuffer[i].sHeader.i32PacketIndex = i;
        psSendBuffer[i].sHeader.i32TrailingPacket = 0;
    }
    
  
    //***** Creating socket file descriptor *****
    if ( (sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0 ) { 
        perror("socket creation failed"); 
        exit(EXIT_FAILURE); 
    } 
  
    memset(&servaddr, 0, sizeof(servaddr)); 
      
    // Filling server information 
    servaddr.sin_family = AF_INET; 
    servaddr.sin_port = htons(UDP_TEST_PORT);
    servaddr.sin_addr.s_addr = inet_addr(LOCAL_ADDRESS);
      
    int n, len; 

    //***** Send Initial Message To Server *****
    struct MetadataPacketClient sHelloPacket = {CLIENT_MESSAGE_HELLO,0};
    sendto(sockfd, (const struct MetadataPacketClient *)&sHelloPacket, sizeof(struct MetadataPacketClient), 
         MSG_CONFIRM, (const struct sockaddr *) &servaddr,  
             sizeof(servaddr)); 
    printf("Hello message sent.\n"); 

    //**** Wait For Response from Server with Configuration Information
    struct MetadataPacketMaster sConfigurationPacket;
    n = recvfrom(sockfd, (struct MetadataPacketMaster *)&sConfigurationPacket, sizeof(struct MetadataPacketMaster),  
                MSG_WAITALL, (struct sockaddr *) &servaddr, 
                &len);
    if(sConfigurationPacket.u32MetadataPacketCode != SERVER_MESSAGE_CONFIGURATION){
        printf("ERROR: Unexpected Message received from server\n");
        return 1;
    }
    printf("Configuration Message Received from Server\n");

    //***** Wait until the time specified by the server before streaming *****
    printf("Waiting Until Specified Time To Transmit Data\n");
    struct timeval stop, start;

    double dTimeToStart_s = sConfigurationPacket.sSpecifiedTransmitStartTime.tv_sec + ((double)sConfigurationPacket.sSpecifiedTransmitStartTime.tv_usec)/1000000.0;
    double dCurrentTime_s = 0;
    do
    {
        gettimeofday(&start, NULL);
        dCurrentTime_s = start.tv_sec + ((double)start.tv_usec)/1000000.0;
    } while(dTimeToStart_s > dCurrentTime_s);
    
    
    //***** Stream data to server *****
    //This has to take place on a number of occasions
    int iNumberOfPacketsSent = 0;
    double dTransmittedTime_s = 0;
    double dEndTime = (double)sConfigurationPacket.sSpecifiedTransmitStartTime.tv_sec + (double)sConfigurationPacket.sSpecifiedTransmitTimeLength.tv_sec + ((double)(sConfigurationPacket.sSpecifiedTransmitTimeLength.tv_usec + sConfigurationPacket.sSpecifiedTransmitStartTime.tv_usec))/1000000.0;
    do
    {
        gettimeofday(&psSendBuffer[iNumberOfPacketsSent].sHeader.sTransmitTime, NULL);
        dTransmittedTime_s = (double)psSendBuffer[iNumberOfPacketsSent].sHeader.sTransmitTime.tv_sec + ((double)(psSendBuffer[iNumberOfPacketsSent].sHeader.sTransmitTime.tv_usec))/1000000.0;
        //printf("%f %f %d\n",dTransmittedTime,dEndTime,iNumberOfPacketsSent);
        //printf("%d %d \n",psSendBuffer[i].header.cTransmitTime.tv_sec,psSendBuffer[i].header.cTransmitTime.tv_usec);
        int temp = sendto(sockfd, (const char *)&psSendBuffer[iNumberOfPacketsSent], sizeof(struct UdpTestingPacket), 
        0, (const struct sockaddr *) &servaddr,  
            sizeof(servaddr)); 
        iNumberOfPacketsSent++;
        if(temp != sizeof(struct UdpTestingPacket)){
            printf("Error Transmitting Data: %d",temp);
            return 1;
        }
        //printf("Sent Packet %ld %d.\n",i,temp); 
    }while(dTransmittedTime_s < dEndTime);

    gettimeofday(&stop, NULL);
    double dTimeTaken_s = (stop.tv_sec - start.tv_sec) + ((double)(stop.tv_usec - start.tv_usec))/1000000;
    int iTotalTransmitBytes = iNumberOfPacketsSent*sizeof(struct UdpTestingPacket);
    double dDataRate_Gibps = ((double)iTotalTransmitBytes)*8.0/dTimeTaken_s/1024.0/1024.0/1024.0;
    printf("It took %f seconds to transmit %d bytes of data(%d packets)\n", dTimeTaken_s,iTotalTransmitBytes,iNumberOfPacketsSent);
    printf("Data Rate: %f Gibps\n",dDataRate_Gibps); 

    //***** Send Trailing Packets to stop server polling the receive socket *****
    sleep(sConfigurationPacket.fWaitAfterStreamTransmitted_s);
    struct UdpTestingPacket sTrailingPacket;
    sTrailingPacket.sHeader.i32TrailingPacket = 1;
    sTrailingPacket.sHeader.i32PacketsSent = iNumberOfPacketsSent;
    //Transmit a few times to be safe - this is UDP after all
    for (size_t i = 0; i < 10; i++)
    {
        int temp = sendto(sockfd, (const char *)&sTrailingPacket, sizeof(struct UdpTestingPacket), 
        0, (const struct sockaddr *) &servaddr,  
            sizeof(servaddr)); 
        if(temp != sizeof(struct UdpTestingPacket)){
            printf("Error Transmitting Data: %d",temp);
            return 1;
        }
    }
 
    close(sockfd); 
    return 0; 
} 
