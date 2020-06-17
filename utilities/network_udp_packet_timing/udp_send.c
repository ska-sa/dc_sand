/**
 * @file    udp_send.c
 *
 * @brief   File that streams UDP data to a central server 
 * 
 * This client transmits according to infromation received from a server. See \ref udp_receive.c for more information.
 *  
 * @author  Gareth Callanan
 *          South African Radio Astronomy Observatory(SARAO)
 */


/* _GNU_SOURCE gives access to the GNU source, specifically needed for the sendmmsg function. sendmmsg has not yet been 
 * implemented.
 */
#define _GNU_SOURCE     

#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h>     //Useful functions like sleep,close and getopt
#include <string.h>     //For memset function
#include <sys/types.h>  //For networking
#include <sys/socket.h> //For networking
#include <arpa/inet.h>  //For networking
#include <netinet/in.h> //For networking
#include <sys/time.h>   //For timing functions

#include "network_packets.h"

#define SERVER_ADDRESS              "10.100.18.14"//TODO: Make a parameter
#define LOCAL_ADDRESS               "127.0.0.1"//TODO: Make default value when now paramter is provided.
#define NUMBER_RINGBUFFER_PACKETS   100000
// Driver code 
int main() 
{ 
    int iSocketFileDescriptor; 
    struct sockaddr_in     sServAddr; 

    //1. ***** Create sample data to be sent *****
    size_t ulMaximumTransmitBytes = NUMBER_RINGBUFFER_PACKETS*sizeof(struct UdpTestingPacket);
    struct UdpTestingPacket * psSendBuffer = malloc(ulMaximumTransmitBytes);
    for (size_t i = 0; i < NUMBER_RINGBUFFER_PACKETS; i++)
    {
        psSendBuffer[i].sHeader.i32TrailingPacket = 0;
    }
    
  
    //2. ***** Creating socket file descriptor *****
    if ( (iSocketFileDescriptor = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0 ) 
    { 
        perror("socket creation failed"); 
        exit(EXIT_FAILURE); 
    } 
  
    memset(&sServAddr, 0, sizeof(sServAddr)); 
      
    // Filling server information 
    sServAddr.sin_family = AF_INET; 
    sServAddr.sin_port = htons(UDP_TEST_PORT);
    sServAddr.sin_addr.s_addr = inet_addr(SERVER_ADDRESS);
      
    int iReceivedBytes, iSockAddressLength; 

    //3. ***** Send Initial Message To Server and receive response*****
    struct MetadataPacketClient sHelloPacket = {CLIENT_MESSAGE_HELLO,0};
    struct MetadataPacketMaster sConfigurationPacket;
    int iMessagesSent=0;
    while(1){
        sendto(iSocketFileDescriptor, (const struct MetadataPacketClient *)&sHelloPacket, 
            sizeof(struct MetadataPacketClient), 
            MSG_CONFIRM, (const struct sockaddr *) &sServAddr,  
                sizeof(sServAddr)); 
        
        printf("%d Hello message sent.\n",iMessagesSent++); 
        sleep(1);
    
        
        iReceivedBytes = recvfrom(iSocketFileDescriptor, (struct MetadataPacketMaster *)&sConfigurationPacket, 
                    sizeof(struct MetadataPacketMaster),  
                    MSG_DONTWAIT, (struct sockaddr *) &sServAddr, 
                    &iSockAddressLength);
        
        if(iReceivedBytes != -1){
            break;
        }
    }

    //4. ***** Parse Response from Server with Configuration Information *****
    if(sConfigurationPacket.u32MetadataPacketCode != SERVER_MESSAGE_CONFIGURATION)
    {
        printf("ERROR: Unexpected Message received from server\n");
        return 1;
    }
    printf("Configuration Message Received from Server\n");
    int iNumWindows = sConfigurationPacket.uNumberOfRepeats;

    struct timeval * psStopTime = malloc(sizeof(struct timeval)*iNumWindows);
    struct timeval * psStartTime = malloc(sizeof(struct timeval)*iNumWindows);
    int *  piNumberOfPacketsSentPerWindow = malloc(sizeof(int)*iNumWindows);
    int64_t i64NumPacketsSentTotal = 0;

    double dWindowTransmitTime = sConfigurationPacket.sSpecifiedTransmitTimeLength.tv_sec + \
            ((double)(sConfigurationPacket.sSpecifiedTransmitTimeLength.tv_usec))/1000000.0;
    double dTimeBetweenWindows = dWindowTransmitTime + ((double)sConfigurationPacket.i32DeadTime_us) /1000000.0;


    printf("Waiting Until Specified Time To Transmit Data\n");
    //5. ***** Stream data to server - a number of windows have to be transferred *****
    for (size_t i = 0; i < iNumWindows; i++)
    {
        //5.1 ***** Determine the time to wait until before transmitting current window. *****
        double dTimeToStart_s = sConfigurationPacket.sSpecifiedTransmitStartTime.tv_sec + \
                ((double)sConfigurationPacket.sSpecifiedTransmitStartTime.tv_usec)/1000000.0;
        dTimeToStart_s = dTimeToStart_s + (dTimeBetweenWindows)*i*sConfigurationPacket.uNumClients;

        //5.2 ***** Wait until determined time. *****
        double dCurrentTime_s = 0;
        do
        {
            gettimeofday(&psStartTime[i], NULL);
            dCurrentTime_s = psStartTime[i].tv_sec + ((double)psStartTime[i].tv_usec)/1000000.0;
        } while(dTimeToStart_s > dCurrentTime_s);

        //5.3 ***** Transmit data contionusly for the entire window period. *****
        piNumberOfPacketsSentPerWindow[i] = 0;
        double dTransmittedTime_s = 0;
        double dEndTime = dTimeToStart_s + dWindowTransmitTime;
        do
        {
            size_t ulPacketRingbufferIndex = i64NumPacketsSentTotal % NUMBER_RINGBUFFER_PACKETS;
            //Fill in packet header data before transmitting
            gettimeofday(&psSendBuffer[ulPacketRingbufferIndex].sHeader.sTransmitTime, NULL);
            psSendBuffer[ulPacketRingbufferIndex].sHeader.i64PacketIndex = i64NumPacketsSentTotal;
            psSendBuffer[ulPacketRingbufferIndex].sHeader.i64TransmitWindowIndex = i;
            psSendBuffer[ulPacketRingbufferIndex].sHeader.i32ClientIndex = sConfigurationPacket.i32ClientIndex;

            dTransmittedTime_s = (double)psSendBuffer[ulPacketRingbufferIndex].sHeader.sTransmitTime.tv_sec + \
                    ((double)(psSendBuffer[ulPacketRingbufferIndex].sHeader.sTransmitTime.tv_usec))/1000000.0;

            int temp = sendto(iSocketFileDescriptor, (const char *)&psSendBuffer[ulPacketRingbufferIndex], 
                    sizeof(struct UdpTestingPacket), 
                    0, (const struct sockaddr *) &sServAddr,  
                    sizeof(sServAddr)); 
            piNumberOfPacketsSentPerWindow[i]++;
            i64NumPacketsSentTotal++;
            if(temp != sizeof(struct UdpTestingPacket))
            {
                printf("Error Transmitting Data: %d",temp);
                return 1;
            }
        }while(dTransmittedTime_s < dEndTime);

        gettimeofday(&psStopTime[i], NULL);
    }


    /*  6 ***** When stream is complete, wait for a length of time specified by the server and then transmit trailing 
     *  packets. These tell the server that the transfer is complete. They are transmitted a number of times as UDP is 
     *  unreliable *****
     */
    sleep(sConfigurationPacket.fWaitAfterStreamTransmitted_s);
    printf("Transmitting Trailing Packets to server\n");
    struct UdpTestingPacket sTrailingPacket;
    sTrailingPacket.sHeader.i32TrailingPacket = 1;
    sTrailingPacket.sHeader.i64PacketsSent = i64NumPacketsSentTotal;
    sTrailingPacket.sHeader.i32ClientIndex = sConfigurationPacket.i32ClientIndex;
    //Transmit a few times to be safe - this is UDP after all
    for (size_t i = 0; i < 5; i++)
    {
        int temp = sendto(iSocketFileDescriptor, (const char *)&sTrailingPacket, sizeof(struct UdpTestingPacket), 
            0, (const struct sockaddr *) &sServAddr,  
            sizeof(sServAddr)); 
        if(temp != sizeof(struct UdpTestingPacket))
        {
            printf("Error Transmitting Data: %d",temp);
            return 1;
        }
    }

    //7 ***** Print out some useful diagnostic information *****
    for (size_t i = 0; i < iNumWindows; i++)
    {
        double dTimeTaken_s = (psStopTime[i].tv_sec - psStartTime[i].tv_sec) + 
                ((double)(psStopTime[i].tv_usec - psStartTime[i].tv_usec))/1000000;
        int iTotalTransmitBytes = piNumberOfPacketsSentPerWindow[i]*sizeof(struct UdpTestingPacket);
        double dDataRate_Gibps = ((double)iTotalTransmitBytes)*8.0/dTimeTaken_s/1024.0/1024.0/1024.0;
        printf("Window %ld\n",i);
        printf("\tIt took %f seconds to transmit %d bytes of data(%d packets)\n", 
                dTimeTaken_s,iTotalTransmitBytes,piNumberOfPacketsSentPerWindow[i]);
        printf("\tData Rate: %f Gibps\n",dDataRate_Gibps); 
    }

    printf("Program Done\n");
 
    free(psStopTime);
    free(psStartTime);
    free(psSendBuffer);
    close(iSocketFileDescriptor); 
    return 0; 
} 
