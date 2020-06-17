/**
 * @file    udp_receive.c
 *
 * @brief   File that receives UDP data from multiple client for the purposes of testing network packet scheduling 
 *          accuracy.
 * 
 * This server receives streams of data from multiple clients(See \ref udp_send.c). The data is transmitted to the 
 * server in predefined windows in order to maximise line rate and reduce packet overlap between clients. All 
 * configuration is specified in this server. The server upon receiveing a request from a client will transfer its 
 * configuration implementation to the client. This allows all clients to be synchronised. 
 * 
 * Each client transmits for a specific window in time. The windows between clients are interleaved. The clients 
 * transmit a certain number of windows. The window length, number of windows and dead time between windows are all
 * specified by the user on program launch.
 * 
 * The received packet metadata is written to file for analysis by seperate scripts. 
 *
 * @author  Gareth Callanan
 *          South African Radio Astronomy Observatory(SARAO)
 */

#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h>     //Useful functions like sleep,close and getopt
#include <getopt.h>     //Useful functions for parsing command line parameters
#include <string.h>     //For memset function
#include <sys/types.h>  //For networking
#include <sys/socket.h> //For networking
#include <arpa/inet.h>  //For networking
#include <netinet/in.h> //For networking
#include <sys/time.h>   //For timing functions

#include "network_packets.h"

//Default values in case command line parameters are not given
#define TRANSMIT_WINDOW_US_DEFAULT 1000 
#define DEAD_TIME_US_DEFAULT 100 
#define TOTAL_WINDOWS_PER_CLIENT_DEFAULT 3 
#define TOTAL_CLIENTS_DEFAULT 2 
#define DONT_PRINT_TO_TERMINAL_DEFAULT 0
#define COMBINE_DEFAULT 0
#define FILE_NAME_DEFAULT "FunnelInTesting"

//This needs to be reworked
#define MAXIMUM_NUMBER_OF_PACKETS   500000000 //Should be able to store about 30 minutes worth of headers at 10 Gbps
#define NUMBER_RINGBUFFER_PACKETS   100000

/** This struct exists for when per window information is stored instead of per packet. This is necessary for longer 
 * tests where it is not practical to store the information of every packet.
 */ 
struct WindowInformation{
    struct timeval sFirstRxTime; //Time first received packet in window was received. ***
    struct timeval sFirstTxTime; //Time first received packet in window was transmitted. ***
    struct timeval sLastRxTime; //Time latest received packet in window was received. ***
    struct timeval sLastTxTime; //Time latest received packet in window was transmitted. ***
    int64_t i64TransmitWindowIndex;//A single client transmits over multiple windows. This value indicates the window \
    index per client. ***
    int32_t i32ClientIndex; //The index of this client this window corresponds to. ***
    int64_t i64PacketsReceived; //The total amount of packets received by this window. ***
    int64_t i64FirstPacketIndex; //The index of the first packet received in this window. ***
    int64_t i64LastPacketIndex; //The index of the last packet received in this window. ***
    int64_t i64MissingIndexes; //If a received packet index is more than one above the last packet index, this \
    indicates that a packet has most likely been dropped - when this occurs this counter is incremented. ***
    int64_t i64OutOfOrderIndexes; //If a received packet index is less than the last packet index, then a packet has \
    been received out of order. This is not expected to occur, however it is still checked for in order to fully \
    charectarise the network. ***
    int64_t i64OverlappingWindowsBack; //If two packets from seperate windows are received in an interleaved fashion, then \
    this is an unwanted overlap. If the overlap occurs at the end of the window, this counter is incremented. ***
    int64_t i64OverlappingWindowsFront; //If two packets from seperate windows are received in an interleaved fashion, then \
    this is an unwanted overlap. If the overlap occurs at the start of the window, this counter is incremented. ***
    double dMinTxRxDiff_s; //The minimum recorded travel time of a single packet. ***
    double dAvgTxRxDiff_s; //The average recorded travel time of a single packet. ***
    double dMaxTxRxDiff_s; //The maximum recorded travel time of a single packet. ***
};

/** \brief Function that parses all command line arguments
 *  
 *  \param[in]  argc                            Value passed into main() indicating the number of command line 
 *                                              parameters passed into the program.
 *  \param[in]  argv                            Array of strings passed into main() containing the different command 
 *                                              line parameters.
 *                              
 *  \param[out] pi8OutputFileName               Name of the file to write to.
 *  \param[out] u32TransmitWindowLength_us      Window length in microseconds.
 *  \param[out] u32DeadTime_us                  Deadtime between windows in microseconds.
 *  \param[out] u32TransmitWindowsPerClient     The number of windows that each client must transmit.
 *  \param[out] u32TotalClients                 The number of transmitter clients that connect to this receiver server.
 *  \param[out] u8NoTerminal                    If 1 indicates that the program must disable printing window information
 *                                              to the terminal.
 *  \param[out] u8Combine                       If 1, indicates that the program discard individual packet information 
 *                                              and instead collect statsistics on a per window basis to reduce RAM 
 *                                              requirements.
 * 
 *  \return Returns 1 if the program must exit, 0 otherwise.
 */
int parse_cmd_parameters(
        int argc, 
        char *argv[],
        char ** pi8OutputFileName,
        uint32_t * u32TransmitWindowLength_us,
        uint32_t * u32DeadTime_us,
        uint32_t * u32TransmitWindowsPerClient,
        uint32_t * u32TotalClients,
        uint8_t * u8NoTerminal,
        uint8_t * u8Combine);

/** \brief Function that calculates metrics based on the received packets and writes it to file on a per packet basis
 *  
 *  \param[in] sStopTime 
 *  \param[in] sStartTime                       Time the very first packet out of all windows and clients was received.
 *  \param[in] psReceiveBuffer                  Pointer to array containing received packets.
 *  \param[in] psRxTimes                        Pointer to array containing the times the packets in 
 *                                              \ref psReceiveBuffer were received.
 *  \param[in] pi8OutputFileName                Name of the file to write to.
 *  \param[in] u32TransmitWindowLength_us       Window length in microseconds.
 *  \param[in] u32DeadTime_us                   Deadtime between windows in microseconds.
 *  \param[in] u32TransmitWindowsPerClient      The number of windows that each client must transmit.
 *  \param[in] u32TotalClients                  The number of transmitter clients that connect to this receiver server.
 *  \param[in] u8NoTerminal                     If 1 indicates that the program must disable printing window information
 *                                              to the terminal.
 */
void calculate_packet_metrics(
        struct timeval sStopTime, 
        struct timeval sStartTime, 
        struct UdpTestingPacketHeader * psReceivedPacketHeaders, 
        struct timeval * psRxTimes ,
        int64_t i64ReceivedPacketsCount, 
        int64_t i64TotalSentPackets,
        char * pi8OutputFileName,
        uint8_t u8NoTerminal);

void calculate_window_metrics_packet_received(
        struct timeval * sReceivedPacketReceivedTime,
        struct UdpTestingPacket * sReceivedPacket,
        struct UdpTestingPacket * sPreviousReceivedPacket,
        struct WindowInformation * psWindowInformation,
        uint32_t u32TransmitWindowsPerClient,
        uint32_t u32TotalClients);

void calculate_window_metrics_all_packets_received(
        struct WindowInformation * psWindowInformation,
        uint32_t u32TransmitWindowsPerClient,
        uint32_t u32TotalClients,
        uint8_t u8NoTerminal,
        char * pi8OutputFileName);

// Driver code 
int main(int argc, char *argv[]) 
{ 
    printf("Funnel In Test Server.\n\n");
    
    //1. ***** Parse command line arguments and set up server *****
    uint32_t u32TransmitWindowLength_us = TRANSMIT_WINDOW_US_DEFAULT;
    uint32_t u32DeadTime_us = DEAD_TIME_US_DEFAULT;
    uint32_t u32TransmitWindowsPerClient = TOTAL_WINDOWS_PER_CLIENT_DEFAULT;
    uint32_t u32TotalClients = TOTAL_CLIENTS_DEFAULT;
    uint8_t  u8NoTerminal = DONT_PRINT_TO_TERMINAL_DEFAULT; 
    uint8_t u8Combine = COMBINE_DEFAULT;
    char * pu8OutputFileName = FILE_NAME_DEFAULT;

    int iRet = parse_cmd_parameters(argc, argv, &pu8OutputFileName, &u32TransmitWindowLength_us, 
            &u32DeadTime_us, &u32TransmitWindowsPerClient, &u32TotalClients, &u8NoTerminal, &u8Combine);
    if(iRet != 0){
        return 0;
    }

    double dExpectedRuntime_s = (int64_t)(u32DeadTime_us + u32TransmitWindowLength_us) 
            * (int64_t)u32TransmitWindowsPerClient * (int64_t)u32TotalClients;
    dExpectedRuntime_s = dExpectedRuntime_s/1000000;
    int32_t i32Hours = dExpectedRuntime_s/3600;
    printf("%f %d\n",dExpectedRuntime_s,i32Hours);
    int32_t i32Minutes = (dExpectedRuntime_s - i32Hours*3600)/60;
    double dSeconds = dExpectedRuntime_s - i32Hours*3600 - i32Minutes*60;
    printf("This program will take %d Hours, %d Minutes, %f Seconds to run\n",i32Hours,i32Minutes,dSeconds);
    
    int iSocketFileDescriptor; 
    struct sockaddr_in sServAddr, sCliAddr; 

    //Allocate buffers if data to store 
    size_t ulRingbufferBytes = NUMBER_RINGBUFFER_PACKETS*sizeof(struct UdpTestingPacket);
    struct UdpTestingPacket * psReceiveBuffer = malloc(ulRingbufferBytes);
    
    struct timeval * psRxTimes;
    struct UdpTestingPacketHeader * psReceivedPacketHeaders;
    
    struct WindowInformation * psWindowInformation;
    if(u8Combine != 0)
    {   
        psRxTimes = malloc(NUMBER_RINGBUFFER_PACKETS*sizeof(struct timeval));
        psWindowInformation = malloc(u32TransmitWindowsPerClient * u32TotalClients * sizeof(struct WindowInformation));
        memset(psWindowInformation,0,u32TransmitWindowsPerClient * u32TotalClients * sizeof(struct WindowInformation));
    }
    else
    {
        psRxTimes = malloc(MAXIMUM_NUMBER_OF_PACKETS*sizeof(struct timeval));
        psReceivedPacketHeaders = malloc(MAXIMUM_NUMBER_OF_PACKETS*sizeof(struct UdpTestingPacketHeader));
    }
    
    
    //2. ***** Creating socket file descriptor *****
    if ( (iSocketFileDescriptor = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0 ) 
    { 
        perror("socket creation failed"); 
        exit(EXIT_FAILURE); 
    } 
    
    memset(&sServAddr, 0, sizeof(sServAddr)); 
    memset(&sCliAddr, 0, sizeof(sCliAddr)); 
    
    // Filling server information 
    sServAddr.sin_family    = AF_INET; // IPv4 
    sServAddr.sin_addr.s_addr = INADDR_ANY; 
    sServAddr.sin_port = htons(UDP_TEST_PORT); 
    
    // Bind the socket with the server address 
    if ( bind(iSocketFileDescriptor, (const struct sockaddr *)&sServAddr,  
            sizeof(sServAddr)) < 0 ) 
    { 
        perror("bind failed"); 
        exit(EXIT_FAILURE); 
    } 
    
    int iSockAddressLength, iReceivedBytes; 
    iSockAddressLength = sizeof(sCliAddr);
  
    //3. ***** Waiting for initial hello messages from clients *****
    struct sockaddr_in * psCliAddrInit = malloc(u32TotalClients*sizeof(struct sockaddr_in));
    memset(psCliAddrInit, 0, sizeof(struct sockaddr_in)*u32TotalClients);
    for (size_t i = 0; i < u32TotalClients; i++)
    {
        
        printf("Waiting For Hello Message From Client %ld of %d\n",i+1,u32TotalClients);
        struct MetadataPacketClient sHelloPacket = {CLIENT_MESSAGE_EMPTY,0};
        uint8_t u8Duplicate = 1;

        while(sHelloPacket.u32MetadataPacketCode != CLIENT_MESSAGE_HELLO || u8Duplicate != 0)
        {
            iReceivedBytes = recvfrom(iSocketFileDescriptor, (struct MetadataPacketClient *)&sHelloPacket, 
                        sizeof(struct MetadataPacketClient),  
                        MSG_WAITALL, ( struct sockaddr *) &psCliAddrInit[i], 
                        &iSockAddressLength); 
            printf("Message Received\n");
            //Check that the message has not been received from a server that already exists
            u8Duplicate = 0;
            for (size_t j = 0; j < i; j++)
            {
                if(psCliAddrInit[i].sin_addr.s_addr == psCliAddrInit[j].sin_addr.s_addr){
                    printf("Hello message Already received from client with this address\n");
                    u8Duplicate = 1;
                }
                //printf("%d %d\n",psCliAddrInit[i].sin_addr.s_addr,psCliAddrInit[j].sin_addr.s_addr);
            }
        }
        printf("Hello Message Received from client %ld\n",i+1);
    }
        
    //4. ***** Determine and send Configuration Information to clients *****
    printf("Sending Configuration Message to client\n");
    struct timeval sCurrentTime;
    gettimeofday(&sCurrentTime,NULL);

    for (size_t i = 0; i < u32TotalClients; i++)
    {
        struct MetadataPacketMaster sConfigurationPacket;
        sConfigurationPacket.u32MetadataPacketCode = SERVER_MESSAGE_CONFIGURATION;
        sConfigurationPacket.sSpecifiedTransmitStartTime.tv_sec = sCurrentTime.tv_sec + 3;
        sConfigurationPacket.sSpecifiedTransmitStartTime.tv_usec = i * (u32TransmitWindowLength_us + 
                u32DeadTime_us);
        sConfigurationPacket.sSpecifiedTransmitTimeLength.tv_sec = 0;
        sConfigurationPacket.sSpecifiedTransmitTimeLength.tv_usec = u32TransmitWindowLength_us;
        sConfigurationPacket.i32DeadTime_us = u32DeadTime_us;
        sConfigurationPacket.uNumberOfRepeats = u32TransmitWindowsPerClient;
        sConfigurationPacket.uNumClients = u32TotalClients;
        sConfigurationPacket.fWaitAfterStreamTransmitted_s = 1;
        sConfigurationPacket.i32ClientIndex = i;

        sendto(iSocketFileDescriptor, (const struct MetadataPacketMaster *)&sConfigurationPacket, \
            sizeof(struct MetadataPacketMaster),  
            MSG_CONFIRM, (const struct sockaddr *) &psCliAddrInit[i], 
                iSockAddressLength); 
    }
    
    

    //5. ***** Receive data stream messages from client *****
    uint8_t * pu8TrailingPacketReceived = (uint8_t*) malloc(u32TotalClients*sizeof(uint8_t));
    memset(pu8TrailingPacketReceived,0,u32TotalClients*sizeof(uint8_t));
    int * piTotalSentPacketsPerClient = (int *) malloc(u32TotalClients*sizeof(int));
    memset(piTotalSentPacketsPerClient,0,u32TotalClients*sizeof(int));
    

    printf("Starting Streaming\n");
    size_t ulReceivedPacketIndex;
    int64_t i64ReceivedPacketsCount = 0;
    struct timeval sStopTime, sStartTime;
    gettimeofday(&sStartTime, NULL);
    while(1)//Keep waiting for data until trailing packets have been received
    {  

        //5.1 ***** *****
        ulReceivedPacketIndex = i64ReceivedPacketsCount % NUMBER_RINGBUFFER_PACKETS;
        iReceivedBytes = recvfrom(iSocketFileDescriptor, (char *)&psReceiveBuffer[ulReceivedPacketIndex], 
                sizeof(struct UdpTestingPacket), MSG_WAITALL, ( struct sockaddr *) &sCliAddr, 
                &iSockAddressLength); 
        if(iReceivedBytes != sizeof(struct UdpTestingPacket))
        {
            printf("****** More than a single packet was received: %d *****",iReceivedBytes);
            return 1;
        }

        //Write the received time to the last received packet.
        if (u8Combine != 0)
        {
            gettimeofday(&psRxTimes[ulReceivedPacketIndex], NULL);
        }
        else
        {
            gettimeofday(&psRxTimes[i64ReceivedPacketsCount], NULL);
        }
        //5.2 ***** *****
        //This if-statement confirms end condition has been received - I think a better way to do this in the \
        future would be to just wait until a set time has passed and then send out-of-band messages to the clients \
        asking for meta data. This is not worth changing for now.
        if(psReceiveBuffer[ulReceivedPacketIndex].sHeader.i32TrailingPacket != 0)
        {
            int iClientIndex = psReceiveBuffer[ulReceivedPacketIndex].sHeader.i32ClientIndex;
            pu8TrailingPacketReceived[iClientIndex] = 1;
            piTotalSentPacketsPerClient[iClientIndex] = 
                    psReceiveBuffer[ulReceivedPacketIndex].sHeader.i64PacketsSent;
            printf("Trailing packet received indicating client %d has finished transmitting.\n",
                    psReceiveBuffer[ulReceivedPacketIndex].sHeader.i32ClientIndex);
            
            uint8_t u8End = 1;
            for (size_t i = 0; i < u32TotalClients; i++)
            {
                if(pu8TrailingPacketReceived[i] == 0){
                    u8End = 0;
                    break;
                }
            }
            
            if(u8End == 1){
                break;
            }
            continue;
        }

        //If this is the very first packet received then reset the start time
        if(i64ReceivedPacketsCount == 0)
        {
            gettimeofday(&sStartTime, NULL);
        }

        //5.3 ***** *****
        //This code is only called if per window information is being gathered
        size_t ulReceivedPacketIndexPrevious;
        if(ulReceivedPacketIndex != 0){
            ulReceivedPacketIndexPrevious = ulReceivedPacketIndex - 1;
        }else{
            ulReceivedPacketIndexPrevious = NUMBER_RINGBUFFER_PACKETS-1;
        }

        if(u8Combine != 0){
        calculate_window_metrics_packet_received(
                &psRxTimes[ulReceivedPacketIndex],
                &psReceiveBuffer[ulReceivedPacketIndex],
                &psReceiveBuffer[ulReceivedPacketIndexPrevious],
                psWindowInformation, u32TransmitWindowsPerClient, u32TotalClients);
        }else{
            //This memcopy only occurs if per packet information is being gathered
            memcpy(&psReceivedPacketHeaders[i64ReceivedPacketsCount],&psReceiveBuffer[ulReceivedPacketIndex].sHeader,
                    sizeof(struct UdpTestingPacketHeader));
        }

        //5.4 ***** *****
        i64ReceivedPacketsCount++;//Not counted in the case of a trailing packet
    }

    printf("All Messages Received\n");

    //Calculate the total number of packets received by adding the number of packets transmitted by each client \
    together.
    int64_t i64TotalSentPackets = 0;
    for (size_t i = 0; i < u32TotalClients; i++)
    {
        i64TotalSentPackets += piTotalSentPacketsPerClient[i];
    }

    printf("Total Packets Sent: %ld, Total Packets Received: %ld\n", i64TotalSentPackets, i64ReceivedPacketsCount);
    
    if(u8Combine == 0){
        sStopTime = psRxTimes[i64ReceivedPacketsCount-1]; //Set stop time equal to last received packet - not simply \
        getting system time here as trailing packets can take quite a while to arrive.
    }

    //6. ***** Analyse data, and calculate and display performance metrics *****
    if(u8Combine != 0)
    {
        calculate_window_metrics_all_packets_received(psWindowInformation, u32TransmitWindowsPerClient, u32TotalClients,
                u8NoTerminal, pu8OutputFileName);
    }
    else
    {
        calculate_packet_metrics(sStopTime,sStartTime,psReceivedPacketHeaders,
                psRxTimes,i64ReceivedPacketsCount,i64TotalSentPackets,pu8OutputFileName,u8NoTerminal);
    }

    //Cleanup
    free(pu8TrailingPacketReceived);
    free(piTotalSentPacketsPerClient);
    free(psCliAddrInit);
    free(psReceiveBuffer);
    free(psRxTimes);
    if(u8Combine != 0){
        free(psWindowInformation);
    }
    else
    {
        free(psReceivedPacketHeaders);
    }
    close(iSocketFileDescriptor);

    return 0;
} 

void calculate_packet_metrics(
        struct timeval sStopTime, 
        struct timeval sStartTime, 
        struct UdpTestingPacketHeader * psReceivedPacketHeaders, 
        struct timeval * psRxTimes, 
        int64_t i64ReceivedPacketsCount, 
        int64_t i64TotalSentPackets,
        char * pi8OutputFileName,
        uint8_t u8NoTerminal)
{
    FILE *pCsvFile;
    FILE *pTextFile;

    char * pi8OutputFileNameCsv = (char *) malloc(1 + strlen(pi8OutputFileName)+ strlen(".csv") );
    char * pi8OutputFileNameTxt = (char *) malloc(1 + strlen(pi8OutputFileName)+ strlen(".txt") );
    strcat(pi8OutputFileNameCsv,pi8OutputFileName);
    strcat(pi8OutputFileNameCsv,".csv");
    strcat(pi8OutputFileNameTxt,pi8OutputFileName);
    strcat(pi8OutputFileNameTxt,".txt");
    pCsvFile = fopen(pi8OutputFileNameCsv,"w");
    pTextFile = fopen(pi8OutputFileNameTxt,"w");
   
    float fTimeTaken_s = (sStopTime.tv_sec - sStartTime.tv_sec) + 
            ((float)(sStopTime.tv_usec - sStartTime.tv_usec))/1000000;
    double fDataRate_Gibps = ((i64ReceivedPacketsCount)*sizeof(struct UdpTestingPacket))
            *8.0/fTimeTaken_s/1024.0/1024.0/1024.0;

    double dRxTime_prev = (double)psRxTimes[0].tv_sec + ((double)psRxTimes[0].tv_usec)/1000000.0;
    double dTxTime_prev = (double)psReceivedPacketHeaders[0].sTransmitTime.tv_sec + 
            ((double)psReceivedPacketHeaders[0].sTransmitTime.tv_usec)/1000000.0;

    double dMinTxRxDiff_s=1,dMinTxTxDiff_s=1,dMinRxRxDiff_s=1;
    double dMaxTxRxDiff_s=-1,dMaxTxTxDiff_s=-1,dMaxRxRxDiff_s=-1;
    double dAvgTxRxDiff_s=0,dAvgTxTxDiff_s=0,dAvgRxRxDiff_s=0;

    int iWindowBoundaries=0;
    uint8_t u8OutOfOrder = 0;
    for (size_t i = 0; i < i64ReceivedPacketsCount; i++)
    {
        if(i != 0 && psReceivedPacketHeaders[i-1].i64PacketIndex > psReceivedPacketHeaders[i].i64PacketIndex 
                && psReceivedPacketHeaders[i-1].i32ClientIndex == psReceivedPacketHeaders[i].i32ClientIndex)
        {
            printf("Data received out of order\n");
            fprintf(pTextFile,"Data received out of order\n");
            u8OutOfOrder = 1;
        }

        double dTxTime = (double)psReceivedPacketHeaders[i].sTransmitTime.tv_sec 
                + ((double)psReceivedPacketHeaders[i].sTransmitTime.tv_usec)/1000000.0;
        double dRxTime = (double)psRxTimes[i].tv_sec + ((double)psRxTimes[i].tv_usec)/1000000.0;

        double dDiffRxTx = dRxTime-dTxTime;
        dAvgTxRxDiff_s+=dDiffRxTx;
        if(dDiffRxTx < dMinTxRxDiff_s && dDiffRxTx != 0)
        {
            dMinTxRxDiff_s = dDiffRxTx;
        }
        if(dDiffRxTx > dMaxTxRxDiff_s)
        {
            dMaxTxRxDiff_s = dDiffRxTx;
        }

        double dDiffRxRx = dRxTime-dRxTime_prev;
        dAvgRxRxDiff_s+=dDiffRxRx;
        if(dDiffRxRx < dMinRxRxDiff_s && dDiffRxRx != 0)
        {
            dMinRxRxDiff_s = dDiffRxRx;
        }
        if(dDiffRxRx > dMaxRxRxDiff_s)
        {
            dMaxRxRxDiff_s = dDiffRxRx;
        }

        double dDiffTxTx = dTxTime-dTxTime_prev;
        dAvgTxTxDiff_s+=dDiffTxTx;
        if(dDiffTxTx < dMinTxTxDiff_s && dDiffTxTx != 0)
        {
            dMinTxTxDiff_s = dDiffTxTx;
        }
        if(dDiffTxTx > dMaxTxTxDiff_s)
        {
            iWindowBoundaries++;
            dMaxTxTxDiff_s = dDiffTxTx;
        }

        //ONly print this if enabled - for long tests, significant time can be wasted here.
        if(u8NoTerminal == 0){
            printf("Packet %ld Client %d Window %ld Client Packet ID %ld TX %fs, RX %fs, Diff RX/TX %fs, Diff TX/TX %fs, " 
                    "Diff RX/RX %fs\n",
                    i, psReceivedPacketHeaders[i].i32ClientIndex, psReceivedPacketHeaders[i].i64TransmitWindowIndex, 
                    psReceivedPacketHeaders[i].i64PacketIndex, dTxTime, dRxTime, dDiffRxTx, dDiffTxTx, dDiffRxRx);
        }
        //Do not write to plain text file if more than 1 GB of packet headers is receieved - it takes too long
        if(i64ReceivedPacketsCount * sizeof(struct UdpTestingPacketHeader) < 1000000000){
            fprintf(pTextFile,"Packet %ld Client %d Window %ld Client Packet ID %ld TX %fs, RX %fs, Diff RX/TX %fs, "
                    "Diff TX/TX %fs, Diff RX/RX %fs\n",
                    i, psReceivedPacketHeaders[i].i32ClientIndex, psReceivedPacketHeaders[i].i64TransmitWindowIndex, 
                    psReceivedPacketHeaders[i].i64PacketIndex, dTxTime, dRxTime, dDiffRxTx, dDiffTxTx, dDiffRxRx);
        }
        fprintf(pCsvFile,"%ld,%d,%ld,%ld,%f,%f\n",
                i, psReceivedPacketHeaders[i].i32ClientIndex, psReceivedPacketHeaders[i].i64TransmitWindowIndex,
                psReceivedPacketHeaders[i].i64PacketIndex, dTxTime, dRxTime);

        dRxTime_prev = dRxTime;
        dTxTime_prev = dTxTime;
    }
    if(i64ReceivedPacketsCount * sizeof(struct UdpTestingPacketHeader) < 100000000){
        fprintf(pTextFile,"Raw packet Values not written to this file to preserve disk space.");
    }

    dAvgTxRxDiff_s = dAvgTxRxDiff_s/(i64ReceivedPacketsCount-1);
    dAvgTxTxDiff_s = dAvgTxTxDiff_s/(i64ReceivedPacketsCount-1);
    dAvgRxRxDiff_s = dAvgRxRxDiff_s/(i64ReceivedPacketsCount-1);

    printf("\n Average Time Between Packets\n");
    fprintf(pTextFile,"\n Average Time Between Packets\n");
    printf("     |  Avg(s) |  Min(s) |  Max(s) |\n");
    fprintf(pTextFile,"     |  Avg(s) |  Min(s) |  Max(s) |\n");
    printf("TX/RX|%9.6f|%9.6f|%9.6f|\n",dAvgTxRxDiff_s,dMinTxRxDiff_s,dMaxTxRxDiff_s);
    fprintf(pTextFile,"TX/RX|%9.6f|%9.6f|%9.6f|\n",dAvgTxRxDiff_s,dMinTxRxDiff_s,dMaxTxRxDiff_s);
    printf("TX/TX|%9.6f|%9.6f|%9.6f|\n",dAvgTxTxDiff_s,dMinTxTxDiff_s,dMaxTxTxDiff_s);
    fprintf(pTextFile,"TX/TX|%9.6f|%9.6f|%9.6f|\n",dAvgTxTxDiff_s,dMinTxTxDiff_s,dMaxTxTxDiff_s);
    printf("RX/RX|%9.6f|%9.6f|%9.6f|\n",dAvgRxRxDiff_s,dMinRxRxDiff_s,dMaxRxRxDiff_s);
    fprintf(pTextFile,"RX/RX|%9.6f|%9.6f|%9.6f|\n",dAvgRxRxDiff_s,dMinRxRxDiff_s,dMaxRxRxDiff_s);
    printf("\n");
    fprintf(pTextFile,"\n");
    printf("It took %f seconds to receive %ld bytes of data (%ld packets)\n", 
           fTimeTaken_s,(i64ReceivedPacketsCount-1)*PACKET_SIZE_BYTES,i64ReceivedPacketsCount-1);
    fprintf(pTextFile,"It took %f seconds to receive %ld bytes of data (%ld packets)\n", 
           fTimeTaken_s,(i64ReceivedPacketsCount-1)*PACKET_SIZE_BYTES,i64ReceivedPacketsCount-1); 
    printf("\n");
    fprintf(pTextFile,"\n");

    if(u8OutOfOrder != 0)
    {
        printf("\n");
        printf("*********Data Received out of order - investigate this. ********\n");
        printf("\n");
        fprintf(pTextFile,"\n*********Data Received out of order - investigate this. ********\n\n");
    }
    else
    {
        printf("\n");
        printf("Data Received in order");
        printf("\n");
        fprintf(pTextFile,"\nData Received in order\n");
    }

    printf("\n");
    printf("%ld of %ld packets received. Drop rate = %.2f %%\n",
            i64ReceivedPacketsCount,i64TotalSentPackets,
            (1-((double)i64ReceivedPacketsCount)/((double)i64TotalSentPackets))*100);
    printf("\n");
    fprintf(pTextFile,"\n%ld of %ld packets received. Drop rate = %.2f %%\n\n",
            i64ReceivedPacketsCount,i64TotalSentPackets,
            (1-((double)i64ReceivedPacketsCount)/((double)i64TotalSentPackets))*100);

    double fDataRateAvg2_Gibps = ((double)sizeof(struct UdpTestingPacket))/dAvgTxTxDiff_s/1024.0/1024.0/1024.0*8;//*8 is \
    for bit to byte conversion
    printf("Data Rate According to Average Packet Tx Time Difference: %f Gibps\n",fDataRateAvg2_Gibps);
    fprintf(pTextFile,"Data Rate According to Average Packet Tx Time Difference: %f Gibps\n",fDataRateAvg2_Gibps);

    printf("\n");

    fclose(pCsvFile);
    fclose(pTextFile);
}

void calculate_window_metrics_packet_received(
        struct timeval * sReceivedPacketReceivedTime,
        struct UdpTestingPacket * sReceivedPacket,
        struct UdpTestingPacket * sPreviousReceivedPacket,
        struct WindowInformation * psWindowInformation,
        uint32_t u32TransmitWindowsPerClient,
        uint32_t u32TotalClients)
{
    struct UdpTestingPacketHeader * sReceivedPacketHeader = &sReceivedPacket->sHeader;
    int64_t i64CurrentWindowIndex = sReceivedPacketHeader->i64TransmitWindowIndex * u32TotalClients
            + sReceivedPacketHeader->i32ClientIndex;
    struct WindowInformation * sCurrentWindow = &psWindowInformation[i64CurrentWindowIndex];
    
    //Calculate Transit time
    double dTransmitTime = sReceivedPacketHeader->sTransmitTime.tv_sec 
                + sReceivedPacketHeader->sTransmitTime.tv_usec/1000000.0;
    double dReceiveTime = sReceivedPacketReceivedTime->tv_sec 
                + sReceivedPacketReceivedTime->tv_usec/1000000.0;
    double dTransferTime = dReceiveTime - dTransmitTime;

    //printf("Current Window Index %ld, %d\n",i64CurrentWindowIndex,u32TotalClients);
    //If this is the first packet in the window, a few fields need to be updated once off
    if(sCurrentWindow->i64PacketsReceived == 0)
    {
        sCurrentWindow->sFirstRxTime = *sReceivedPacketReceivedTime;
        sCurrentWindow->sFirstTxTime = sReceivedPacketHeader->sTransmitTime;
        sCurrentWindow->i64FirstPacketIndex = sReceivedPacketHeader->i64PacketIndex;
        sCurrentWindow->i64TransmitWindowIndex = sReceivedPacketHeader->i64TransmitWindowIndex;
        sCurrentWindow->i32ClientIndex = sReceivedPacketHeader->i32ClientIndex;
        sCurrentWindow->dMaxTxRxDiff_s = dTransferTime;
        sCurrentWindow->dMinTxRxDiff_s = dTransferTime;
        sCurrentWindow->dAvgTxRxDiff_s = 0;
    }
    ////This if statements only occurs when not at an expected window change boundary
    else
    {
        //Check that the previous packet is not from an unexpected different window - if it is, this means that an \
        unwanted packet overlap has occured
        struct UdpTestingPacketHeader * sPreviousPacketReceivedPacketHeader = &sPreviousReceivedPacket->sHeader;
        int64_t i64PreviousPacketWindowIndex = sPreviousPacketReceivedPacketHeader->i64TransmitWindowIndex 
            * u32TotalClients + sPreviousPacketReceivedPacketHeader->i32ClientIndex;
        //printf("Packet Overlap: %ld %ld\n",i64PreviousPacketWindowIndex,i64CurrentWindowIndex);
        if(i64PreviousPacketWindowIndex < i64CurrentWindowIndex && i64CurrentWindowIndex != 0)
        {
            //printf("Packet Overlap: %ld %ld\n",i64PreviousPacketWindowIndex,i64CurrentWindowIndex);
            struct WindowInformation * sPreviousWindow = &psWindowInformation[i64PreviousPacketWindowIndex];
            sPreviousWindow->i64OverlappingWindowsBack++;
            sCurrentWindow->i64OverlappingWindowsFront++;
        }

        //If current and previous packet are in the same window, check that no packets have been dropped or received \
        out of order
        if(i64PreviousPacketWindowIndex == i64CurrentWindowIndex)
        {   
            //Packet is missing and assumed dropped
            int iPacketDifference = sReceivedPacketHeader->i64PacketIndex - sCurrentWindow->i64LastPacketIndex;
            if(iPacketDifference != 1)
            {
                //printf("Packet has been dropped\n");
                sCurrentWindow->i64MissingIndexes += iPacketDifference;  
            }

            //Packet is received out of order - would be VERY suprised to see this in the current test set up
            if(sReceivedPacketHeader->i64PacketIndex < sCurrentWindow->i64LastPacketIndex)
            {
                printf("Packet Received out of order\n");
                sCurrentWindow->i64OutOfOrderIndexes++;  
            }
        }
    }

    //Check if this is a new maximum/minimum transit time
    if(sCurrentWindow->dMaxTxRxDiff_s < dTransferTime){
        sCurrentWindow->dMaxTxRxDiff_s = dTransferTime;
    }
    if(sCurrentWindow->dMinTxRxDiff_s > dTransferTime)
    {
        sCurrentWindow->dMinTxRxDiff_s = dTransferTime;
    }

    sCurrentWindow->dAvgTxRxDiff_s += dTransferTime;
    sCurrentWindow->sLastRxTime = *sReceivedPacketReceivedTime;
    sCurrentWindow->sLastTxTime = sReceivedPacketHeader->sTransmitTime;
    sCurrentWindow->i64LastPacketIndex = sReceivedPacketHeader->i64PacketIndex;
    sCurrentWindow->i64PacketsReceived++;

}

void calculate_window_metrics_all_packets_received(
        struct WindowInformation * psWindowInformation,
        uint32_t u32TransmitWindowsPerClient,
        uint32_t u32TotalClients,
        uint8_t u8NoTerminal,
        char * pi8OutputFileName)
{
    FILE *pCsvFile;
    FILE *pTextFile;

    char * pi8OutputFileNameCsv = (char *) malloc(1 + strlen(pi8OutputFileName)+ strlen(".csv") );
    char * pi8OutputFileNameTxt = (char *) malloc(1 + strlen(pi8OutputFileName)+ strlen(".txt") );
    strcat(pi8OutputFileNameCsv,pi8OutputFileName);
    strcat(pi8OutputFileNameCsv,".csv");
    strcat(pi8OutputFileNameTxt,pi8OutputFileName);
    strcat(pi8OutputFileNameTxt,".txt");
    pCsvFile = fopen(pi8OutputFileNameCsv,"w");
    pTextFile = fopen(pi8OutputFileNameTxt,"w");

    for (size_t i = 0; i < u32TotalClients*u32TransmitWindowsPerClient; i++)
    {
        //Calculate Runtime
        double dStartTimeRx_s = psWindowInformation[i].sFirstRxTime.tv_sec 
                + psWindowInformation[i].sFirstRxTime.tv_usec/1000000.0;
        double dEndTimeRx_s = psWindowInformation[i].sLastRxTime.tv_sec 
                + psWindowInformation[i].sLastRxTime.tv_usec/1000000.0;
        double dWindowRunTimeRx_s = dEndTimeRx_s - dStartTimeRx_s;

        double dStartTimeTx_s = psWindowInformation[i].sFirstTxTime.tv_sec 
                + psWindowInformation[i].sFirstTxTime.tv_usec/1000000.0;
        double dEndTimeTx_s = psWindowInformation[i].sLastTxTime.tv_sec 
                + psWindowInformation[i].sLastTxTime.tv_usec/1000000.0;
        double dWindowRunTimeTx_s = dEndTimeTx_s - dStartTimeTx_s;

        //Calculate Data rate using the runtime values that have just been calculated
        double dDataTransfered_bytes = psWindowInformation[i].i64PacketsReceived * sizeof(struct UdpTestingPacket);

        double dDataRate_Gibps = dDataTransfered_bytes*8/dWindowRunTimeTx_s/1024.0/1024.0/1024.0;

        //Do division to go from total to average differences
        psWindowInformation[i].dAvgTxRxDiff_s = 
                    psWindowInformation[i].dAvgTxRxDiff_s/((double)psWindowInformation[i].i64PacketsReceived);

        if(u8NoTerminal == 0){
            printf("i: %ld, Client: %d,  Window: %ld, Packets: %ld, Missing: %ld, Rx_Runtime: %f, Tx_Runtime: %f, " 
                    "DataRate(Gibs): %f Avg Tx/Rx: %f, Min Tx/Rx: %f, Max Tx/Rx %f, Overlap Front %ld, "
                    "Overlap Back %ld, Start Tx %f, End Tx %f, Start Rx %f, End Rx %f\n",
                    i,psWindowInformation[i].i32ClientIndex,
                    psWindowInformation[i].i64TransmitWindowIndex,
                    psWindowInformation[i].i64PacketsReceived,
                    psWindowInformation[i].i64MissingIndexes,
                    dWindowRunTimeRx_s,
                    dWindowRunTimeTx_s,
                    dDataRate_Gibps,
                    psWindowInformation[i].dAvgTxRxDiff_s,
                    psWindowInformation[i].dMinTxRxDiff_s,
                    psWindowInformation[i].dMaxTxRxDiff_s,
                    psWindowInformation[i].i64OverlappingWindowsFront,
                    psWindowInformation[i].i64OverlappingWindowsBack,
                    dStartTimeTx_s, dEndTimeTx_s, dStartTimeRx_s, dEndTimeRx_s);
        }
        fprintf(pTextFile,"i: %ld, Client: %d,  Window: %ld, Packets: %ld, Missing: %ld, Rx_Runtime: %f, "
                "Tx_Runtime: %f, DataRate(Gibs): %f Avg Tx/Rx: %f, Min Tx/Rx: %f, Max Tx/Rx %f, Overlap Front %ld, "
                "Overlap Back %ld, Start Tx %f, End Tx %f, Start Rx %f, End Rx %f\n",
                i,psWindowInformation[i].i32ClientIndex,
                psWindowInformation[i].i64TransmitWindowIndex,
                psWindowInformation[i].i64PacketsReceived,
                psWindowInformation[i].i64MissingIndexes,
                dWindowRunTimeRx_s,
                dWindowRunTimeTx_s,
                dDataRate_Gibps,
                psWindowInformation[i].dAvgTxRxDiff_s,
                psWindowInformation[i].dMinTxRxDiff_s,
                psWindowInformation[i].dMaxTxRxDiff_s,
                psWindowInformation[i].i64OverlappingWindowsFront,
                psWindowInformation[i].i64OverlappingWindowsBack,
                dStartTimeTx_s, dEndTimeTx_s, dStartTimeRx_s, dEndTimeRx_s);
        fprintf(pCsvFile,"%ld, %d, %ld, %ld, %ld, %f, %f, %f, %f, %f, %f, %ld, %ld, %f, %f, %f, %f\n",
                i,psWindowInformation[i].i32ClientIndex,
                psWindowInformation[i].i64TransmitWindowIndex,
                psWindowInformation[i].i64PacketsReceived,
                psWindowInformation[i].i64MissingIndexes,
                dWindowRunTimeRx_s,
                dWindowRunTimeTx_s,
                dDataRate_Gibps,
                psWindowInformation[i].dAvgTxRxDiff_s,
                psWindowInformation[i].dMinTxRxDiff_s,
                psWindowInformation[i].dMaxTxRxDiff_s,
                psWindowInformation[i].i64OverlappingWindowsFront,
                psWindowInformation[i].i64OverlappingWindowsBack,
                dStartTimeTx_s, dEndTimeTx_s, dStartTimeRx_s, dEndTimeRx_s);
    }
    
    fclose(pCsvFile);
    fclose(pTextFile);
}


int parse_cmd_parameters(
        int argc, 
        char * argv[], 
        char ** pi8OutputFileName,
        uint32_t * u32TransmitWindowLength_us,
        uint32_t * u32DeadTime_us,
        uint32_t * u32TransmitWindowsPerClient,
        uint32_t * u32TotalClients,
        uint8_t * u8NoTerminal,
        uint8_t * u8Combine)
{
    int opt; 
      
    // put ':' in the starting of the 
    // string so that program can  
    //distinguish between '?' and ':'  
    while((opt = getopt(argc, argv, ":t:d:hpcn:o:w:")) != -1)  
    {  
        switch(opt)  
        {  
            case 'h':  
                printf(
                    "Program for testing network performance when scheduling packet sending across multiple "
                    "transmitters at specific time intervals.\n\n");
                    
                printf(
                    "This program creates the main receiving server. The udp_send program will create a "
                    "transmitter client that will connect to this server. All configuration information from this "
                    "program will be sent to the client.\n\n");

                printf(
                    "There will be NUM_TRANSMITTERS clients. Each tranmsitter will  transmit data for a specific "
                    "WINDOW_LENGTH in microseconds. Each client will transmit while the others wait. There will be "
                    "NUM_WINDOWS interleaved windows for each client. Between each transfer window there will be "
                    "DEAD_TIME microseconds where no client sends data.\n\n");

                printf(
                    "This program will write all packet header data to a plain text and csv file. For longer runs, "
                    "this file can get impractically large. The -c command line option tells the program collect "
                    "statistics on a per window basis and only write these values to file. This greatly reduces the "
                    "size of the output file.\n\n");

                printf("Options:\n");
                printf("    -c                    Combine outputs by window.\n");
                printf("    -d DEAD_TIME          The amount of dead time between windows.\n");
                printf("    -h                    Print this message and exit.\n");
                printf("    -n NUM_WINDOWS        The number of windows each client will transmit.\n");
                printf("    -o FILE               Write results to FILE.\n");
                printf("    -p                    Disable Print to terminal\n");
                printf("    -t NUM_TRANSMITTERS   The number of clients that will transmit to the host.\n");
                printf("    -w WINDOW_LENGTH      The length of a window in microseconds.\n");
                
                return 1;
            case 'c': 
                *u8Combine = 1;
                printf("Per window statistics gathering enabled.\n"); 
                break;
            case 'd': 
                *u32DeadTime_us = atoi(optarg);
                printf("Deadtime set to %d us.\n",*u32DeadTime_us); 
                break;
            case 'n':  
                *u32TransmitWindowsPerClient = atoi(optarg);
                printf("Each client will transmit %d windows.\n",*u32TransmitWindowsPerClient);    
                break;  
            case 'o':  
                *pi8OutputFileName = optarg;
                printf("Output File name set to %s\n",*pi8OutputFileName); 
                break;  
            case 'p': 
                *u8NoTerminal = 1;
                printf("Full terminal output disabled.\n"); 
                break;
            case 't':
                *u32TotalClients = atoi(optarg);
                printf("Number of transmitters set to %d\n",*u32TotalClients);  
                break;  
            case 'w':
                *u32TransmitWindowLength_us = atoi(optarg);
                printf("Transmit window length set to %d us.\n",*u32TransmitWindowLength_us);  
                break;  
            case '?':
                printf("Unknown option: %c\n", optopt); 
                return 1;
        }  
    }  

    // optind is for the extra arguments 
    // which are not parsed 
    for(; optind < argc; optind++){      
        printf("extra arguments: %s\n", argv[optind]); 
        return 1; 
    } 

    return 0;
}