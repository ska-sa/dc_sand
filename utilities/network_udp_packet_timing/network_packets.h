
#include <time.h> //Gives access clock time
#include <stdint.h>

#define PACKET_SIZE_BYTES   4096 //4KiB

//UDP Port to use for sending streams of test data
#define UDP_TEST_PORT       8080

//Number of packets to be sent in a stream. TODO: Replace with a configurable command line parameter.
#define NUMBER_OF_PACKETS   40000

/** Struct that stores \ref UdpTestingPacket metadata. The header struct is seperate from the packet struct as this 
 *  allows fields to be added to the header without having to manually re-calculate the the size of the 
 *  \ref UdpTestingPacket data payload. The size of the data payload in \ref UdpTestingPacket can instead be set to 
 *  data[PACKET_SIZE_BYTES-sizeof(struct UdpTestingPacketHeader).
 */
struct UdpTestingPacketHeader{
    struct tm cTransmitTime; //Time first packet in stream was sent
    int32_t i32PacketIndex; //Index of the current packet in stream
};

/** Packet that will be transmitted to the server. Contains dummy data and useful header data. See 
 *  \ref UdpTestingPacketHeader for more detail on how the size is kept consistent.
 */
struct UdpTestingPacket{
    struct UdpTestingPacketHeader header; // Header containing all useful data
    uint8_t data[PACKET_SIZE_BYTES-sizeof(struct UdpTestingPacketHeader)]; //Dummy data to be part of the packet
};


/** Metadata packet that will be transmitted out of band from the server to the client for configuring the test
 */
struct MetadataPacketMaster{
    uint32_t u32MetadataPacketCode; //Code specifying the type of metadata within the packet
    struct tm cSpecifiedTransmitTime; //Time the client must start transmitting data.
    float fWaitAfterStreamTransmitted_ms;
};

/** Metadata packet that will be transmitted out of band from the client to the server for configuring the test
 */
struct MetadataPacketClient{
    uint32_t u32MetadataPacketCode; //Code specifying the type of metadata within the packet
    float fTransmitTimeClient_ms;
};