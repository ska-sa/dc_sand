#ifndef MEM_RATE_TEST_ASM_H
#define MEM_RATE_TEST_ASM_H

#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <cassert>

//Set to 1 to use huge pages. Remanant of old code from SDP. This could be adjusted so that this is a parameter in the allocate function
#define HUGE_PAGES 0

/** Function to allocate memory. Using mmap instead of malloc as it allows for huge pages to be used.
 *  \param size Size in bytes to allocate 
 *  \param useHugePages Set to 0 to not use huge pages. Huge pages need to be configured correctly on unix OS or else this throws an error when trying to allocate huge pages.
 */

static char *allocate(std::size_t size, uint8_t useHugePages)
{
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (useHugePages)
        flags |= MAP_HUGETLB;
    void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    assert(addr != MAP_FAILED);
    return (char *) addr;
}

/** \brief Assembler code to perform an unrolled write to RAM using the 256-bit AVX registers.
 * 
 *  \details Assembler code to perform an unrolled write to RAM using the 256-bit AVX registers. For some reason writing to RAM 
 *  reports an order of magnitude faster data rates than reading from RAM. I do not think this is correct but I would need
 *  to investigate further.
 *  \param memarea  Pointer to buffer to write data to. Pointer should have been allocated with \ref allocate() function.
 *  \param size     Number of bytes to write in a single transfer.
 *  \param repeats  Number of times to repeat the transfer.
 */
void ScanWrite256PtrUnrollLoop(char* memarea, size_t size, size_t repeats)
{
    uint64_t value = 0xC0FFEEEEBABE0000;

    asm volatile(
        "vbroadcastsd %[value], %%ymm0 \n" // ymm0 = test value
        "1: \n" // start of repeat loop
        "mov    %[memarea], %%rax \n"   // rax = reset loop iterator
        "2: \n" // start of write loop
        "vmovdqa %%ymm0, 0*32(%%rax) \n"
        "vmovdqa %%ymm0, 1*32(%%rax) \n"
        "vmovdqa %%ymm0, 2*32(%%rax) \n"
        "vmovdqa %%ymm0, 3*32(%%rax) \n"
        "vmovdqa %%ymm0, 4*32(%%rax) \n"
        "vmovdqa %%ymm0, 5*32(%%rax) \n"
        "vmovdqa %%ymm0, 6*32(%%rax) \n"
        "vmovdqa %%ymm0, 7*32(%%rax) \n"
        "vmovdqa %%ymm0, 8*32(%%rax) \n"
        "vmovdqa %%ymm0, 9*32(%%rax) \n"
        "vmovdqa %%ymm0, 10*32(%%rax) \n"
        "vmovdqa %%ymm0, 11*32(%%rax) \n"
        "vmovdqa %%ymm0, 12*32(%%rax) \n"
        "vmovdqa %%ymm0, 13*32(%%rax) \n"
        "vmovdqa %%ymm0, 14*32(%%rax) \n"
        "vmovdqa %%ymm0, 15*32(%%rax) \n"
        "add    $16*32, %%rax \n"
        // test write loop condition
        "cmp    %[end], %%rax \n"       // compare to end iterator
        "jb     2b \n"
        // test repeat loop condition
        "dec    %[repeats] \n"          // until repeats = 0
        "jnz    1b \n"
        : [repeats] "+r" (repeats)
        : [memarea] "r" (memarea), [end] "r" (memarea+size),
          [value] "m" (value)
        : "rax", "xmm0", "cc", "memory");
}


/** Assembler code to perform an unrolled read from RAM using the 256-bit AVX registers.
 *  \param memarea  Pointer to buffer to read data from. Pointer should have been allocated with \ref allocate() function.
 *  \param size     Number of bytes to read in a single transfer.
 *  \param repeats  Number of times to repeat the transfer.
 */
void ScanRead256PtrUnrollLoop(char* memarea, size_t size, size_t repeats)
{
    asm volatile(
        "1: \n" // start of repeat loop
        "mov    %[memarea], %%rax \n"   // rax = reset loop iterator
        "2: \n" // start of read loop
        "vmovdqa 0*32(%%rax), %%ymm0 \n"
        "vmovdqa 1*32(%%rax), %%ymm0 \n"
        "vmovdqa 2*32(%%rax), %%ymm0 \n"
        "vmovdqa 3*32(%%rax), %%ymm0 \n"
        "vmovdqa 4*32(%%rax), %%ymm0 \n"
        "vmovdqa 5*32(%%rax), %%ymm0 \n"
        "vmovdqa 6*32(%%rax), %%ymm0 \n"
        "vmovdqa 7*32(%%rax), %%ymm0 \n"
        "vmovdqa 8*32(%%rax), %%ymm0 \n"
        "vmovdqa 9*32(%%rax), %%ymm0 \n"
        "vmovdqa 10*32(%%rax), %%ymm0 \n"
        "vmovdqa 11*32(%%rax), %%ymm0 \n"
        "vmovdqa 12*32(%%rax), %%ymm0 \n"
        "vmovdqa 13*32(%%rax), %%ymm0 \n"
        "vmovdqa 14*32(%%rax), %%ymm0 \n"
        "vmovdqa 15*32(%%rax), %%ymm0 \n"
        "add    $16*32, %%rax \n"
        // test read loop condition
        "cmp    %[end], %%rax \n"       // compare to end iterator
        "jb     2b \n"
        // test repeat loop condition
        "dec    %[repeats] \n"          // until repeats = 0
        "jnz    1b \n"
        : [repeats] "+r" (repeats)
        : [memarea] "r" (memarea), [end] "r" (memarea+size)
        : "rax", "xmm0", "cc", "memory");
}
#endif