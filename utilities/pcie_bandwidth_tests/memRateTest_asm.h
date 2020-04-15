
#ifndef MEM_RATE_TEST_ASM_H
#define MEM_RATE_TEST_ASM_H

#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <cassert>


// typedef void (*testfunc_type)(char* memarea, size_t size, size_t repeats);

// struct TestFunction
// {
//     // identifier of the test function
//     const char* name;

//     // function to call
//     testfunc_type func;

//     // prerequisite CPU feature
//     const char* cpufeat;

//     // number of bytes read/written per access (for latency calculation)
//     unsigned int bytes_per_access;

//     // bytes skipped foward to next access point (including bytes_per_access)
//     unsigned int access_offset;

//     // number of accesses before and after
//     unsigned int unroll_factor;

//     // fill the area with a permutation before calling the func
//     bool make_permutation;

//     // constructor which also registers the function
//     TestFunction(const char* n, testfunc_type f, const char* cf,
//                  unsigned int bpa, unsigned int ao, unsigned int unr,
//                  bool mp);

//     // test CPU feature support
//     bool is_supported() const;
// };

// std::vector<TestFunction*> g_testlist;

// TestFunction::TestFunction(const char* n, testfunc_type f, const char* cf,
//                            unsigned int bpa, unsigned int ao, unsigned int unr,
//                            bool mp)
//     : name(n), func(f), cpufeat(cf),
//       bytes_per_access(bpa), access_offset(ao), unroll_factor(unr),
//       make_permutation(mp)
// {
//     g_testlist.push_back(this);
// }

// #define REGISTER_CPUFEAT(func, cpufeat, bytes, offset, unroll)  \
//     static const struct TestFunction* _##func##_register =       \
//         new TestFunction(#func,func,cpufeat,bytes,offset,unroll,false);


#define HUGE_PAGES 0

static char *allocate(std::size_t size)
{
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (HUGE_PAGES)
        flags |= MAP_HUGETLB;
    void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    assert(addr != MAP_FAILED);
    return (char *) addr;
}

// 256-bit writer in an unrolled loop (Assembler version)
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

//REGISTER_CPUFEAT(ScanWrite256PtrUnrollLoop, "avx", 32, 32, 16);

// 256-bit reader in an unrolled loop (Assembler version)
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

//REGISTER_CPUFEAT(ScanRead256PtrUnrollLoop, "avx", 32, 32, 16);

#endif