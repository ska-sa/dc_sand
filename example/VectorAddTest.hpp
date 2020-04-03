#ifndef __VECTORADDTEST_HPP__
#define __VECTORADDTEST_HPP__

#include "UnitTest.hpp"
#include <cstdint>

class VectorAddTest : public UnitTest
{
    public:
        VectorAddTest(size_t uVectorLength);
        ~VectorAddTest();

    protected:
        void simulate_input() override;

        void transfer_HtoD() override;

        void run_kernel() override;

        void transfer_DtoH() override;

        void verify_output() override;

    private:
        //Host pointers
        int *m_piHVectorA;
        int *m_piHVectorB;
        int *m_piHVectorC;

        //device pointers
        int *m_piDVectorA;
        int *m_piDVectorB;
        int *m_piDVectorC;
        
        size_t m_uVectorLength;
};

#endif