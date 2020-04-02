#ifndef __TEST_HPP__
#define __TEST_HPP__

//TODO: Doxygen documentation for the class.
// Also a short explanation of how-to-use, either in the README.md but probably in the doxygen \detail section.
class Test
{
    public:
        virtual void ~Test() {};

        void  run_test();
        int   get_result();
        float get_time();
    
    protected:
        virtual void simulate_input() = 0;
        virtual void run_operation()  = 0;
        virtual void verify_output()  = 0;
    
    private:
        float m_fElapsedTime_ms;
        int   m_iResult;
}

#endif