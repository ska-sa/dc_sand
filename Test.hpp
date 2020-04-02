#ifndef __TEST_HPP__
#define __TEST_HPP__

///\todo Examples of how to use this class.
///\todo brief and detailed descriptions.
class Test
{
    public:
        /// The destructor must be virtual to ensure that derived classes' destructors are also called when the objects are destroyed.
        virtual ~Test();

        /// Execute the test.
        void  run_test();
        /// Retrieve the test result.
        int   get_result();
        /// Retrieve the amount of time that the test took.
        float get_time();
    
    protected:
        // The constructor is protected as a reminder that we can't instantiate a pure virtual class directly.
        Test();

        // These functions make the class pure virtual and must be implelented by derived classes.
        ///\brief Generate simulated input data.
        virtual void simulate_input() = 0;
        virtual void run_operation()  = 0;
        virtual int  verify_output()  = 0;
    
    private:
        /// For recording the start of the test.
        cudaEvent_t m_StartEvent;
        /// For recording the end of the test.
        cudaEvent_t m_StopEvent;

        /// Stores the elapsed time of the test once completed. Unit: ms.
        float m_fElapsedTime_ms;
        /// Stores the result of the test.
        int   m_iResult;
}

#endif