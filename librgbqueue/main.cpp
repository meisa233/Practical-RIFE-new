#include "execute.h"

#if (defined(_WIN32) || defined(_WIN64))
#define MYLIB_API __declspec(dllexport)  
#else  
#define MYLIB_API  
#endif  

extern "C"
{
    Execute execute;
    MYLIB_API void  show_matrix(int* matrix, int rows, int columns) {
        execute.show_matrix(matrix, rows, columns);
    }
    MYLIB_API void  show_matrix_uint8(uint8_t* matrix, int rows, int columns, int channel)
    {
        execute.show_matrix_uint8(matrix, rows, columns, channel);
    }

    // 传递uchar数组到c++
    MYLIB_API void  show_uchar_matrix(char* matrix, int rows, int columns) {
        execute.show_uchar_matrix(matrix, rows, columns); 
    }

    //void TransformRGBtoMatrix(uint8_t* rgb, uint8_t*** rgbMatrix, int width, int height, int channel);
    //int  GetFrameMatrix(char* matrix, int pts);

    MYLIB_API int  GetFrameMatrix(char* matrix, int pts) {
        return execute.GetFrameMatrix(matrix, pts);
    }

    MYLIB_API int FinishFrameGenerate(char* matrix, int pts){
        return execute.FinishFrameGenerate(matrix, pts);
    }    

  //    int TestReadFromFile(const char* filename);
    MYLIB_API int  TestReadFromFile(char* filename){
        return execute.TestReadFromFile(filename);
    }

    MYLIB_API void startTcpServerThread() {
        return execute.startTcpServerThread();
    }
    MYLIB_API int  GetQueueSize() { return execute.GetQueueSize(); }
}

char matrix[2][1920][1080][3];

int main(int argc, char* argv[])
{
    freopen("debug.log", "w+", stdout);

#if USE_RGB_FILE
    execute.TestReadFromFile((char *)"20240418_150649_001.mxf.rgb");
#else
    //execute.TestReadFromFile((char*)"20240418_150649_001.mxf.yuv");

    execute.startTcpServerThread();
#endif
    int i = 0;
    int size = 0;
    int ret = 0;
    do
    {
        static int num = 0;
        size = execute.GetQueueSize();
        if (size <= 1)
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << stamp() << "sleep for "<<num++<<" seconds!" << std::endl;
        }
    } while (size <=0);

    for (size_t i = 0; ; )
    {
        std::cout << stamp() << "Execute::OneDeal begin" << std::endl;

        ret = execute.GetFrameMatrix((char *)matrix[i%2], i);
        if (ret < 0)
        {
            if (ret == -EAGAIN)
            {
                if (execute.isRecvSocketOpen())
                {
                    std::cout << stamp() << "Execute::OneDeal sleep1" << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                } 
            }
            break;
        }

        ret = execute.GetFrameMatrix((char*)matrix[(i+1)%2], i+1);
        if (ret < 0)
        {
            if (ret == -EAGAIN)
            {
                if (execute.isRecvSocketOpen())
                {
                    std::cout << stamp() << "Execute::OneDeal sleep2" << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
            }
            break;
        }
        execute.FinishFrameGenerate((char*)matrix[i%2], i);

        i++;

        std::cout << stamp() << "Execute::OneDeal End" << std::endl;
    }

    execute.stopTcpServerThread();
    std::cout << stamp() << "Execute::main End" << std::endl;

}
