#include "execute.h"
#include <cstring>

#define MYLIB_API 

extern "C"
{
    Execute execute;
    MYLIB_API void  show_matrix(int* matrix, int rows, int columns) {
        execute.show_matrix(matrix, rows, columns);
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
    MYLIB_API int  GetQueueSize() { return execute.GetQueueSize(); }
}

char matrix[2][1920][1080][3];

int main(int argc, char* argv[])
{
    freopen("debug.log", "w+", stdout);

#if USE_RGB_FILE
    execute.TestReadFromFile((char *)"20240418_150649_001.mxf.rgb");
#else
    execute.TestReadFromFile((char*)"20240418_150649_001.mxf.yuv");
#endif
    int i = 0;
    int size = execute.GetQueueSize();
    for (size_t i = 0; i < size-1; i++)
    {
        std::cout << stamp() << "Execute::OneDeal begin" << std::endl;

        execute.GetFrameMatrix((char *)matrix[i%2], i);
        execute.GetFrameMatrix((char*)matrix[(i+1)%2], i+1);

        execute.FinishFrameGenerate((char*)matrix[i%2], i);

        std::cout << stamp() << "Execute::OneDeal End" << std::endl;
    }
}
