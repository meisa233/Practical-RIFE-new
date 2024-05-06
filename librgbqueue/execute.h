#pragma once
#ifndef PY_C_TEST_EXCUTE_H
#define PY_C_TEST_EXCUTE_H

#define USE_RGB_FILE 1
#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>  
#include <mutex>
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/core/core.hpp"

#include <iomanip>
#include <sstream>  

std::string date_time(std::time_t posix);
std::string stamp();


typedef struct Result {
    int a;
    const char* b;
} result;

typedef struct VideoFrame
{
    int dts;
    int width, height, channel;

    uint8_t buffer[1920][1080][3];
}VideoFrame;




class Execute {
public:
    // 传递int数组到c++
    void show_matrix(int* matrix, int rows, int columns);

    // 传递uchar数组到c++
    void show_uchar_matrix(char* matrix, int rows, int columns);

    int TestReadFromFile(char* filename);

    void TransformRGBtoMatrix(uint8_t* rgb, uint8_t ***rgbMatrix, int width, int height, int channel);
    void TransformYUVtoMatrix(uint8_t* yuv422, uint8_t*** rgbMatrix, int width, int height, int channel);

    int  GetFrameMatrix(char* matrix, int pts);
    int FinishFrameGenerate(char* matrix, int pts);
    int  GetQueueSize() { return inputQueue.size(); }



private:
    std::vector<VideoFrame*> inputQueue;
    std::mutex               m_videoFrameQueueMutex;

    std::unordered_map<int, bool> finishMap;
#if USE_RGB_FILE
    static const int itemsize = 1920 * 1080 * 3;
#else
    static const int itemsize = 1920 * 1080 * 2;
#endif
    // 分配内存以存储文件内容  
    uint8_t buff[itemsize];
};


#endif //PY_C_TEST_EXCUTE_H