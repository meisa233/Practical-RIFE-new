#include "execute.h"
#include <cstring>


// python 传递数组到 c++ 端时, 需要将其拉平为 1 维.
void Execute::show_matrix(int* matrix, int rows, int columns) {
    int row, col;
    printf("the matrix point %p ==>%lld\n", matrix, matrix);

    for (row = 0; row < rows; row++) {
        for (col = 0; col < columns; col++) {
            printf("matrix[%d][%d] = %d\n", row, col, matrix[row * columns + col]);
        }
    }
}

void Execute::show_uchar_matrix(char* matrix, int rows, int columns) {
    int row, col;
    printf("the matrix point %p ==>%lld\n", matrix, matrix);
    for (row = 0; row < rows; row++) {
        for (col = 0; col < columns; col++) {
            printf("matrix[%d][%d] = %d\n", row, col, int(matrix[row * columns + col]));
        }
    }
}
/**
 * 测试从文件读取RGB数据.
 * 
 */
int Execute::TestReadFromFile(char * filename)
{
    std::ifstream file(filename, std::ios::binary);

    std::cout << stamp() << "Execute::TestReadFromFile begin,filename:" << filename << std::endl;

    if (!file.is_open()) {
        std::cerr << "Failed to open file!" << std::endl;
        return -1;
    }

    // 读取文件大小  
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    int Cnt = size / itemsize;

    // 读取文件内容到缓冲区  
    for (size_t i = 0; i < Cnt; i++)
    {
        VideoFrame* ptr = new VideoFrame();
        if (!ptr)
        {
            std::cerr << "malloc mem failed!" << std::endl;
            file.close();
            return -1;
        }
        if (!file.read((char *)buff, itemsize)) {
            std::cerr << "Failed to read file!" << std::endl;
            file.close();
            return -1;
        }
        ptr->dts = i;
        ptr->width = 1920;
        ptr->height = 1080;
        ptr->channel = 3;
#if USE_RGB_FILE
        TransformRGBtoMatrix(buff, (uint8_t ***)ptr->buffer, 1920, 1080, 3);
#else
        TransformYUVtoMatrix(buff, (uint8_t ***)ptr->buffer, 1920, 1080, 3);
#endif
        inputQueue.push_back(ptr);
    }
    // 关闭文件  
    file.close();

    std::cout << stamp() << "Execute::TestReadFromFile end cnt("<< Cnt<<")"<< std::endl;
    return 0;
}
/**
 * 将普通RGB转换为矩阵式.
 * 
 */
void Execute::TransformRGBtoMatrix(uint8_t *rgb, uint8_t *** rgbMatrix, int width, int height, int channel)
{
    std::cout << stamp() << "Execute::TransformRGBtoMatrix begin" << std::endl;

    uint8_t* rgbPtr = rgb;
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;

    uint8_t* rgbMatrixPtr = (uint8_t*)rgbMatrix;
    uint8_t* tmpPtr;
    memcpy(rgbMatrixPtr, rgb, itemsize);

  /* for (size_t k = 0; k < channel; k++)
    {
        for (i = 0; i < height; i++)
        {
            for (j = 0; j < width; j++)
            {
                tmpPtr = rgbMatrixPtr + i * width * channel + j * channel + k;
                *tmpPtr = *rgbPtr;
                rgbPtr++;
            }
        }
    }
    */
    std::cout << stamp() << "Execute::TransformRGBtoMatrix end" << std::endl;

}

/**
 * 将YUV422数据转换为RGB数据.
 * 
 * \param yuv422
 * \param rgbMatrix
 * \param width
 * \param height
 * \param channel
 */
void Execute::TransformYUVtoMatrix(uint8_t* yuv422, uint8_t*** rgbMatrix, int width, int height, int channel)
{
    uint8_t* yPtr, *uPtr, *vPtr;
    uint8_t* rgbMatrixPtr = (uint8_t*)rgbMatrix;

    int ylen = width * height;
    int ulen = ylen >> 1;
    int vlen = ulen;

    yPtr = yuv422;
    uPtr = yPtr + ylen;
    vPtr = uPtr + ulen;
    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            uint8_t* rgb = rgbMatrixPtr + i * width * channel + j;
            rgb[0] = (*yPtr) + 1.28033 * (*vPtr);
            rgb[1] = (*yPtr) - 0.21482 * (*uPtr) - 0.38059*(*vPtr);
            rgb[2] = (*yPtr) - 2.12798 * (*uPtr);
            yPtr++;
            uPtr += j % 2;
            vPtr += j % 2;
        }
    }

}

int Execute::GetFrameMatrix(char* matrix, int pts)
{
    size_t i;
    std::cout << stamp() << "Execute::GetFrameMatrix begin pts(" << pts << ")" << std::endl;

    std::unique_lock<std::mutex> lock(m_videoFrameQueueMutex);
    std::cout << stamp() << "Execute::GetFrameMatrix getLock pts(" << pts << ")" << std::endl;

    for (i= 0; i < inputQueue.size(); i++)
    {
        if (inputQueue[i]->dts == pts)
        {
            memcpy(matrix, inputQueue[i]->buffer, sizeof(inputQueue[i]->buffer));
            break;
        }

        if (inputQueue[i]->dts > pts)
        {
            return -1;
        }
    }

    if (i >= inputQueue.size())
    {
        return -11;
    }
    std::cout << stamp() << "Execute::GetFrameMatrix end pts(" << pts << ")" << std::endl;
    return 0;
}


int Execute::FinishFrameGenerate(char* matrix, int pts)
{
    std::cout << stamp() << "Execute::FinishFrameGenerate begin pts("<<pts<<")" << std::endl;

    size_t i;

    finishMap[pts] = true;

    std::unique_lock<std::mutex> lock(m_videoFrameQueueMutex);

    std::cout << stamp() << "Execute::FinishFrameGenerate getLock pts(" << pts << ")" << std::endl;


    for (auto it = inputQueue.begin(); it != inputQueue.end(); )
    {
        if ((*it)->dts < pts)
        {
            if (finishMap[(*it)->dts])
            {
                VideoFrame* ptr = *it;
                it = inputQueue.erase(it);
                delete (ptr);                
            }
            else
            {
                it++;
            }
        }
        else
        {
            break;
        }
    }
    std::cout << stamp() << "Execute::FinishFrameGenerate end pts(" << pts << ")" << std::endl;
    return 0;
}


std::string date_time(std::time_t posix)
{
    char buf[20]; // big enough for 2015-07-08 10:06:51\0
    std::tm tp = *std::localtime(&posix);
    return { buf, std::strftime(buf, sizeof(buf), "%F %T", &tp) };
}

std::string stamp()
{
    using namespace std;
    using namespace std::chrono;

    auto now = system_clock::now();

    // use microseconds % 1000000 now
    auto us = duration_cast<microseconds>(now.time_since_epoch()) % 1000000;

    std::ostringstream oss;
    oss.fill('0');

    oss << date_time(system_clock::to_time_t(now));
    oss << '.' << setw(6) << us.count();

    return oss.str();

}
