#ifndef __FACEBOXES_PREPROCESS_H_
#define __FACEBOXES_PREPROCESS_H_

#include <string>
#include <utility>

#include <opencv2/opencv.hpp>

template <typename T>
void crop(T *data, int height, int width, 
          T *crop_data, int y1, int x1, int crop_height, int crop_width, 
          int nchannels) {
    for (int i = 0; i < crop_height; ++i) {
        for (int j = 0; j < crop_width; ++j) {
            for (int c = 0; c < nchannels; ++c) {
                int crop_idx = i * crop_width * nchannels + j * nchannels + c;
                int src_idx = (i + y1) * width * nchannels + (j + x1) * nchannels + c;
                crop_data[crop_idx] = data[src_idx];
            }
        }
    }
}

cv::Mat faceboxes_preprocess(cv::Mat &rawimg, int height, int width, float &ratio) {
    cv::Mat dstimg;

    int imgheight = rawimg.rows;
    int imgwidth = rawimg.cols;
    ratio = std::min((float)height / imgheight, (float)width / imgwidth);
    int rszheight = round(imgheight * ratio);
    int rszwidth = round(imgwidth * ratio);
    cv::resize(rawimg, dstimg, cv::Size(rszwidth, rszheight));

    int bottom = height - rszheight;
    int right = width - rszwidth;
    cv::copyMakeBorder(dstimg, dstimg, 0, bottom, 0, right, cv::BORDER_CONSTANT, 0);

    dstimg.convertTo(dstimg, CV_8UC3);
    return std::move(dstimg);
}

template <typename T1, typename T2>
void cvtType(T1 dst, T2 src, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = src[i];
    }
}

template <typename T>
std::shared_ptr<T> copyto(std::vector<cv::Mat> images, int dp) {
    int persize = images[0].rows * images[0].cols * images[0].channels();
    std::shared_ptr<T> ptr(new T[dp * persize], [](T *ptr){delete [] ptr;});
    for (int i = 0; i < images.size(); ++i) {
        memcpy(ptr.get() + i * persize, images[i].ptr<T>(0), sizeof(T) * persize);
    }
    return std::move(ptr);
}   

#endif  // __FACEBOXES_PREPROCESS_H_
