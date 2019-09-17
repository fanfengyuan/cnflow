#include "cnflow.h"

#include <string>
#include <vector>

#include "cnrt.h"

int main() {
    const int device = 1;

    cnrtInit(0);
    cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, device);
    cnrtSetCurrentDevice(dev);

    std::vector<std::string> imagepaths(10000, "datas/face.jpg");

    const int dp_faceboxes = 1;

    cnflow::CnFlow flower;
    flower.device = device;
    flower.faceboxes_height = 500;
    flower.faceboxes_width = 500;
    flower.putImageList(imagepaths);
    flower.addFaceBoxesPreprocessEx(32);
    flower.addFaceBoxesInfer(32 / dp_faceboxes + 1, dp_faceboxes);
    flower.addFaceBoxesPostProcess(32);
    flower.join();

    return 0;
}
