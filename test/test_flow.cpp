#include "cnflow.h"

#include <string>
#include <vector>

#include "cnrt.h"

int main() {
    const int device = 0;

    std::vector<std::string> imagepaths(10000, "datas/face.jpg");

    const int dp_faceboxes = 1;

    cnflow::CnFlow flower;
    flower.faceboxes_model_path = "offline_models/faceboxes-500x500.cambricon";
    flower.faceboxes_func_name = "fusion_0";
    flower.device = device;
    flower.fake_input = false;
    flower.putImageList(imagepaths);
    flower.addFaceBoxesPreprocessEx(32);
    flower.addFaceBoxesInfer(32 / dp_faceboxes + 1, dp_faceboxes);
    flower.addFaceBoxesPostProcess(32);
    flower.join();

    return 0;
}
