#include "cnflow.h"

#include <string>
#include <vector>

#include "cnrt.h"

int main() {
    const int device = 0;

    std::vector<std::string> imagepaths(10000, "datas/face.jpg");

    // Set to 1 for MLU270
    const int dp_faceboxes = 1;

    // The cycle time for imagepaths. If -1, it will not stop.
    int epoch = 1;

    cnflow::CnFlow flower;
    // Set the cambricon offline model path.
    flower.faceboxes_model_path = "offline_models/faceboxes-500x500.cambricon";
    // Set the cambricon offline model func name.
    flower.faceboxes_func_name = "fusion_0";
    // Set the device id.
    flower.device = device;
    // If true, use the fake image (all 1) for input.
    flower.fake_input = false;
    flower.putImageList(imagepaths, epoch);
    flower.addFaceBoxesPreprocessEx(32);
    flower.addFaceBoxesInfer(32 / dp_faceboxes + 1, dp_faceboxes);
    flower.addFaceBoxesPostProcess(32);
    flower.join();

    return 0;
}
