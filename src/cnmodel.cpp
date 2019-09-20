#include "cnmodel.h"
#include "tsque.h"

namespace cnmodel {

uint64_t time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000000 + ts.tv_nsec / 1000);
}

cnrtRet_t cnrtMallocBuffer(uint32_t frame_num, 
                           uint32_t frame_size, 
                           uint32_t data_parallelism, 
                           void **mlu_ptrs) {
    void *param = nullptr;
    static int type = CNRT_MALLOC_EX_PARALLEL_FRAMEBUFFER;
    cnrtAllocParam(&param);
    cnrtAddParam(param, (char *)"type", sizeof(type), &type);
    cnrtAddParam(param, (char *)"data_parallelism", sizeof(data_parallelism), 
                 &data_parallelism);
    cnrtAddParam(param, (char *)"frame_num", sizeof(frame_num), &frame_num);
    cnrtAddParam(param, (char *)"frame_size", sizeof(frame_size), &frame_size);
    cnrtMallocBufferEx(mlu_ptrs, param);
    cnrtDestoryParam(param);
    return CNRT_RET_SUCCESS;
}

CnMemManager::CnMemManager(size_t num_buffers, std::vector<size_t> size_in_bytes) {
    n_size = size_in_bytes.size();
    for (int i = 0; i < num_buffers; ++i) {
        void **ptrs;
        ptrs = (void **)malloc(sizeof(void *) * n_size);
        for (int n = 0; n < n_size; ++n) {
            void *ptr = nullptr;
            CNRT_CHECK_V2(cnrtMalloc(&ptr, size_in_bytes[n]));
            // CNRT_CHECK_V2(cnrtMallocBuffer(1, size_in_bytes[n], 1, &ptr));
            ptrs[n] = ptr;
        }
        buffer.push(ptrs);
    }
}

CnMemManager::~CnMemManager() {
    while (buffer.size() > 0) {
        void **ptrs = buffer.pop();
        for (int n = 0; n < n_size; ++n) {
            CNRT_CHECK_V2(cnrtFree(ptrs[n]));
        }
        free(ptrs);
    }
}

void **CnMemManager::pop() {
    return buffer.pop();
}

void CnMemManager::push(void **ptrs) {
    buffer.push(ptrs);
}

CnModel::CnModel(const char *_modelpath, const char *_funcname, 
                   int _device, int _dp,
                   bool need_buffer,
                   cnrtDataType_t _input_dtype, cnrtDimOrder_t _input_order,
                   cnrtDataType_t _output_dtype, cnrtDimOrder_t _output_order):
  modelpath(_modelpath), funcname(_funcname), device(_device), dp(_dp) {
    cnrtInit(0);
    cnrtDev_t dev;
    CNRT_CHECK_V2(cnrtGetDeviceHandle(&dev, device));
    CNRT_CHECK_V2(cnrtSetCurrentDevice(dev));

    CNRT_CHECK_V2(cnrtLoadModel(&model, modelpath.c_str()));   
    CNRT_CHECK_V2(cnrtCreateFunction(&function));
    CNRT_CHECK_V2(cnrtExtractFunction(&function, model, funcname.c_str()));
    CNRT_CHECK_V2(cnrtGetInputDataDesc(&input_descS, &input_num, function));
    CNRT_CHECK_V2(cnrtGetOutputDataDesc(&output_descS, &output_num, function));

    cnrtInitFuncParam_t init_func_param;
    init_func_param.data_parallelism = &dp;
    init_func_param.affinity = &affinity;
    init_func_param.muta = &muta;
    init_func_param.end = CNRT_PARAM_END;
    CNRT_CHECK_V2(cnrtInitFunctionMemory_V2(function, &init_func_param));
    invoke_func_param.affinity = &affinity;
    invoke_func_param.data_parallelism = &dp;
    invoke_func_param.end = CNRT_PARAM_END;
    CNRT_CHECK_V2(cnrtCreateStream(&stream));
    CNRT_CHECK_V2(cnrtCreateEvent(&event_start));
    CNRT_CHECK_V2(cnrtCreateEvent(&event_end));

    int buffer_size = 33;

    input_data_bytes.resize(0);
    input_shapes.resize(input_num);
    for (int i = 0; i < input_num; ++i) {
        cnrtDataDesc_t data_desc = input_descS[i];
        CNRT_CHECK_V2(cnrtSetHostDataLayout(data_desc, _input_dtype, _input_order));

        uint32_t n, c, h, w;
        cnrtGetDataShape(data_desc, &n, &c, &h, &w);
        int data_size = n * h * w * ALIGN_UP(c, 128 / sizeof(uint16_t));
        input_data_bytes.push_back(ALIGN_UP(sizeof(uint16_t) * data_size, 64 * 1024));

        input_shapes[i].n = n;
        input_shapes[i].c = c;
        input_shapes[i].h = h;
        input_shapes[i].w = w;
    }
    std::vector<size_t> ibytes;
    for (int i = 0; i < input_num; ++i) {
        ibytes.push_back(dp * input_data_bytes[i]);
    }
    if (buffer_size > 0 && need_buffer)
        input_buffer = new CnMemManager(buffer_size, ibytes);
    for (int i = 0; i < output_num; ++i) {
        int data_count;
        cnrtDataDesc_t data_desc = output_descS[i];
        CNRT_CHECK_V2(cnrtSetHostDataLayout(data_desc, _output_dtype, _output_order));
        CNRT_CHECK_V2(cnrtGetHostDataCount(data_desc, &data_count));
        output_data_counts.push_back(data_count);

        uint32_t n, c, h, w;
        cnrtGetDataShape(data_desc, &n, &c, &h, &w);
        int data_size = n * h * w * ALIGN_UP(c, 128 / sizeof(uint16_t));
        output_data_bytes.push_back(ALIGN_UP(sizeof(uint16_t) * data_size, 64 * 1024));
    }
    std::vector<size_t> obytes;
    for (int i = 0; i < output_num; ++i) {
        obytes.push_back(dp * output_data_bytes[i]);
    }
    if (buffer_size > 0 && need_buffer)
        output_buffer = new CnMemManager(buffer_size, obytes);
}

void CnModel::invoke_ex(void **_input_mlu_ptrS, void **_output_mlu_ptrS) {
    void *param[input_num + output_num]; 
    for (int i = 0; i < input_num; ++i){
        param[i] = _input_mlu_ptrS[i];
    }
    for (int i = 0; i < output_num; ++i){
        param[input_num + i] = _output_mlu_ptrS[i];
    }
    
    CNRT_CHECK_V2(cnrtPlaceEvent(event_start, stream));
    CNRT_CHECK_V2(cnrtInvokeFunction(function, dim, param, func_type, stream, (void *)&invoke_func_param));
    CNRT_CHECK_V2(cnrtPlaceEvent(event_end, stream));
    CNRT_CHECK_V2(cnrtSyncStream(stream));
    CNRT_CHECK_V2(cnrtEventElapsedTime(event_start, event_end, &ptv));
}

void **CnModel::deviceAllocInput() {
    return input_buffer->pop();
}

void **CnModel::deviceAllocOutput() {
    return output_buffer->pop();
}

void CnModel::copyin(void **mlu_ptr, void **cpu_ptr) {
    CNRT_CHECK_V2(cnrtMemcpyBatchByDescArray(mlu_ptr, cpu_ptr, 
        input_descS, input_num, dp, CNRT_MEM_TRANS_DIR_HOST2DEV));
}

std::shared_ptr<std::shared_ptr<float>> CnModel::copyout(void **mlu_ptr) {
    std::vector<void *> output_cpu_ptrS(output_num);
    std::shared_ptr<std::shared_ptr<float>> s_output_cpu_ptrS(new std::shared_ptr<float>[output_num], [](std::shared_ptr<float> *ptr){delete [] ptr;});
    for (int i = 0; i < output_num; ++i) {
        std::shared_ptr<float> s_output_ptr(new float[output_data_counts[i] * dp], [](float *ptr){delete [] ptr;});
        s_output_cpu_ptrS.get()[i] = s_output_ptr;
        output_cpu_ptrS[i] = s_output_ptr.get();
    }

    CNRT_CHECK_V2(cnrtMemcpyBatchByDescArray(output_cpu_ptrS.data(), mlu_ptr, 
        output_descS, output_num, dp, CNRT_MEM_TRANS_DIR_DEV2HOST));
    return s_output_cpu_ptrS;
}


void CnModel::freeInput(void **input_mlu) {
    input_buffer->push(input_mlu);
}

void CnModel::freeOutput(void **output_mlu) {
    output_buffer->push(output_mlu);
}

std::shared_ptr<std::shared_ptr<float>> CnModel::invoke(void **ptr) {
        uint64_t t1 = time();

        void **input_mlu_ptr = deviceAllocInput();
        void **output_mlu_ptr = deviceAllocOutput();

        uint64_t t2 = time(); 

        copyin(input_mlu_ptr, ptr);

        uint64_t t3 = time();

        /// The input image size should be 60 X 60
        invoke_ex(input_mlu_ptr, output_mlu_ptr);

        uint64_t t4 = time();

        auto output = copyout(output_mlu_ptr);

        uint64_t t5 = time();

        freeInput(input_mlu_ptr);
        freeOutput(output_mlu_ptr);

        uint64_t t6 = time();
        LOG(INFO) << " Malloc time: " << t2 - t1 
                  << " Copy in time: " << t3 - t2
                  << " Invoke time: " << t4 - t3
                  << " Copy out time: " << t5 - t4
                  << " Free time: " << t6 - t5
                  << " us";

        return output;
}

CnModel::~CnModel() {
    CNRT_CHECK_V2(cnrtDestroyStream(stream));
    CNRT_CHECK_V2(cnrtDestroyFunction(function));
    CNRT_CHECK_V2(cnrtUnloadModel(model));
    CNRT_CHECK_V2(cnrtDestroyEvent(&event_start));
    CNRT_CHECK_V2(cnrtDestroyEvent(&event_end));
}

}  // namespace mlu
