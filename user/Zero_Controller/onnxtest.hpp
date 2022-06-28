#include <iostream>
#include <onnxruntime_c_api.h>
#include <assert.h>
#include <vector>


void CheckStatus(OrtStatus* status, const OrtApi* ortApi)
{
    if (status != NULL) {
        const char* sMsg = ortApi->GetErrorMessage(status);
        std::cerr << "ONNX Runtime error: " << sMsg << std::endl;
        ortApi->ReleaseStatus(status);
        exit(1);
    }
}

/*
Helper function to print the important info about a tensor, i.e. shape and data type.
*/
void PrintTensorInfo(std::string Type, size_t Idx, const char* Name, OrtTypeInfo* TypeInfo, const OrtApi* ortApi)
{
    const OrtTensorTypeAndShapeInfo* pTensorInfo;
    ONNXTensorElementDataType type;
    size_t iNumDims;
    std::vector<int64_t> NodeDims;

    CheckStatus(ortApi->CastTypeInfoToTensorInfo(TypeInfo, &pTensorInfo), ortApi);
    CheckStatus(ortApi->GetTensorElementType(pTensorInfo, &type), ortApi);
    CheckStatus(ortApi->GetDimensionsCount(pTensorInfo, &iNumDims), ortApi);
    NodeDims.resize(iNumDims);
    CheckStatus(ortApi->GetDimensions(pTensorInfo, (int64_t*)NodeDims.data(), iNumDims), ortApi);
    std::cout << "- " << Type << " " << Idx << ": name=" << Name << ", type=" << type << ", ndim=" << iNumDims << ", shape=[";
    for (size_t dimIdx = 0; dimIdx < iNumDims; dimIdx++)
    {
        std::cout << NodeDims[dimIdx] << ((dimIdx >= iNumDims-1)?"":",");
    }
    std::cout << "]" << std::endl;
}

int test_load_model()
{
    const OrtApi* pOrtApi = nullptr;
    OrtEnv* pOnnxEnv = nullptr;
    OrtSessionOptions* pOnnxSessionOpts;
    OrtSession* pOnnxSession = nullptr;
    OrtAllocator* pOnnxMemAllocator = nullptr;
    OrtMemoryInfo* pOnnxMemAllocInfo = nullptr;
    float pdData[48];
    int64_t piShape[] {1,48};
    OrtValue* pOnnxInput = nullptr;
    OrtValue* pOnnxOutput = nullptr;
    const char** psInputNames = nullptr;
    const char** psOutputNames = nullptr;
    size_t iNumInputs, iNumOutputs;
    int iFlag;
    float* pfOutputData = nullptr;

    pOrtApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    CheckStatus(pOrtApi->CreateEnv(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "onnx.log", &pOnnxEnv), pOrtApi);
    assert(pOnnxEnv != NULL);
    std::cout << "ONNX environment created successfully." << std::endl;

    CheckStatus(pOrtApi->CreateSessionOptions(&pOnnxSessionOpts), pOrtApi);
    CheckStatus(pOrtApi->SetSessionGraphOptimizationLevel(pOnnxSessionOpts, ORT_ENABLE_ALL), pOrtApi);

    // Create a session from the saved model using default settings, so it'll run on the CPU.

    const char* sModelName = "./data/policy_1.onnx";

    CheckStatus(pOrtApi->CreateSession(pOnnxEnv, sModelName, pOnnxSessionOpts, &pOnnxSession), pOrtApi);
    assert(pOnnxSession != NULL);
    std::cout << "ONNX session create successfully." << std::endl;

    // Print info about the inputs and outputs of the model.
    // Shouldd be one input and one output, both having dimensions of [-1,10] and data type equal to 1 (float).
    CheckStatus(pOrtApi->SessionGetInputCount(pOnnxSession, &iNumInputs), pOrtApi);
    std::cout << "Model has " << iNumInputs << " input(s)." << std::endl;
    CheckStatus(pOrtApi->GetAllocatorWithDefaultOptions(&pOnnxMemAllocator), pOrtApi);
    psInputNames = new const char*[iNumInputs]; // You need to keep track of the names of the inputs to the network so you can specify multiple inputs when runningt the network, if it has multiple inputs.
    for (size_t inIdx = 0; inIdx < iNumInputs; inIdx++)
    {
        char* sName;
        OrtTypeInfo* pTypeinfo;
        CheckStatus(pOrtApi->SessionGetInputName(pOnnxSession, inIdx, pOnnxMemAllocator, &sName), pOrtApi);
        CheckStatus(pOrtApi->SessionGetInputTypeInfo(pOnnxSession, inIdx, &pTypeinfo), pOrtApi);
        psInputNames[inIdx] = sName;
        PrintTensorInfo("Input", inIdx, sName, pTypeinfo, pOrtApi);
        pOrtApi->ReleaseTypeInfo(pTypeinfo);
    }
    CheckStatus(pOrtApi->SessionGetOutputCount(pOnnxSession, &iNumOutputs), pOrtApi);
    std::cout << "Model has " << iNumOutputs << " output(s)." << std::endl;
    psOutputNames = new const char*[iNumOutputs]; // Also need to get the name of the network's output so that one can tell the runtime which outputs to calculate.
    for (size_t outIdx = 0; outIdx < iNumOutputs; outIdx++)
    {
        char* sName;
        OrtTypeInfo* pTypeinfo;
        CheckStatus(pOrtApi->SessionGetOutputName(pOnnxSession, outIdx, pOnnxMemAllocator, &sName), pOrtApi);
        CheckStatus(pOrtApi->SessionGetOutputTypeInfo(pOnnxSession, outIdx, &pTypeinfo), pOrtApi);
        psOutputNames[outIdx] = sName;
        PrintTensorInfo("Output", outIdx, sName, pTypeinfo, pOrtApi);
        pOrtApi->ReleaseTypeInfo(pTypeinfo);
    }

    for (size_t idx = 0; idx < 48; idx++)
        pdData[idx] = 1.0f;

    // Create the input tensor and fill it with data.
    CheckStatus(pOrtApi->CreateCpuMemoryInfo(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault, &pOnnxMemAllocInfo), pOrtApi);
    std::cout << "Created CPU memory allocator." << std::endl;
    CheckStatus(pOrtApi->CreateTensorWithDataAsOrtValue(pOnnxMemAllocInfo, pdData, sizeof(float) * 48, piShape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &pOnnxInput), pOrtApi);
    assert(pOnnxInput != NULL);
    CheckStatus(pOrtApi->IsTensor(pOnnxInput, &iFlag), pOrtApi);
    assert(iFlag);
    std::cout << "Allocated input tensor." << std::endl;

    // Finally execute the neural network.
    CheckStatus(pOrtApi->Run(pOnnxSession, NULL, (const char* const*)psInputNames, (const OrtValue* const*)&pOnnxInput, 1, (const char* const*)psOutputNames, 1, &pOnnxOutput), pOrtApi);

    CheckStatus(pOrtApi->GetTensorMutableData(pOnnxOutput, (void**)&pfOutputData), pOrtApi);
    std::cout << "Output tensor: [";
    for (size_t idx = 0; idx < 12; idx++)
        std::cout << pfOutputData[idx] << ((idx==12-1)?"":",");
    std::cout << "]" << std::endl;

    std::cout << "Verifying results: ";
    bool bSuccess = true;
    for (size_t idx = 0; idx < 12; idx++)
    {
        bSuccess &= pfOutputData[idx] == 561.0;
    }
    std::cout << (bSuccess?"PASS":"FAIL") << std::endl;

    // Cleanup time.
    delete psInputNames;
    delete psOutputNames;
    pOrtApi->ReleaseValue(pOnnxInput);
    pOrtApi->ReleaseValue(pOnnxOutput);
    pOrtApi->ReleaseMemoryInfo(pOnnxMemAllocInfo);
    // pOrtApi->ReleaseAllocator(pOnnxMemAllocator);
    // OrtReleaseSession(pOnnxSession);
    pOrtApi->ReleaseEnv(pOnnxEnv);
    std::cout << "ONNX Runtime variables released." << std::endl;

    return 0;
}