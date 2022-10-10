// ConsoleApplication2.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std::chrono;

cv::dnn::Net _Net;
const std::string _Labels[1] = { "Axle" };

class Target
{
public:
    std::string Label;
    //std::string NumLabel;
    int ClassId;
    float Confidence;
    float Probability;
    cv::Rect2d Box;
public:
    Target()
    {

    }
    Target(std::string label, int classId, float confidence, float probability, cv::Rect2d box)
    {
        Label = label;
        ClassId = classId;
        Confidence = confidence;
        Probability = probability;
        Box = box;
    }
    cv::Point2d GetBoxCenter()
    {
        return cv::Point2d(Box.x + Box.width / 2, Box.y + Box.height / 2);
    }
    Target Copy()
    {
        Target m = Target();
        m.Box = cv::Rect2d();
        m.Label = this->Label;
        m.ClassId = this->ClassId;
        m.Confidence = this->Confidence;
        m.Probability = this->Probability;
        m.Box.x = this->Box.x;
        m.Box.y = this->Box.y;
        m.Box.width = this->Box.width;
        m.Box.height = this->Box.height;
        return m;
    }
};


class PredictResult
{
public:
    PredictResult()
    {

    }
    ~PredictResult()
    {
        //Img.release();
    }
    PredictResult(cv::Mat img)
    {
        img.copyTo(Img);
    }
    void Dispose()
    {
        //Img.release();
    }
    cv::Mat Img;
    bool ErrFlag = false;
    std::vector<int> AxialType;
    std::vector<Target> Targets;
    std::vector<Target> ErrTargets;
};


PredictResult GetResultDirect(cv::Mat& output, cv::Mat& image, float threshold, float nmsThreshold, int SIZE_Width, int SIZE_Height)
{
    output = output.reshape(1, (SIZE_Width / 32 * SIZE_Width / 32 + SIZE_Width / 16 * SIZE_Width / 16 + SIZE_Width / 8 * SIZE_Width / 8) * 3);
    auto classIds = std::vector<int>();
    auto confidences = std::vector<float>();
    auto probabilities = std::vector<float>();
    auto boxes = std::vector<cv::Rect2d>();
    for (int i = 0; i < output.rows; i++)
    {
        auto confidence = output.at<float>(i, 4);
        if (confidence > threshold)
        {
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(output.row(i).colRange(5, output.cols), NULL, NULL, &minLoc,
                &maxLoc);
            auto prob = (output.at<float>(i, 5 + maxLoc.x));
            if (prob > threshold)
            {
                auto cx = output.at<float>(i, 0);
                auto cy = output.at<float>(i, 1);
                auto w = output.at<float>(i, 2);
                auto h = output.at<float>(i, 3);
                auto left = cx - w / 2.f;
                auto top = cy - h / 2.f;
                classIds.push_back(maxLoc.x);
                confidences.push_back(confidence);
                probabilities.push_back(prob);
                boxes.push_back(cv::Rect2d(left, top, w, h));
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, threshold, nmsThreshold, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        cv::rectangle(image,
            cv::Rect((int)boxes[idx].x, (int)boxes[idx].y, (int)boxes[idx].width,
                (int)boxes[idx].height), cv::Scalar(0, 0, 255), 3);
    }

    PredictResult result = PredictResult(image);
    for (auto i : indices)
    {
        result.Targets.push_back(Target(_Labels[classIds[i]], classIds[i], confidences[i],
            probabilities[i], boxes[i]));
    }

    //result.Targets = Filter1(result.Targets, 0.3);
    return result;
}


PredictResult GetResult5_1(cv::Mat& org, long& costTime, float threshold, float nmsThreshold, int SIZE_Width, int SIZE_Height)
{

    cv::Mat mat = org.clone();
    cv::Mat blob = cv::dnn::blobFromImage(mat, 1.0 / 255, cv::Size(SIZE_Width, SIZE_Height), cv::Scalar(), true, false);
    _Net.setInput(blob);
    std::vector<std::string> outNames = _Net.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outs(outNames.size());
    _Net.forward(outs, outNames[0]);
    auto result = GetResultDirect(outs[0], mat, threshold, nmsThreshold, SIZE_Width, SIZE_Height);

    return result;

}


int main()
{
    
    _Net = cv::dnn::readNet("J:\\1.Deep_learning\\3.模型文件\\model\\Axle-SN-20220826\\v5_best_wheel+whitebg+kdt_m_20220826.onnx");
    _Net.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA);
    _Net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    int SIZE_Width = 640;
    int SIZE_Height = 640;
    cv::Mat mat = cv::imread("I:\\C++\\test\\ConsoleApplication2\\x64\\Debug\\0_XK70526_white_2022-2-4-15-39-50-560_Q2_LaneNo1_VehType1_AxleNum2_Class11_XKH1_DKH4_ZXH0_JZX0_ZX0_YW0_H2364.jpg");
    float ratio = (float)SIZE_Width / MAX(mat.cols, mat.rows);
    int new_w = (int)(mat.cols * ratio);
    new_w = new_w > SIZE_Width - 1 ? SIZE_Width : new_w;
    int new_h = (int)(mat.rows * ratio);
    new_h = new_h > SIZE_Height - 1 ? SIZE_Height : new_h;
    int dw = SIZE_Width - new_w;
    int dh = SIZE_Height - new_h;
    auto dw1 = dw / 2;
    auto dh1 = dh / 2;
    auto dw2 = dw - dw1;
    auto dh2 = dh - dh1;
    auto newSize = cv::Size(new_w, new_h);
    cv::resize(mat, mat, newSize, 0, 0, /*ratio < 1 ? cv::InterpolationFlags::INTER_AREA : */cv::InterpolationFlags::INTER_LINEAR);
    cv::copyMakeBorder(mat, mat, dh1, dh2, dw1, dw2, cv::BorderTypes::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114));
    long costTime;

    while (true)
    {
        system_clock::time_point t1 = system_clock::now();
        auto result = GetResult5_1(mat, costTime, 0.5, 0.5, SIZE_Width, SIZE_Height);
        system_clock::time_point t2 = system_clock::now();
        milliseconds ms = duration_cast<milliseconds>(t2 - t1);

        std::cout << "耗时：" << ms.count() << "ms" << std::endl;
    }
    

    std::cout << "Hello World!\n";
}



// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
