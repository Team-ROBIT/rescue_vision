#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <cmath>

#include <ros/ros.h>
#include <ros/network.h>
#include <ros/package.h>
#include <string>
#include <std_msgs/String.h>
#include <sstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "../include/hazmat.hpp"

constexpr float CONFIDENCE_THRESHOLD = 0.5; // 확률 경계값
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 15;

const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

int main(int argc, char **argv)
{
    vision_rescue::Hazmat start(argc, argv);
    start.run();
}

namespace vision_rescue
{

    using namespace cv;
    using namespace std;
    using namespace ros;

    Hazmat::Hazmat(int argc, char **argv) : init_argc(argc),
                                            init_argv(argv),
                                            isRecv(false)
    {
        std::string packagePath = ros::package::getPath("rescue_vision");
        cout << packagePath << endl;
        std::string dir = packagePath + "/yolo/";
        {
            std::ifstream class_file(dir + "classes.txt");
            if (!class_file)
            {
                std::cerr << "failed to open classes.txt\n";
            }

            std::string line;
            while (std::getline(class_file, line))
                class_names.push_back(line);
        }

        std::string modelConfiguration = dir + "yolov7_tiny_hazmat.cfg";
        std::string modelWeights = dir + "2023_06_12.weights";

        net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
        //     net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        //     net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        init();
    }

    Hazmat::~Hazmat()
    {
        if (ros::isStarted())
        {
            ros::shutdown(); // explicitly needed since we use ros::start();
            ros::waitForShutdown();
        }
    }

    bool Hazmat::init()
    {
        ros::init(init_argc, init_argv, "Hazmat");
        if (!ros::master::check())
        {
            return false;
        }
        ros::start(); // explicitly needed since our nodehandle is going out of scope.
        ros::NodeHandle n;
        image_transport::ImageTransport img(n);
        img_result = n.advertise<sensor_msgs::Image>("img_result", 100);
        n.getParam("/hazmat/camera", param);
        ROS_INFO("Starting Rescue Vision With Camera : %s", param.c_str());
        img_sub = img.subscribe(param, 100, &Hazmat::imageCallBack, this); /// camera/color/image_raw
        // Add your ros communications here.
        return true;
    }

    void Hazmat::run()
    {
        ros::Rate loop_rate(10);
        while (ros::ok())
        {
            ros::spinOnce();
            loop_rate.sleep();
            if (isRecv == true)
            {
                update();
                img_result.publish(cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::BGR8, frame).toImageMsg());
            }
        }
    }

    void Hazmat::imageCallBack(const sensor_msgs::ImageConstPtr &msg_img)
    {
        if (!isRecv)
        {
            original = new cv::Mat(cv_bridge::toCvCopy(msg_img, sensor_msgs::image_encodings::BGR8)->image);
            if (original != NULL)
            {
                isRecv = true;
            }
        }
    }

    void Hazmat::update()
    {
        clone_mat = original->clone();
        cv::resize(clone_mat, clone_mat, cv::Size(640, 360), 0, 0, cv::INTER_CUBIC);
        set_yolo();
        delete original;
        isRecv = false;
    }

    void Hazmat::set_yolo()
    {
        auto output_names = net.getUnconnectedOutLayersNames();
        frame = clone_mat.clone();
        std::vector<cv::Mat> detections;

        auto total_start = std::chrono::steady_clock::now();
        cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(416, 416), cv::Scalar(), true, false, CV_32F);
        net.setInput(blob);

        auto dnn_start = std::chrono::steady_clock::now();
        net.forward(detections, output_names);
        auto dnn_end = std::chrono::steady_clock::now();

        std::vector<int> indices[NUM_CLASSES];
        std::vector<cv::Rect> boxes[NUM_CLASSES];
        std::vector<float> scores[NUM_CLASSES];

        for (auto &output : detections)
        {
            const auto num_boxes = output.rows;
            for (int i = 0; i < num_boxes; i++)
            {
                auto x = output.at<float>(i, 0) * frame.cols; // 중심 x
                auto y = output.at<float>(i, 1) * frame.rows; // 중심 y
                auto width = output.at<float>(i, 2) * frame.cols;
                auto height = output.at<float>(i, 3) * frame.rows;
                cv::Rect rect(x - width / 2, y - height / 2, width, height);

                for (int c = 0; c < NUM_CLASSES; c++)
                {
                    auto confidence = *output.ptr<float>(i, 5 + c);
                    if (confidence >= CONFIDENCE_THRESHOLD)
                    {
                        boxes[c].push_back(rect);
                        scores[c].push_back(confidence);
                    }
                }
            }
        }

        for (int c = 0; c < NUM_CLASSES; c++)
            cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

        for (int c = 0; c < NUM_CLASSES; c++)
        {
            for (size_t i = 0; i < indices[c].size(); ++i)
            {
                const auto color = colors[c % NUM_COLORS];
                auto idx = indices[c][i];
                const auto &rect = boxes[c][idx];

                // Check for overlapping boxes of the same class
                isOverlapping = false;
                if (indices[c].size() != 0)
                {
                    for (size_t j = 0; j < indices[c].size(); ++j)
                    {
                        if (j != i)
                        {
                            auto idx2 = indices[c][j];
                            const auto &rect2 = boxes[c][idx2];
                            if (isRectOverlapping(rect, rect2))
                            {
                                // If there is an overlapping box, check the sizes
                                if (rect2.area() < rect.area())
                                {
                                    // If the other box is smaller, mark it as overlapping and break
                                    isOverlapping = true;
                                    break;
                                }
                                else
                                {
                                    // If the current box is smaller, skip it
                                    continue;
                                }
                            }
                        }
                    }
                }

                // Draw the box only if it is not overlapping with a smaller box
                if (!isOverlapping)
                {
                    cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

                    std::ostringstream label_ss;
                    label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
                    auto label = label_ss.str();

                    int baseline;
                    auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                    cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
                    cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
                }
            }
        }

        auto total_end = std::chrono::steady_clock::now();

        float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
        float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
        std::ostringstream stats_ss;
        stats_ss << std::fixed << std::setprecision(2);
        stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
        auto stats = stats_ss.str();

        int baseline;
        auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        cv::rectangle(frame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));
    }

    bool Hazmat::isRectOverlapping(const cv::Rect &rect1, const cv::Rect &rect2)
    {
        int x1 = std::max(rect1.x, rect2.x);
        int y1 = std::max(rect1.y, rect2.y);
        int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
        int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

        // If the intersection area is positive, the rectangles overlap
        if (x1 < x2 && y1 < y2)
            return true;
        else
            return false;
    }

}