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
#include <string>
#include <std_msgs/String.h>
#include <sstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "../include/thermal.hpp"

int main(int argc, char **argv)
{
    vision_rescue::Thermal start(argc, argv);
    start.run();
}

namespace vision_rescue
{

    using namespace cv;
    using namespace std;
    using namespace ros;

    Thermal::Thermal(int argc, char **argv) : init_argc(argc),
                                              init_argv(argv),
                                              isRecv(false)
    {
        init();
    }

    Thermal::~Thermal()
    {
        if (ros::isStarted())
        {
            ros::shutdown(); // explicitly needed since we use ros::start();
            ros::waitForShutdown();
        }
    }

    bool Thermal::init()
    {
        ros::init(init_argc, init_argv, "QR_DETECT");
        if (!ros::master::check())
        {
            return false;
        }
        ros::start(); // explicitly needed since our nodehandle is going out of scope.
        ros::NodeHandle n;
        image_transport::ImageTransport img(n);
        img_thermal = n.advertise<sensor_msgs::Image>("img_thermal", 100);
        img_thermal_gray = n.advertise<sensor_msgs::Image>("img_thermal_gray", 100);
        n.getParam("/thermal/camera", param);
        ROS_INFO("Starting Rescue Vision With Camera : %s", param.c_str());
        img_sub = img.subscribe("/capra_thermal_cam/image_raw", 100, &Thermal::imageCallBack, this); /// camera/color/image_raw
        // Add your ros communications here.
        return true;
    }

    void Thermal::run()
    {
        ros::Rate loop_rate(10);
        while (ros::ok())
        {
            ros::spinOnce();
            loop_rate.sleep();
            if (isRecv == true)
            {
                update();
                img_thermal.publish(cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::BGR8, output_thermal).toImageMsg());
                img_thermal_gray.publish(cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::MONO8, gray_clone).toImageMsg());
            }
        }
    }

    void Thermal::imageCallBack(const sensor_msgs::ImageConstPtr &msg_img)
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

    void Thermal::update()
    {
        clone_mat = original->clone();
        cv::resize(clone_mat, clone_mat, cv::Size(400, 300), 0, 0, cv::INTER_CUBIC);
        output_thermal = clone_mat.clone();

        cvtColor(output_thermal, thermal_binary, CV_RGB2GRAY);
        threshold(thermal_binary, thermal_binary, 0, 255, THRESH_OTSU);
        // dilate(thermal_binary, thermal_binary,)

        // applyColorMap(clone_mat, output_thermal, COLORMAP_JET); // 색감 변환(JET, INFERNO...)
        // gray_clone = clone_mat.clone();
        // cvtColor(gray_clone, gray_clone, COLOR_BGR2GRAY);

        // minMaxLoc(clone_mat, &minVal, &maxVal, &minLoc, &maxLoc);
        // normalize(clone_mat, thermal_8bit, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        // cvtColor(thermal_8bit, thermal_8bit, cv::COLOR_GRAY2RGB);

        // minTemp=(1.8*(minVal/100.0)+32.0);
        // maxTemp=(1.8*(maxVal/100.0)+32.0);

        delete original;
        isRecv = false;
    }
}