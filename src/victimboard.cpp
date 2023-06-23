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

#include "../include/victimboard.hpp"

int main(int argc, char **argv)
{
    vision_rescue::Victimboard start(argc, argv);
    start.run();
}

namespace vision_rescue
{

    using namespace cv;
    using namespace std;
    using namespace ros;

    Victimboard::Victimboard(int argc, char **argv) : init_argc(argc),
                                                  init_argv(argv),
                                                  isRecv(false)
    {
        init();
    }

    Victimboard::~Victimboard()
    {
        if (ros::isStarted())
        {
            ros::shutdown(); // explicitly needed since we use ros::start();
            ros::waitForShutdown();
        }
    }

    bool Victimboard::init()
    {
        ros::init(init_argc, init_argv, "Victimboard");
        if (!ros::master::check())
        {
            return false;
        }
        ros::start(); // explicitly needed since our nodehandle is going out of scope.
        ros::NodeHandle n;
        image_transport::ImageTransport img(n);
        img_ad = n.advertise<sensor_msgs::Image>("img_ad", 100);
        //n.getParam("/qr_detect/camera", param);
        //ROS_INFO("Starting Rescue Vision With Camera : %s", param.c_str());
        img_sub = img.subscribe(param, 100, &Victimboard::imageCallBack, this); /// camera/color/image_raw
        // Add your ros communications here.
        return true;
    }

    void Victimboard::run()
    {
        ros::Rate loop_rate(10);
        while (ros::ok())
        {
            ros::spinOnce();
            loop_rate.sleep();
            if (isRecv == true)
            {
                update();
                img_ad.publish(cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::MONO8, Image_to_Binary_adaptive).toImageMsg());
            }
        }
    }

    void Victimboard::imageCallBack(const sensor_msgs::ImageConstPtr &msg_img)
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

    void Victimboard::update()
    {
        clone_mat = original->clone();
        cv::resize(clone_mat, clone_mat, cv::Size(640, 360), 0, 0, cv::INTER_CUBIC);
        Image_to_Binary_OTSU=clone_mat.clone();
        GaussianBlur(Image_to_Binary_OTSU, Image_to_Binary_OTSU, Size(15,15), 2.0);
        cvtColor(Image_to_Binary_OTSU, Image_to_Binary_OTSU, CV_RGB2GRAY);
        threshold(Image_to_Binary_OTSU, Image_to_Binary_OTSU, 0, 255, THRESH_OTSU);
        dilate(Image_to_Binary_OTSU, Image_to_Binary_OTSU, mask, Point(-1, -1), 2);

        Image_to_Binary_adaptive = clone_mat.clone();
        Image_to_Binary_adaptive.convertTo(Image_to_Binary_adaptive, -1, 1.0, 70); //brightness control before threshold
        GaussianBlur(Image_to_Binary_adaptive, Image_to_Binary_adaptive, Size(11, 11), 2.0);
        cvtColor(Image_to_Binary_adaptive, Image_to_Binary_adaptive, CV_RGB2GRAY);
        adaptiveThreshold(Image_to_Binary_adaptive, Image_to_Binary_adaptive, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 85, 2);//85
            

        output_qr = clone_mat.clone();
        delete original;
        isRecv = false;
    }
}