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
        img_divide = n.advertise<sensor_msgs::Image>("img_divide", 100);
        n.getParam("/victimboard/camera", param);
        ROS_INFO("Starting Rescue Vision With Camera : %s", param.c_str());
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
                img_divide.publish(cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::BGR8, Captured_Image_RGB).toImageMsg());
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
        Image_to_Binary_OTSU = clone_mat.clone();
        GaussianBlur(Image_to_Binary_OTSU, Image_to_Binary_OTSU, Size(15, 15), 2.0);
        cvtColor(Image_to_Binary_OTSU, Image_to_Binary_OTSU, CV_RGB2GRAY);
        threshold(Image_to_Binary_OTSU, Image_to_Binary_OTSU, 0, 255, THRESH_OTSU);
        dilate(Image_to_Binary_OTSU, Image_to_Binary_OTSU, mask, Point(-1, -1), 2);

        Image_to_Binary_adaptive = clone_mat.clone();
        Image_to_Binary_adaptive.convertTo(Image_to_Binary_adaptive, -1, 1.0, 70); // brightness control before threshold
        GaussianBlur(Image_to_Binary_adaptive, Image_to_Binary_adaptive, Size(11, 11), 2.0);
        cvtColor(Image_to_Binary_adaptive, Image_to_Binary_adaptive, CV_RGB2GRAY);
        adaptiveThreshold(Image_to_Binary_adaptive, Image_to_Binary_adaptive, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 85, 2); // 85

        Captured_Image_RGB = clone_mat.clone();
        Captured_Image_to_Binary = Image_to_Binary_OTSU;
        //----------------------------------------------------------------------------------------------------------------------------

        labeling(Captured_Image_to_Binary, image_x, image_y, image_width_divided3, image_height_divided3);

        roi = Mat::zeros(Captured_Image_to_Binary.rows, Captured_Image_to_Binary.cols, CV_8UC1);
        adaptive_capture = Image_to_Binary_adaptive;

        cout << "before" << endl;

        // if (!(((image_x - 10 < 0)) || ((image_x - 10 + image_width_divided3 + 20) > 640) || ((image_y - 10) < 0) || ((image_y - 10 + image_height_divided3 + 20) > 360)))
        //{
        cout << "after" << endl;
        divide_box();
        //}

        delete original;
        isRecv = false;
    }

    void Victimboard::divide_box()
    {
        rectangle(roi, Rect(image_x - 10, image_y - 10, image_width_divided3 + 20, image_height_divided3 + 20), Scalar::all(255), -1, LINE_8, 0);
        bitwise_and(adaptive_capture, roi, result, noArray());
        dilate(result, result, mask, Point(-1, -1), 2);

        labeling(result, image_x, image_y, image_width_divided3, image_height_divided3);
        image_width_divided3 = image_width_divided3 / 3;
        image_height_divided3 = image_height_divided3 / 3;

        rectangle(Captured_Image_RGB, Rect(image_x, image_y, image_width_divided3 * 3, image_height_divided3 * 3), cv::Scalar(0, 0, 255), 2);                                              // draw rect
        line(Captured_Image_RGB, Point(image_x + image_width_divided3, image_y), Point(image_x + image_width_divided3, image_y + image_height_divided3 * 3), cv::Scalar(0, 0, 255), 2, 8); // divide 9
        line(Captured_Image_RGB, Point(image_x + image_width_divided3 * 2, image_y), Point(image_x + image_width_divided3 * 2, image_y + image_height_divided3 * 3), cv::Scalar(0, 0, 255), 2, 8);
        line(Captured_Image_RGB, Point(image_x, image_y + image_height_divided3), Point(image_x + image_width_divided3 * 3, image_y + image_height_divided3), cv::Scalar(0, 0, 255), 2, 8);
        line(Captured_Image_RGB, Point(image_x, image_y + image_height_divided3 * 2), Point(image_x + image_width_divided3 * 3, image_y + image_height_divided3 * 2), cv::Scalar(0, 0, 255), 2, 8);

        divided_Image_data[0].position = Rect(image_x, image_y, image_width_divided3, image_height_divided3);
        divided_Image_data[1].position = Rect(image_x + image_width_divided3, image_y, image_width_divided3, image_height_divided3);
        divided_Image_data[2].position = Rect(image_x + image_width_divided3 * 2, image_y, image_width_divided3, image_height_divided3);
        divided_Image_data[3].position = Rect(image_x, image_y + image_height_divided3, image_width_divided3, image_height_divided3);
        divided_Image_data[4].position = Rect(image_x + image_width_divided3 * 2, image_y + image_height_divided3, image_width_divided3, image_height_divided3);
        divided_Image_data[5].position = Rect(image_x, image_y + image_height_divided3 * 2, image_width_divided3, image_height_divided3);
        divided_Image_data[6].position = Rect(image_x + image_width_divided3, image_y + image_height_divided3 * 2, image_width_divided3, image_height_divided3);
        divided_Image_data[7].position = Rect(image_x + image_width_divided3 * 2, image_y + image_height_divided3 * 2, image_width_divided3, image_height_divided3);
        // 0 1 2
        // 3   4
        // 5 6 7
    }

    void Victimboard::labeling(const Mat &input_img, int &_x, int &_y, int &_width, int &_height)
    {
        int num_labels = connectedComponentsWithStats(input_img, img_labeling, stats, centroids, 8, CV_32S); // labeling

        int max_area = 0;
        for (int i = 1; i < num_labels; i++)
        {
            int area = stats.at<int>(i, CC_STAT_AREA);
            int left = stats.at<int>(i, CC_STAT_LEFT);
            int top = stats.at<int>(i, CC_STAT_TOP);
            int width = stats.at<int>(i, CC_STAT_WIDTH);
            int height = stats.at<int>(i, CC_STAT_HEIGHT);

            if (area > max_area) // find the largest of several labels
            {
                max_area = area;

                _x = left; // save rect data of max label
                _y = top;
                _width = width;
                _height = height;
            }
        }
    }
}