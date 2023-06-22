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

#include "../include/qr_detect.hpp"

int main(int argc,char ** argv)
{
    vision_rescue::QR_DETECT start(argc, argv);
    start.run();
}

namespace vision_rescue
{

    using namespace cv;
    using namespace std;
    using namespace ros;

    QR_DETECT::QR_DETECT(int argc, char **argv) : 
    init_argc(argc),
    init_argv(argv),
    isRecv(false)
    {
        init();
    }

    QR_DETECT::~QR_DETECT()
    {
        if (ros::isStarted())
        {
            ros::shutdown(); // explicitly needed since we use ros::start();
            ros::waitForShutdown();
        }
    }

    bool QR_DETECT::init()
    {
        ros::init(init_argc, init_argv, "QR_DETECT");
        if (!ros::master::check())
        {
            return false;
        }
        ros::start(); // explicitly needed since our nodehandle is going out of scope.
        ros::NodeHandle n;
        image_transport::ImageTransport img(n);
        img_qr=n.advertise<sensor_msgs::Image>("img_qr", 100);
        img_sub = img.subscribe("/camera1/usb_cam/image_raw", 100, &QR_DETECT::imageCallBack, this); ///camera/color/image_raw
        // Add your ros communications here.
        return true;
    }

    void QR_DETECT::run()
    {
        ros::Rate loop_rate(10);
        while (ros::ok())
        {
            ros::spinOnce();
            loop_rate.sleep();
            if (isRecv == true)
            {
                update();
                img_qr.publish(cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::BGR8, output_qr).toImageMsg());
            }
        }
    }

    void QR_DETECT::imageCallBack(const sensor_msgs::ImageConstPtr &msg_img)
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

    void QR_DETECT::update()
    {
        clone_mat = original->clone();
        cv::resize(clone_mat, clone_mat, cv::Size(640, 360), 0, 0, cv::INTER_CUBIC);
        QRCodeDetector detector;
        cvtColor(clone_mat, gray_clone, COLOR_BGR2GRAY);
        
        output_qr = clone_mat.clone();

        if (detector.detect(gray_clone, points)) {
        polylines(output_qr, points, true, Scalar(0, 0, 0), 2);

        info = detector.decode(gray_clone, points);

        if (!info.empty()) {
            //polylines(output_qr, points, true, Scalar(255, 0, 0), 2);
            putText(output_qr, info, points[0], 0.5, 1, Scalar(0, 0, 255), 1, 8);
            cout<<info<<endl;
        }
    }   
        delete original;
        isRecv=false; 
    }
}