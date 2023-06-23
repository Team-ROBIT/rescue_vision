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

#include <ros/ros.h>
#include <ros/network.h>
#include <string>
#include <std_msgs/String.h>
#include <sstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "../include/findc.hpp"

int main(int argc, char **argv)
{
    vision_rescue::Findc start(argc, argv);
    start.run();
}

namespace vision_rescue
{

    using namespace cv;
    using namespace std;
    using namespace ros;

    Findc::Findc(int argc, char **argv) : init_argc(argc),
                                          init_argv(argv),
                                          isRecv(false)
    {
        init();
    }

    Findc::~Findc()
    {
        if (ros::isStarted())
        {
            ros::shutdown(); // explicitly needed since we use ros::start();
            ros::waitForShutdown();
        }
    }

    void Findc::update()
    {

        clone_mat = original->clone();
        cv::resize(clone_mat, clone_mat, cv::Size(640, 360), 0, 0, cv::INTER_CUBIC);
        gray_clone = clone_mat.clone();
        cvtColor(gray_clone, gray_clone, COLOR_BGR2GRAY);
        threshold(gray_clone, clone_binary, 80, 255, cv::THRESH_BINARY);
        HoughCircles(gray_clone, circles, HOUGH_GRADIENT, 1, 200, 200, 10, range_radius_small, range_radius_big); // 100->50
        for (size_t i = 0; i < circles.size(); i++)
        {
            c = circles[i];
            Point center(c[0], c[1]);
            radius = c[2];

            // 작은 원 반지름 : 큰 원 반지름 -> 1.2 : 2.8 -> 2.8 / 1.2 = 2.3
            if (!(((c[1] - 5.3 * radius) < 0) || ((c[1] + 5.3 * radius) > 360) || ((c[0] - 5.3 * radius) < 0) || ((c[0] + 5.3 * radius) > 640)))
            { // 300, 400
                choose_circle(center, radius);
                if (find_ok == true)
                    catch_c(center, radius);
                circle(clone_mat, center, radius, Scalar(0, 255, 0), 2); // 하나의 원
                circle(clone_mat, center, 1, Scalar(255, 0, 0), 3);
            }
        }

        if (!circles.empty())
        {
            // cout<<degrees<<endl;
        }

        all_clear();
    }

    void Findc::choose_circle(Point center, int radius)
    {
        /*
        int start_y=c[1] - radius;
        int end_y=c[1] + radius;
        int start_x=c[0] - radius;
        int end_x=c[0] + radius;
        */

        in_cup_mat = clone_mat(Range(c[1] - radius, c[1] + radius), Range(c[0] - radius, c[0] + radius));
        cv::resize(in_cup_mat, in_cup_mat, cv::Size(300, 300), 0, 0, cv::INTER_CUBIC);
        cvtColor(in_cup_mat, in_cup_gray, cv::COLOR_BGR2GRAY);
        threshold(in_cup_gray, in_cup_binary, 80, 255, cv::THRESH_BINARY);
        in_cup_binary = ~in_cup_binary;
        find_ok = check_black(in_cup_binary);
    }

    void Findc::catch_c(Point center, int radius)
    {

        int big_radius = 2.3 * radius;

        expand_cup_mat = clone_mat(Range(c[1] - big_radius, c[1] + big_radius), Range(c[0] - big_radius, c[0] + big_radius));
        cv::resize(expand_cup_mat, expand_cup_mat, cv::Size(300, 300), 0, 0, cv::INTER_CUBIC);
        cvtColor(expand_cup_mat, expand_cup_gray, cv::COLOR_BGR2GRAY);
        threshold(expand_cup_gray, expand_cup_binary, 80, 255, cv::THRESH_BINARY);
        expand_cup_binary = ~expand_cup_binary;

        int big_radius2 = 5.3 * radius;

        expand_cup_mat2 = clone_mat(Range(c[1] - big_radius2, c[1] + big_radius2), Range(c[0] - big_radius2, c[0] + big_radius2));
        cv::resize(expand_cup_mat2, expand_cup_mat2, cv::Size(300, 300), 0, 0, cv::INTER_CUBIC);
        cvtColor(expand_cup_mat2, expand_cup_gray2, cv::COLOR_BGR2GRAY);
        threshold(expand_cup_gray2, expand_cup_binary2, 70, 255, cv::THRESH_BINARY);
        expand_cup_binary2 = ~expand_cup_binary2;

        //
        first_ring = check_black(expand_cup_binary);
        second_ring = check_black(expand_cup_binary2);

        ok_cup_mat = in_cup_mat.clone();
        // Canny(in_cup_gray, in_cup_canny, 400, 50);

        if (second_ring == true)
        {
            // 세번째 원이다
            range_radius_small++;
        }
        else if (first_ring == true)
        {
            // 두번째 원이다
            cout << "check!" << endl;
            detect_way();
        }
        else
        {
            // 가장 큰 원이다
            range_radius_big--;
        }

        // find_contour();
    }

    void Findc::remove_text()
    {
    }

    void Findc::detect_way()
    {
        last_binary = in_cup_binary.clone();

        double sumAngles2 = 0.0;
        int count2 = 0;
        int radius2 = 125; // 원의 반지름
        double temp_radian2 = 0;
        double angleRadians2 = 0;

        for (int y = 0; y < expand_cup_binary.rows; y++)
        {
            for (int x = 0; x < expand_cup_binary.cols; x++)
            {
                if (expand_cup_binary.at<uchar>(y, x) == 0)
                {
                    // 중심 좌표로부터의 거리 계산
                    double distance = std::sqrt(std::pow(x - 150, 2) + std::pow(y - 150, 2));

                    if (std::abs(distance - radius2) < 1.0) // 거리가 125인 지점 판단
                    {

                        // circle(ok_cup_mat, Point(x,y), 5, Scalar(0 ,0, 255), 1, -1);
                        angleRadians2 = std::atan2(y - 150, x - 150) * 180.0 / CV_PI;
                        sumAngles2 += angleRadians2;
                        count2++;
                        if (abs(temp_radian2 - angleRadians2) > 180)
                        {
                            sumAngles2 = 180 * count2;
                            break;
                        }
                        temp_radian2 = angleRadians2;
                    }
                }
            }
            if (abs(temp_radian2 - angleRadians2) > 180)
                break;
        }

        double averageAngle2 = 0;
        if (count2 > 0)
        {
            averageAngle2 = -(sumAngles2 / count2);
            cout << "1: " << averageAngle2 << endl;
        }

        //-------------------------------------------------------big----------------------------------------------------

        range_radius_small = 10;
        range_radius_big = 120;

        double sumAngles = 0.0;
        int count = 0;
        int radius = 125; // 원의 반지름
        double temp_radian = 0;
        double angleRadians = 0;

        for (int y = 0; y < last_binary.rows; y++)
        {
            for (int x = 0; x < last_binary.cols; x++)
            {
                if (last_binary.at<uchar>(y, x) == 0)
                {
                    // 중심 좌표로부터의 거리 계산
                    double distance = std::sqrt(std::pow(x - 150, 2) + std::pow(y - 150, 2));

                    if (std::abs(distance - radius) < 1.0) // 거리가 125인 지점 판단
                    {

                        circle(ok_cup_mat, Point(x, y), 5, Scalar(0, 0, 255), 1, -1);
                        angleRadians = std::atan2(y - 150, x - 150) * 180.0 / CV_PI;
                        sumAngles += angleRadians;
                        count++;
                        if (abs(temp_radian - angleRadians) > 180)
                        {
                            sumAngles = 180 * count;
                            break;
                        }
                        temp_radian = angleRadians;
                    }
                }
            }
            if (abs(temp_radian - angleRadians) > 180)
                break;
        }

        if (count > 0)
        {
            double averageAngle = -(sumAngles / count);
            // double averageAngle = std::fmod(-(sumAngles / count) + 360.0, 360.0);
            cout << averageAngle << endl;

            int averageAngle_calc = 0;

            averageAngle_calc = (90 - averageAngle2) + averageAngle;
            if (averageAngle_calc > 180)
                averageAngle_calc -= 360;
            int averageAngle_i = (averageAngle_calc + 22.5 * ((averageAngle_calc > 0) ? 1 : -1)) / 45.0;
            cout << averageAngle_i << endl;
            switch (averageAngle_i)
            {
            case -4:
                cout << "left" << endl;
                break;
            case -3:
                cout << "left_down" << endl;
                break;
            case -2:
                cout << "down" << endl;
                break;
            case -1:
                cout << "right_down" << endl;
                break;
            case 0:
                cout << "right" << endl;
                break;
            case 1:
                cout << "right_up" << endl;
                break;
            case 2:
                cout << "up" << endl;
                break;
            case 3:
                cout << "left_up" << endl;
                break;
            case 4:
                cout << "left" << endl;
                break;
            }
        }
    }

    bool Findc::check_black(const Mat &binary_mat)
    {
        int cnt = 0;

        bool left = binary_mat.at<uchar>(150, 20);
        bool up = binary_mat.at<uchar>(20, 150);
        bool down = binary_mat.at<uchar>(280, 150);
        bool right = binary_mat.at<uchar>(150, 280);
        cout << up << " " << left << " " << right << " " << down << endl;

        if (up == 1)
            cnt++;
        if (left == 1)
            cnt++;
        if (right == 1)
            cnt++;
        if (down == 1)
            cnt++;

        if (cnt >= 2)
            return true;
        else
            return false;

        // if()
    }

    void Findc::all_clear()
    {
        circles.clear();
        delete original;
        isRecv = false;
    }

    bool Findc::init()
    {
        ros::init(init_argc, init_argv, "findc");
        if (!ros::master::check())
        {
            return false;
        }
        ros::start(); // explicitly needed since our nodehandle is going out of scope.
        ros::NodeHandle n;
        img_tr = n.advertise<sensor_msgs::Image>("img_tr", 100);
        img_cup = n.advertise<sensor_msgs::Image>("img_cup", 100);
        img_cup_binary = n.advertise<sensor_msgs::Image>("img_cup_binary", 100);
        img_expand_binary = n.advertise<sensor_msgs::Image>("img_expand_binary", 100);
        img_expand_binary2 = n.advertise<sensor_msgs::Image>("img_expand_binary2", 100);
        image_transport::ImageTransport img(n);
        n.getParam("/findc/camera", param);
        ROS_INFO("Starting Rescue Vision With Camera : %s", param.c_str());
        img_sub = img.subscribe(param, 100, &Findc::imageCallBack, this);
        // img_sub2 = img.subscribe("/capra_thermal/image_raw", 100, &Findc::imageCallBack, this);
        //  Add your ros communications here.
        return true;
    }

    void Findc::run()
    {
        ros::Rate loop_rate(10);
        while (ros::ok())
        {
            ros::spinOnce();
            loop_rate.sleep();
            if (isRecv == true)
            {
                update();
                img_tr.publish(cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::BGR8, clone_mat).toImageMsg());
                img_cup.publish(cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::BGR8, ok_cup_mat).toImageMsg());
                // img_cup_expand.publish(cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::BGR8, expand_cup_mat2).toImageMsg());
                img_cup_binary.publish(cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::MONO8, last_binary).toImageMsg());
                img_expand_binary.publish(cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::MONO8, expand_cup_binary).toImageMsg());
                img_expand_binary2.publish(cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::MONO8, expand_cup_binary2).toImageMsg());
            }
        }
    }

    void Findc::imageCallBack(const sensor_msgs::ImageConstPtr &msg_img)
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

    void Findc::find_contour()
    {
        int x = c[0] - 5 * radius;
        int y = c[1] - 5 * radius;
        int width = 10 * radius;
        int height = 10 * radius;
        findContours(clone_binary(Rect(x, y, width, height)), contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        threshold(clone_binary, contour_lane, 100, 255, THRESH_MASK);

        for (int i = 0; i < contours.size(); i++)
        {
            drawContours(clone_mat, contours, i, Scalar(0, 255, 0), 1);
            drawContours(contour_lane, contours, i, Scalar::all(255), 1);
        }
    }
}
/*
void Findc::find_contour()
{
    findContours(expand_cup_binary2, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    //threshold(in_cup_binary, contour_lane, 100, 255, THRESH_MASK);

    for (int i = 0; i < contours.size(); i++)
    {
        drawContours(expand_cup_mat2, contours, i, Scalar(0, 255, 0), 1);
        //drawContours(contour_lane, contours, i, Scalar::all(255), 1);
    }
}

}
*/