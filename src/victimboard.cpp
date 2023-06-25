#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <queue>
#include <sstream>

#include <ros/network.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sstream>
#include <std_msgs/String.h>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../include/victimboard.hpp"

#include <opencv2/xfeatures2d.hpp>

#define PI 3.141592

constexpr float CONFIDENCE_THRESHOLD = 0.5; // 확률 경계값
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 15;

const cv::Scalar colors[] = {
    {0, 255, 255}, {255, 255, 0}, {0, 255, 0}, {255, 0, 0}};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

int main(int argc, char **argv)
{
    vision_rescue::Victimboard start(argc, argv);
    start.run();
}

namespace vision_rescue
{
    bool check_black(const Mat &binary_mat)
    {
        int cnt = 0;

        bool left = binary_mat.at<uchar>(150, 20);
        bool up = binary_mat.at<uchar>(20, 150);
        bool down = binary_mat.at<uchar>(280, 150);
        bool right = binary_mat.at<uchar>(150, 280);
        // cout << up << " " << left << " " << right << " " << down << endl;

        if (up == 1)
            cnt++;
        if (left == 1)
            cnt++;
        if (right == 1)
            cnt++;
        if (down == 1)
            cnt++;

        if (cnt >= 3)
            return true;
        else
            return false;

        // if()
    }
    using namespace cv;
    using namespace std;
    using namespace ros;

    Victimboard::Victimboard(int argc, char **argv)
        : init_argc(argc), init_argv(argv), isRecv(false), isRecv_thermal(false)
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
        img_result = n.advertise<sensor_msgs::Image>("img_result", 100);
        img_result_thermal = n.advertise<sensor_msgs::Image>("img_result_thermal", 100);
        img_binary_vt = n.advertise<sensor_msgs::Image>("img_binary_vt", 100);
        findc_pub = n.advertise<sensor_msgs::Image>("findc_vt", 100);
        down_c = n.advertise<sensor_msgs::Image>("down_c", 100);
        up_c = n.advertise<sensor_msgs::Image>("up_c", 100);
        img_binary_thermal = n.advertise<sensor_msgs::Image>("img_binary_thermal", 100);

        n.getParam("/victimboard/camera", param);
        ROS_INFO("Starting Rescue Vision With Camera : %s", param.c_str());

        img_sub = img.subscribe(param, 100, &Victimboard::imageCallBack,
                                this); /// camera/color/image_raw
        img_sub_thermal = img.subscribe("/capra_thermal_cam/image_raw", 100,
                                        &Victimboard::imageCallBack_thermal, this);
        movement_count = 0;
        // count_Movement_find_circle = 0;
        movement_Find_center_img.release();

        count__Rotation_Direction = 0;
        count_Movement_find_circle = 0;
        Rotation_Direction[0] = 0;
        Rotation_Direction[1] = 0;
        start_motion = false;
        //  Add your ros communications here.
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
                if (isRecv_thermal == true)
                {
                    set_thermal();
                    img_result_thermal.publish(
                        cv_bridge::CvImage(std_msgs::Header(),
                                           sensor_msgs::image_encodings::BGR8,
                                           thermal_mat)
                            .toImageMsg());
                }
                update();
                img_ad.publish(cv_bridge::CvImage(std_msgs::Header(),
                                                  sensor_msgs::image_encodings::MONO8,
                                                  Image_to_Binary_adaptive)
                                   .toImageMsg());
                img_result.publish(cv_bridge::CvImage(std_msgs::Header(),
                                                      sensor_msgs::image_encodings::BGR8,
                                                      Captured_Image_RGB) // Captured_Image_RGB
                                       .toImageMsg());
            }
        }
    }

    void Victimboard::imageCallBack(const sensor_msgs::ImageConstPtr &msg_img)
    {
        if (!isRecv)
        {
            original = new cv::Mat(
                cv_bridge::toCvCopy(msg_img, sensor_msgs::image_encodings::BGR8)
                    ->image);
            if (original != NULL)
            {
                isRecv = true;
            }
        }
    }

    void Victimboard::imageCallBack_thermal(
        const sensor_msgs::ImageConstPtr &msg_img)
    {
        if (!isRecv_thermal)
        {
            original_thermal = new cv::Mat(
                cv_bridge::toCvCopy(msg_img, sensor_msgs::image_encodings::BGR8)
                    ->image);
            if (original_thermal != NULL)
            {
                isRecv_thermal = true;
            }
        }
    }

    void Victimboard::set_thermal()
    {
        thermal_mat = original_thermal->clone();
        applyColorMap(thermal_mat, thermal_mat, COLORMAP_INFERNO);
        CLAHE(thermal_mat);
        cv::resize(thermal_mat, thermal_mat, cv::Size(400, 300), 0, 0, cv::INTER_CUBIC);

        /*Mat gray_thermal;
        Mat binary_thermal;
        cvtColor(clone_thermal_mat, gray_thermal, CV_RGB2GRAY);
        threshold(gray_thermal, binary_thermal, 100, 255, THRESH_OTSU);
        dilate(binary_thermal, binary_thermal, mask, Point(-1, -1), 5);
        img_binary_thermal.publish(
            cv_bridge::CvImage(std_msgs::Header(),
                               sensor_msgs::image_encodings::MONO8,
                               gray_thermal)
                .toImageMsg());*/

        delete original_thermal;
        isRecv_thermal = false;
    }

    void Victimboard::update()
    {
        clone_mat = original->clone();
        cv::resize(clone_mat, clone_mat, cv::Size(640, 360), 0, 0, cv::INTER_CUBIC);

        if (start_motion)
        {
            Divided_Image__Rotation_Direction[0] = clone_mat(divided_Image_data[save_image_position__Rotation_Direction[0]].position).clone();
            Divided_Image__Rotation_Direction[1] = clone_mat(divided_Image_data[save_image_position__Rotation_Direction[1]].position).clone();

            resize(Divided_Image__Rotation_Direction[0], Divided_Image__Rotation_Direction[0], Size(300, 300), 0, 0, CV_INTER_LINEAR);
            resize(Divided_Image__Rotation_Direction[1], Divided_Image__Rotation_Direction[1], Size(300, 300), 0, 0, CV_INTER_LINEAR);

            if (movement_Find_center_img.empty())
            {
                img_cvtcolor_gray(Divided_Image__Rotation_Direction[count__Rotation_Direction], movement_Find_center_img);
                movement_Find_center_img.setTo(0); //.setTo(0) -> set all pixel 0
                // Clean Mat for points of binarization, Do not initialize while running
            }
            img_Detect_movement(Divided_Image__Rotation_Direction[count__Rotation_Direction]);
        }

        Image_to_Binary_OTSU = clone_mat.clone();
        GaussianBlur(Image_to_Binary_OTSU, Image_to_Binary_OTSU, Size(15, 15), 2.0);
        cvtColor(Image_to_Binary_OTSU, Image_to_Binary_OTSU, CV_RGB2GRAY);
        threshold(Image_to_Binary_OTSU, Image_to_Binary_OTSU, 0, 255, THRESH_OTSU);
        dilate(Image_to_Binary_OTSU, Image_to_Binary_OTSU, mask, Point(-1, -1), 2);

        Image_to_Binary_adaptive = clone_mat.clone();
        Image_to_Binary_adaptive.convertTo(Image_to_Binary_adaptive, -1, 1.0,
                                           70); // brightness control before threshold
        GaussianBlur(Image_to_Binary_adaptive, Image_to_Binary_adaptive, Size(11, 11),
                     2.0);
        cvtColor(Image_to_Binary_adaptive, Image_to_Binary_adaptive, CV_RGB2GRAY);
        adaptiveThreshold(Image_to_Binary_adaptive, Image_to_Binary_adaptive, 255,
                          ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 85, 2); // 85

        Captured_Image_RGB = clone_mat.clone();
        Captured_Image_to_Binary = Image_to_Binary_OTSU;
        //----------------------------------------------------------------------------------------------------------------------------

        // if (!(((image_x - 10 < 0)) || ((image_x - 10 + image_width_divided3 + 20) >
        // 640) || ((image_y - 10) < 0) || ((image_y - 10 + image_height_divided3 +
        // 20) > 360)))
        divide_box();
        set_yolo();
        detect_location();

        if (exist_c)
        {

            up_c.publish(
                cv_bridge::CvImage(std_msgs::Header(),
                                   sensor_msgs::image_encodings::BGR8,
                                   C_up)
                    .toImageMsg());
            down_c.publish(
                cv_bridge::CvImage(std_msgs::Header(),
                                   sensor_msgs::image_encodings::BGR8,
                                   C_down)
                    .toImageMsg());
            exist_c = false;
        }

        delete original;
        isRecv = false;
    }

    void Victimboard::img_cvtcolor_gray(Mat &input, Mat &output)
    {
        cvtColor(input, output, COLOR_BGR2GRAY);
    }

    void Victimboard::img_Detect_movement(Mat &input_img)
    {

        Mat gray_img;
        Mat binary_img;

        Point MovementBox_center;
        bool calculate_flag = false;

        CLAHE(input_img);
        cvtColor(input_img, gray_img, COLOR_BGR2GRAY);

        int nThreshold_Value = 70;

        threshold(gray_img, binary_img, nThreshold_Value, 255, THRESH_BINARY_INV); // reverse binary
        binary_img = region_of_interest(binary_img);

        RobitLabeling BOX(binary_img, 0, 4);
        BOX.doLabeling();
        BOX.sortingRecBlobs(); // labeling -> in [0], largest area

        if (BOX.m_nBlobs > 0) // draw dot(dot is MovementBox_centor) for HoughCircles
        {
            cv::rectangle(Divided_Image__Rotation_Direction[count__Rotation_Direction], BOX.m_recBlobs[0], Scalar(0, 0, 255), 5, 8);
            MovementBox_center = Point(BOX.m_recBlobs[0].x + (BOX.m_recBlobs[0].width / 2), BOX.m_recBlobs[0].y + (BOX.m_recBlobs[0].height / 2));
            circle(movement_Find_center_img, Point(MovementBox_center.x, MovementBox_center.y), 1, Scalar(255, 255, 255), -1);
        }
        //---============================
        movement_count += 1;

        Center = Point(movement_Find_center_img.rows / 2, movement_Find_center_img.cols / 2);

        if (movement_count == 4)
            Vec1 = Point(MovementBox_center.x - Center.x, MovementBox_center.y - Center.y);
        else if (movement_count > 15)
        {
            Vec2 = Point(MovementBox_center.x - Center.x, MovementBox_center.y - Center.y);

            double A = pow(Vec1.x, 2) + pow(Vec1.y, 2);
            double B = pow(Vec2.x, 2) + pow(Vec2.y, 2);
            double A_1 = sqrt(A);
            double B_1 = sqrt(B);

            double A_2 = (Vec1.x) * (Vec2.y) - (Vec2.x) * (Vec1.y);
            double B_2 = A_1 * B_1;

            double degree = (double)(asin(A_2 / B_2));
            Rotation_Direction[count__Rotation_Direction] = degree;

            //        cout << "A  :  " << A << "   B  :  " << B << endl;
            //        cout << "A_1  :  " << A_1<<"   B_1  :  " << B_1 << endl;
            //        cout << "degree   :   " << degree << endl;
            int p_x = divided_Image_data[save_image_position__Rotation_Direction[count__Rotation_Direction]].position.x;
            int p_y = divided_Image_data[save_image_position__Rotation_Direction[count__Rotation_Direction]].position.y;

            if (degree > 0 && degree > 0.4)
            {
                cout << endl
                     << endl
                     << save_image_position__Rotation_Direction[count__Rotation_Direction] << ": " << degree << ": CW" << endl;
                putText(Captured_Image_RGB, "CW", Point(p_x, p_y), 0.5, 1, Scalar(0, 0, 0), 2, 8);
            }
            else if (degree < 0 && degree < -0.4)
            {
                cout << endl
                     << endl
                     << save_image_position__Rotation_Direction[count__Rotation_Direction] << ": " << degree << ": CCW" << endl;
                putText(Captured_Image_RGB, "CCW", Point(p_x, p_y), 0.5, 1, Scalar(0, 0, 0), 2, 8);
            }
            else
            {
                cout << endl
                     << endl
                     << save_image_position__Rotation_Direction[count__Rotation_Direction] << ": " << degree << ": STOP" << endl;
                putText(Captured_Image_RGB, "STOP", Point(p_x, p_y), 0.5, 1, Scalar(0, 0, 0), 2, 8);
            }

            count__Rotation_Direction++;

            if (count__Rotation_Direction == 2)
            {
                start_motion = false;
                count__Rotation_Direction = 0;
                //            text_msg.motion1 = Write_motionData(Rotation_Direction[0]);
                //            text_msg.motion2 = Write_motionData(Rotation_Direction[1]);
            }

            movement_Find_center_img.release(); // initalize after finishing img_Detect_movement
            movement_count = 0;
        }

        img_binary_vt.publish(
            cv_bridge::CvImage(std_msgs::Header(),
                               sensor_msgs::image_encodings::MONO8,
                               binary_img)
                .toImageMsg());
    }

    Mat Victimboard::region_of_interest(Mat input_img)
    {
        Mat img_mask = Mat::zeros(input_img.rows, input_img.cols, CV_8UC1);
        circle(img_mask, Point(input_img.rows / 2, input_img.cols / 2), nROI_radius, Scalar::all(255), -1);

        Mat img_masked;
        bitwise_and(input_img, img_mask, img_masked); // img_egdes && img_mask --> img_masked
        return img_masked;
    }

    void Victimboard::CLAHE(Mat &image)
    {
        cvtColor(image, image, CV_RGB2Lab);
        vector<cv::Mat> lab(3);
        split(image, lab);
        Ptr<cv::CLAHE> clahe = createCLAHE(3, Size(8, 8));
        Mat dst;
        clahe->apply(lab[0], dst);
        dst.copyTo(lab[0]);
        merge(lab, image);
        cvtColor(image, image, CV_Lab2RGB);
    }

    void Victimboard::detect_location()
    {

        for (int i = 0; i < 8; i++)
        {
            divided_Image_data[i].Image = clone_mat(divided_Image_data[i].position);
            resize(divided_Image_data[i].Image, divided_Image_data[i].Image,
                   Size(200, 200), 0, 0, CV_INTER_LINEAR);
            orb->detect(divided_Image_data[i].Image, divided_Image_data[i].Keypoints,
                        noArray());
            // cout << i << ") KeyPoints: " << divided_Image_data[i].Keypoints.size()<< endl;
        }

        int count_KeyPoints[8];
        for (int i = 0; i < 8; i++)
            count_KeyPoints[i] = divided_Image_data[i].Keypoints.size();
        sort(count_KeyPoints, count_KeyPoints + 8);

        if ((count_KeyPoints[0] != count_KeyPoints[1]) &&
            (count_KeyPoints[1] == count_KeyPoints[2])) // if [0][1][1]....
        {
            for (int j = 0; j < 8; j++)
            {
                if (divided_Image_data[j].Keypoints.size() ==
                    count_KeyPoints[0]) // count_KeyPoints[0] == count_KeyPoints[1]
                    save_image_position__Rotation_Direction[0] = j;
            }
            if (save_image_position__Rotation_Direction[0] == 1)
                save_image_position__Rotation_Direction[1] = 6;
            else if (save_image_position__Rotation_Direction[0] == 3)
                save_image_position__Rotation_Direction[1] = 4;
            else if (save_image_position__Rotation_Direction[0] == 4)
                save_image_position__Rotation_Direction[1] = 3;
            else if (save_image_position__Rotation_Direction[0] == 6)
                save_image_position__Rotation_Direction[1] = 1;
        }
        // if [0][0][0].... retry
        else if (count_KeyPoints[0] ==
                 count_KeyPoints[1])
        { // if the number of Keypoints is same
            int count = 0;
            for (int j = 0; j < 8; j++)
            {
                if (divided_Image_data[j].Keypoints.size() ==
                    count_KeyPoints[0]) // count_KeyPoints[0] == count_KeyPoints[1]
                {
                    save_image_position__Rotation_Direction[count] = j;
                    count++;
                    if (count == 2)
                    {
                        break;
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < 8; i++)
            {
                if (count_KeyPoints[0] == divided_Image_data[i].Keypoints.size())
                    save_image_position__Rotation_Direction[0] = i;
                if (count_KeyPoints[1] == divided_Image_data[i].Keypoints.size())
                    save_image_position__Rotation_Direction[1] = i;
            }
        }

        if ((save_image_position__Rotation_Direction[0] == 1 && save_image_position__Rotation_Direction[1] == 6) || (save_image_position__Rotation_Direction[1] == 1 && save_image_position__Rotation_Direction[0] == 6))
        {
            start_motion = true;
        }
        else if ((save_image_position__Rotation_Direction[0] == 3 && save_image_position__Rotation_Direction[1] == 4) || (save_image_position__Rotation_Direction[1] == 3 && save_image_position__Rotation_Direction[0] == 4))
        {
            start_motion = true;
        }
        else
        {
            start_motion = false;
        }

        // cout << endl<< "Rotation Direction Image: "<< save_image_position__Rotation_Direction[0] << ", "<< save_image_position__Rotation_Direction[1] << endl<< endl;

        //------------------------------------------------hazmat----------------------------------------------

        for (const auto &point : hazmat_loc)
        {
            // cout << point.x << "   " << point.y << endl;
            // cout << divided_Image_data[0].position.x << "     " <<
            // divided_Image_data[0].position.y << endl;

            if (divided_Image_data[0].position.contains(cv::Point(point.x, point.y)) || divided_Image_data[7].position.contains(cv::Point(point.x, point.y)))
            {
                C_up = divided_Image_data[2].Image;
                C_down = divided_Image_data[5].Image;
                exist_c = true;
            }
            else if (divided_Image_data[2].position.contains(cv::Point(point.x, point.y)) || divided_Image_data[5].position.contains(cv::Point(point.x, point.y)))
            {
                C_up = divided_Image_data[0].Image;
                C_down = divided_Image_data[7].Image;
                exist_c = true;
            }
        }
    }

    void Victimboard::thermal_location(int loc)
    {
    }

    void Victimboard::divide_box()
    {

        labeling(Captured_Image_to_Binary, image_x, image_y, image_width_divided3,
                 image_height_divided3);

        roi = Mat::zeros(Captured_Image_to_Binary.rows, Captured_Image_to_Binary.cols,
                         CV_8UC1);
        adaptive_capture = Image_to_Binary_adaptive;
        rectangle(roi,
                  Rect(image_x - 10, image_y - 10, image_width_divided3 + 20,
                       image_height_divided3 + 20),
                  Scalar::all(255), -1, LINE_8, 0);
        bitwise_and(adaptive_capture, roi, result, noArray());
        dilate(result, result, mask, Point(-1, -1), 2);

        labeling(result, image_x, image_y, image_width_divided3,
                 image_height_divided3);
        image_width_divided3 = image_width_divided3 / 3;
        image_height_divided3 = image_height_divided3 / 3;

        rectangle(Captured_Image_RGB,
                  Rect(image_x, image_y, image_width_divided3 * 3,
                       image_height_divided3 * 3),
                  cv::Scalar(0, 0, 255), 2); // draw rect
        line(Captured_Image_RGB, Point(image_x + image_width_divided3, image_y),
             Point(image_x + image_width_divided3,
                   image_y + image_height_divided3 * 3),
             cv::Scalar(0, 0, 255), 2, 8); // divide 9
        line(Captured_Image_RGB, Point(image_x + image_width_divided3 * 2, image_y),
             Point(image_x + image_width_divided3 * 2,
                   image_y + image_height_divided3 * 3),
             cv::Scalar(0, 0, 255), 2, 8);
        line(Captured_Image_RGB, Point(image_x, image_y + image_height_divided3),
             Point(image_x + image_width_divided3 * 3,
                   image_y + image_height_divided3),
             cv::Scalar(0, 0, 255), 2, 8);
        line(Captured_Image_RGB, Point(image_x, image_y + image_height_divided3 * 2),
             Point(image_x + image_width_divided3 * 3,
                   image_y + image_height_divided3 * 2),
             cv::Scalar(0, 0, 255), 2, 8);

        divided_Image_data[0].position =
            Rect(image_x, image_y, image_width_divided3, image_height_divided3);
        divided_Image_data[1].position =
            Rect(image_x + image_width_divided3, image_y, image_width_divided3,
                 image_height_divided3);
        divided_Image_data[2].position =
            Rect(image_x + image_width_divided3 * 2, image_y, image_width_divided3,
                 image_height_divided3);
        divided_Image_data[3].position =
            Rect(image_x, image_y + image_height_divided3, image_width_divided3,
                 image_height_divided3);
        divided_Image_data[4].position =
            Rect(image_x + image_width_divided3 * 2, image_y + image_height_divided3,
                 image_width_divided3, image_height_divided3);
        divided_Image_data[5].position =
            Rect(image_x, image_y + image_height_divided3 * 2, image_width_divided3,
                 image_height_divided3);
        divided_Image_data[6].position =
            Rect(image_x + image_width_divided3, image_y + image_height_divided3 * 2,
                 image_width_divided3, image_height_divided3);
        divided_Image_data[7].position = Rect(
            image_x + image_width_divided3 * 2, image_y + image_height_divided3 * 2,
            image_width_divided3, image_height_divided3);
        // 0 1 2
        // 3   4
        // 5 6 7
    }

    void Victimboard::labeling(const Mat &input_img, int &_x, int &_y, int &_width,
                               int &_height)
    {
        int num_labels = connectedComponentsWithStats(
            input_img, img_labeling, stats, centroids, 8, CV_32S); // labeling

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

    void Victimboard::set_yolo()
    {
        auto output_names = net.getUnconnectedOutLayersNames();
        frame = clone_mat.clone();
        std::vector<cv::Mat> detections;

        auto total_start = std::chrono::steady_clock::now();
        cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(416, 416), cv::Scalar(),
                               true, false, CV_32F);
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
                    hazmat_loc.emplace_back(rect.x + (rect.width / 2),
                                            rect.y + (rect.height / 2));
                    cv::rectangle(Captured_Image_RGB, cv::Point(rect.x, rect.y),
                                  cv::Point(rect.x + rect.width, rect.y + rect.height),
                                  color, 3);

                    std::ostringstream label_ss;

                    string name = class_names[c];

                    label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];

                    auto label = label_ss.str();
                    int baseline;
                    auto label_bg_sz = cv::getTextSize(
                        label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                    cv::rectangle(
                        Captured_Image_RGB,
                        cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10),
                        cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
                    cv::putText(Captured_Image_RGB, label.c_str(),
                                cv::Point(rect.x, rect.y - baseline - 5),
                                cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
                }
            }
        }

        auto total_end = std::chrono::steady_clock::now();

        float inference_fps =
            1000.0 /
            std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start)
                .count();
        float total_fps =
            1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(
                         total_end - total_start)
                         .count();
        std::ostringstream stats_ss;
        stats_ss << std::fixed << std::setprecision(2);
        stats_ss << "Inference FPS: " << inference_fps
                 << ", Total FPS: " << total_fps;
        auto stats = stats_ss.str();

        int baseline;
        auto stats_bg_sz = cv::getTextSize(
            stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        cv::rectangle(Captured_Image_RGB, cv::Point(0, 0),
                      cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(Captured_Image_RGB, stats.c_str(),
                    cv::Point(0, stats_bg_sz.height + 5),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));
    }

    bool Victimboard::isRectOverlapping(const cv::Rect &rect1,
                                        const cv::Rect &rect2)
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

} // namespace vision_rescue