#include <iostream>
#include <opencv2/opencv.hpp>

struct ArmorDetection
{
    //这里直接赋值，对变量初始化一下，下面就不用写构造函数了
    std::vector<cv::Point2f> corners;
    float confidence=0.0f;
    int color_id=-1;
    int number_id=-1;

    cv::Rect getBoundingRect()const
    {
        if(corners.empty()||corners.size()!=4)
        {
            return cv::Rect(0,0,0,0);
        }

        float min_x=corners[0].x;
        float max_x=corners[0].x;
        float min_y=corners[0].y;
        float max_y=corners[0].y;

        for(const auto& point:corners)
        {
            if(point.x<min_x){min_x=point.x;}
            if(point.x>max_x){max_x=point.x;}
            if(point.y<min_y){min_y=point.y;}
            if(point.y>max_y){max_y=point.y;}
        }

        return cv::Rect(
            static_cast<int>(min_x),
            static_cast<int>(min_y),
            static_cast<int>(max_x-min_x),
            static_cast<int>(max_y-min_y)
        );
    }
        // 调试函数：打印检测框信息
    void print() const {
        std::cout << "置信度: " << confidence;
        std::cout << ", 矩形: (" << getBoundingRect().x << "," 
                  << getBoundingRect().y << ") " 
                  << getBoundingRect().width << "x" 
                  << getBoundingRect().height;
        std::cout << std::endl;
    }
};



float calculateIOU(const cv::Rect& rect1,const cv::Rect& rect2)
{
    int x1=std::max(rect1.x,rect2.x);
    int y1=std::max(rect1.y,rect2.y);
    int x2=std::min((rect1.x+rect1.width),(rect2.x+rect2.width));
    int y2=std::min((rect1.y+rect1.height),(rect2.y+rect2.height));

    int intersection_area=std::max(0,x2-x1)*std::max(0,y2-y1);

    int area1=rect1.width*rect1.height;
    int area2=rect2.width*rect2.height;
    int union_area=area1+area2-intersection_area;

    if(union_area==0)
    {
        return 0.0f;
    }

    return static_cast<float>(intersection_area)/static_cast<float>(union_area);
}



float sigmoid(float x)
{
    return 1.0f/(1.0f+std::exp(-x));
}



cv::Mat LetterBox(const cv::Mat& input)
{
    int w=input.cols;
    int h=input.rows;
    float scale=std::min(640.0f/w,640.0f/h);
    int new_w=static_cast<int>(w*scale);
    int new_h=static_cast<int>(h*scale);
    cv::Mat resized;
    cv::resize(input,resized,cv::Size(new_w,new_h));
    cv::Mat canvas=cv::Mat::zeros(640,640,input.type());
    cv::Mat roi=canvas(cv::Rect(0,0,new_w,new_h));
    resized.copyTo(roi);
    return canvas;
}

std::vector<ArmorDetection> nmsBasic(const std::vector<ArmorDetection>& detections,float iou_threshold=0.5f)
{
    std::cout<<"nms启动"<<std::endl;

    if(detections.empty())
    {
        std::cout<<"没有检测框需要处理"<<std::endl;
        return  {};
    }

    std::vector<ArmorDetection> remaining=detections;
    std::vector<ArmorDetection> result;
    std::vector<ArmorDetection> new_remaining;

    while(!remaining.empty())
    {
        std::vector<ArmorDetection> new_remaining;
        int best_index=0;
        float best_confidence=remaining[0].confidence;

        for(int i=1;i<remaining.size();i++)
        {
            if(remaining[i].confidence>best_confidence)
            {
                best_confidence=remaining[i].confidence;
                best_index=i;
            }
        }

        ArmorDetection best_detection=remaining[best_index];
        result.push_back(best_detection);
        std::cout << "保留框 " << result.size() << ": ";
        best_detection.print();

        remaining.erase(remaining.begin()+best_index);

        for(int i=0;i<remaining.size();i++)
        {
            float iou=calculateIOU(best_detection.getBoundingRect(),remaining[i].getBoundingRect());
            if(iou<=iou_threshold)
            {
                new_remaining.push_back(remaining[i]);
            }
            else
            {
                std::cout << "  抑制框(IOU=" << iou << "): ";
                remaining[i].print();
            }
        }
        remaining=new_remaining;

    }
    std::cout << "NMS完成，保留 " << result.size() << " 个框" << std::endl;
    return result;
}



int main()
{
    cv::dnn::Net net=cv::dnn::readNetFromONNX("../resources/yolov5.onnx");
    if(net.empty())
    {
        std::cout<<"模型加载失败"<<std::endl;
        return -1;
    }
    std::cout<<"模型加载成功"<<std::endl;


    cv::VideoCapture cap("../resources/demo.avi");
    if (!cap.isOpened()) 
    {
       std::cout << "无法打开视频文件！" << std::endl;
       return -1;
    }
    std::cout << "视频总帧数: " << cap.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;

    cv::Mat frame;
    int frame_count = 0;
    while (cap.read(frame)) {
    frame_count++;
    std::cout << "\n--- 处理第 " << frame_count << " 帧 ---" << std::endl;

    // cv::Mat try_image_0=cv::imread("../resources/测试帧.png");
    cv::Mat try_image=LetterBox(frame);

    
    // cv::Mat test_image(640, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    
    // std::cout << "测试图像创建成功" << std::endl;
    // std::cout << "尺寸: " << test_image.cols << "x" << test_image.rows << std::endl;
    
    // // 调试测试图片的函数
    // cv::imshow("测试图像", test_image);
    // cv::waitKey(100000);
    // cv::destroyWindow("测试图像");

    cv::Mat blob;  // 这个变量将保存预处理后的图像
    
    // 这是YOLO预处理的关键函数，我们一个一个参数解释
    cv::dnn::blobFromImage(try_image,  // 输入图像
                           blob,        // 输出blob（预处理结果）
                           1/255.0,     // 参数1: 归一化，把0-255变成0-1
                           cv::Size(640, 640),  // 参数2: 调整到640×640（这里图像已经是640×640，所以没变化）
                           cv::Scalar(0,0,0),   // 参数3: 均值减法（这里不减，所以填0,0,0）
                           true,        // 参数4: 交换红蓝通道（BGR转RGB）
                           false);      // 参数5: 不裁剪
    std::cout << "预处理完成！" << std::endl;
    std::cout << "blob形状: ";
    for (int i = 0; i < blob.dims; i++) {
        std::cout << blob.size[i];
        if (i < blob.dims - 1) std::cout << " × ";
    }
    std::cout << std::endl;
    
    // 检查blob形状是否正确
    // 应该是: 1 × 3 × 640 × 640 （1张图，3个颜色通道，640高，640宽）
    if (blob.dims == 4 && blob.size[0] == 1 && blob.size[1] == 3 && 
        blob.size[2] == 640 && blob.size[3] == 640) {
        std::cout << "✓ blob形状正确，可以用于推理" << std::endl;
    } else {
        std::cout << "⚠ blob形状可能有问题" << std::endl;
    }

    net.setInput(blob);
    cv::Mat output = net.forward();


    std::cout << "  输出形状: ";
    for (int i = 0; i < output.dims; i++) {
        std::cout << output.size[i];
        if (i < output.dims - 1) std::cout << " × ";
    }
    std::cout << std::endl;
    // ========== 修改点1：创建存储所有检测框的vector ==========
    std::vector<ArmorDetection> all_detections;
    
    std::cout << "\n开始解析YOLO输出并收集检测框..." << std::endl;
    std::cout << "置信度阈值: 0.7" << std::endl;

    int total_detection = output.size[1];
    int high_confidence_count = 0;

    // ========== 修改点2：完善检测框收集逻辑 ==========
    for(int i = 0; i < total_detection; i++)
    {
        const float* detection = output.ptr<float>(0, i);
        float confidence = sigmoid(detection[8]);
        
        if(confidence > 0.7)
        {
            high_confidence_count++;
            
            // 创建ArmorDetection对象
            ArmorDetection armor;
            armor.confidence = confidence;
            
            // 提取4个角点坐标（根据您提供的YOLO输出格式）
            // 注意：根据您的表格，角点顺序是 (0,1), (6,7), (4,5), (2,3)
            armor.corners.clear();
            armor.corners.push_back(cv::Point2f(detection[0], detection[1]));  // 角点1
            armor.corners.push_back(cv::Point2f(detection[6], detection[7]));  // 角点2
            armor.corners.push_back(cv::Point2f(detection[4], detection[5]));  // 角点3
            armor.corners.push_back(cv::Point2f(detection[2], detection[3]));  // 角点4
            
            // 提取颜色分类（9-12列）
            float color_logits[4] = {detection[9], detection[10], detection[11], detection[12]};
            armor.color_id = 0;
            for (int c = 1; c < 4; c++) {
                if (color_logits[c] > color_logits[armor.color_id]) {
                    armor.color_id = c;
                }
            }
            
            // 提取数字分类（13-21列）
            float num_logits[9];
            for (int n = 0; n < 9; n++) {
                num_logits[n] = detection[13 + n];
            }
            armor.number_id = 0;
            for (int n = 1; n < 9; n++) {
                if (num_logits[n] > num_logits[armor.number_id]) {
                    armor.number_id = n;
                }
            }
            
            // 保存到all_detections
            all_detections.push_back(armor);
            
            // 调试输出
            std::cout << "\n高置信度检测框 #" << high_confidence_count << ":" << std::endl;
            std::cout << "  置信度: " << confidence << std::endl;
            std::cout << "  位置: ";
            armor.print();
        }
    }
    
    std::cout << "\n收集到 " << all_detections.size() << " 个高置信度检测框" << std::endl;
    
    // ========== 修改点3：应用NMS过滤 ==========
    std::cout << "\n=== 应用NMS过滤 ===" << std::endl;
    std::vector<ArmorDetection> filtered_detections = nmsBasic(all_detections, 0.5f);
    
    std::cout << "\nNMS过滤结果:" << std::endl;
    std::cout << "  过滤前: " << all_detections.size() << " 个框" << std::endl;
    std::cout << "  过滤后: " << filtered_detections.size() << " 个框" << std::endl;
    
    // ========== 修改点4：绘制NMS过滤后的结果 ==========
    std::cout << "\n开始绘制NMS过滤后的检测结果..." << std::endl;
    
    cv::Mat result = try_image.clone();
    
    // 颜色映射
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255),    // 红色 (0)
        cv::Scalar(255, 0, 0),    // 蓝色 (1)
        cv::Scalar(0, 255, 255),  // 黄色 (2)
        cv::Scalar(255, 0, 255)   // 紫色 (3)
    };
    
    // 数字标签（根据您的表格）
    std::vector<std::string> number_labels = {
        "1号", "2号", "3号", "4号", "5号",
        "哨兵", "前哨站", "基地", "非装甲板"
    };
    
    for (const auto& armor : filtered_detections) {
        // 绘制四边形边框（使用颜色分类）
        for (int j = 0; j < 4; j++) {
            int next_j = (j + 1) % 4;
            cv::line(result, 
                    armor.corners[j], 
                    armor.corners[next_j],
                    colors[armor.color_id % colors.size()],  // 根据颜色ID选择颜色
                    2);  // 线宽2
        }
        
        // 绘制角点
        for (int j = 0; j < 4; j++) {
            cv::circle(result, armor.corners[j], 
                      5, cv::Scalar(0, 255, 0), -1);  // 绿色角点
        }
        
        // 计算中心点用于显示标签
        cv::Point2f center(0, 0);
        for (const auto& corner : armor.corners) {
            center += corner;
        }
        center /= 4.0f;
        
        // 显示数字标签和置信度
        std::string label;
        if (armor.number_id >= 0 && armor.number_id < number_labels.size()) {
            label = number_labels[armor.number_id] + 
                   " (" + std::to_string(int(armor.confidence * 100)) + "%)";
        } else {
            label = "未知 (" + std::to_string(int(armor.confidence * 100)) + "%)";
        }
        
        cv::putText(result, label, 
                   center - cv::Point2f(40, 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        
        // 显示颜色标签
        std::string color_text;
        switch (armor.color_id) {
            case 0: color_text = "红"; break;
            case 1: color_text = "蓝"; break;
            case 2: color_text = "熄灭"; break;
            case 3: color_text = "紫"; break;
            default: color_text = "未知";
        }
        
        cv::putText(result, color_text, 
                   center + cv::Point2f(-20, 25),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[armor.color_id % colors.size()], 2);
    }
    
    std::cout << "绘制了 " << filtered_detections.size() << " 个检测框" << std::endl;
    
    // 显示结果
    cv::namedWindow("Detection Results (NMS Filtered)", cv::WINDOW_NORMAL);
    cv::imshow("Detection Results (NMS Filtered)", result);
    
    // 保存结果
    cv::imwrite("detection_result_nms.jpg", result);
    std::cout << "结果已保存到: detection_result_nms.jpg" << std::endl;
    
    // 可选：同时显示原始图像用于对比
    cv::namedWindow("Original Image", cv::WINDOW_NORMAL);
    cv::imshow("Original Image", try_image);
    
    cv::waitKey(30);

}
    cv::destroyAllWindows();
    
    return 0;
}