#include "review.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Record {
    std::string json;
    long long frame_id = 0;
    std::string image;
};

void put(cv::Mat& img, const std::string& s, cv::Point p, double sc, const cv::Scalar& c) {
    int base = 0;
    cv::Size sz = cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, sc, 1, &base);
    cv::rectangle(img, {p.x - 4, p.y - sz.height - 6, sz.width + 8, sz.height + base + 10}, cv::Scalar(20, 20, 20),
                  cv::FILLED);
    cv::putText(img, s, p, cv::FONT_HERSHEY_SIMPLEX, sc, c, 1, cv::LINE_AA);
}

cv::Mat render(const std::string& dir, const Record& r, const cv::Matx33d& H, const std::string& human) {
    cv::FileStorage fs(r.json, cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
    cv::Mat img = cv::imread((fs::path(dir) / r.image).string());
    if (img.empty()) {
        img = cv::Mat(720, 1280, CV_8UC3, cv::Scalar(40, 0, 0));
        put(img, "missing image " + r.image, {20, 360}, 1.0, cv::Scalar(0, 0, 255));
        return img;
    }

    // Stored pseudo-labels: TuSimple-style x per h_sample row.
    std::vector<int> rows;
    fs["h_samples"] >> rows;
    for (const cv::FileNode& lane : fs["lanes_image"]) {
        std::string slot = static_cast<std::string>(lane["slot"]);
        bool inner = slot == "left" || slot == "right";
        std::vector<cv::Point> pts;
        std::size_t i = 0;
        for (const cv::FileNode& x : lane["x"]) {
            int xv = static_cast<int>(x);
            if (i < rows.size() && xv >= 0) pts.push_back({xv, rows[i]});
            ++i;
        }
        cv::Scalar c = inner ? cv::Scalar(240, 210, 60) : cv::Scalar(190, 130, 70);
        for (const cv::Point& p : pts) cv::circle(img, p, 3, c, cv::FILLED, cv::LINE_AA);
        if (pts.size() > 1) cv::polylines(img, pts, false, c, 1, cv::LINE_AA);
    }
    // Raw evidence that produced them.
    for (const cv::FileNode& m : fs["measurements"]) {
        bool ok = static_cast<int>(m["accepted"]) != 0;
        for (const cv::FileNode& p : m["points_px"]) {
            cv::Point q(static_cast<int>(static_cast<double>(p[0])), static_cast<int>(static_cast<double>(p[1])));
            cv::circle(img, q, 2, ok ? cv::Scalar(80, 230, 80) : cv::Scalar(60, 60, 240), cv::FILLED);
        }
    }
    // Predicted path (vehicle frame, metres) through the stored homography.
    std::vector<cv::Point> path;
    for (const cv::FileNode& p : fs["path"]) {
        double X = p[0], Y = p[1];
        if (X < 6.0) continue;
        cv::Vec3d q = H * cv::Vec3d(X, Y, 1.0);
        path.push_back({cvRound(q[0] / q[2]), cvRound(q[1] / q[2])});
    }
    if (path.size() > 1) cv::polylines(img, path, false, cv::Scalar(40, 150, 255), 3, cv::LINE_AA);

    std::string reasons;
    for (const cv::FileNode& n : fs["reasons"]) reasons += static_cast<std::string>(n) + " ";
    char b[256];
    std::snprintf(b, sizeof(b), "frame %lld  t=%.2fs  %s  conf %.2f", r.frame_id,
                  static_cast<double>(fs["media_time_s"]), static_cast<std::string>(fs["status"]).c_str(),
                  static_cast<double>(fs["confidence"]));
    put(img, b, {16, 32}, 0.8, cv::Scalar(255, 255, 255));
    put(img, "saved because: " + reasons, {16, 62}, 0.6, cv::Scalar(200, 200, 200));
    put(img, "human label: " + (human.empty() ? std::string("none") : human), {16, 90}, 0.6,
        human == "bad" ? cv::Scalar(60, 60, 240) : cv::Scalar(120, 230, 120));
    put(img, "dots = stored lane labels, green/red = evidence, orange = path", {16, img.rows - 16}, 0.55,
        cv::Scalar(200, 200, 200));
    return img;
}

}  // namespace

int review_session(const std::string& dir, const std::string& out_dir) {
    std::ifstream in(fs::path(dir) / "frames.jsonl");
    if (!in) {
        std::cerr << "no frames.jsonl in " << dir << "\n";
        return 1;
    }
    cv::FileStorage session((fs::path(dir) / "session.json").string(), cv::FileStorage::READ);
    if (!session.isOpened()) {
        std::cerr << "no session.json in " << dir << "\n";
        return 1;
    }
    std::vector<double> h;
    session["camera"]["H_ground_to_image"] >> h;
    if (h.size() != 9) {
        std::cerr << "session.json has no ground homography\n";
        return 1;
    }
    cv::Matx33d H(h.data());

    std::vector<Record> records;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        cv::FileStorage fs(line, cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
        Record r;
        r.json = line;
        r.frame_id = static_cast<int>(fs["frame_id"]);
        r.image = static_cast<std::string>(fs["image"]);
        records.push_back(r);
    }
    std::cout << records.size() << " records in " << dir << "\n";
    if (records.empty()) return 0;

    if (!out_dir.empty()) {
        fs::create_directories(out_dir);
        for (const Record& r : records) {
            cv::imwrite((fs::path(out_dir) / ("review_" + std::to_string(r.frame_id) + ".jpg")).string(),
                        render(dir, r, H, ""));
        }
        std::cout << "wrote " << records.size() << " renders to " << out_dir << "\n";
        return 0;
    }

    std::ofstream labels(fs::path(dir) / "labels.jsonl", std::ios::app);
    std::vector<std::string> human(records.size());
    std::size_t i = 0;
    std::cout << "a/d = previous/next, g/b = label good/bad, q = quit\n";
    while (true) {
        cv::imshow("Atlas Vision review", render(dir, records[i], H, human[i]));
        int key = cv::waitKey(0);
        if (key == 'q' || key == 27) break;
        if (key == 'd' && i + 1 < records.size()) ++i;
        if (key == 'a' && i > 0) --i;
        if (key == 'g' || key == 'b') {
            human[i] = key == 'g' ? "good" : "bad";
            labels << "{\"frame_id\":" << records[i].frame_id << ",\"human\":\"" << human[i] << "\"}\n";
            labels.flush();
        }
    }
    cv::destroyAllWindows();
    return 0;
}
