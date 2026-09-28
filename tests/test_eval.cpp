#include "test_main.hpp"
#include "../tools/lane_rasterize.hpp"
#include "../tools/session_eval.hpp"
#include "../tools/tusimple.hpp"
#include "../tools/tusimple_eval.hpp"
#include "../tools/tusimple_score.hpp"
#include "../vision/cv_compat.hpp"

namespace {

std::vector<int> rows_240_710() {
    std::vector<int> r;
    for (int v = 240; v <= 710; v += 10) r.push_back(v);
    return r;
}

// x = x0 + slope * (y - 240) on every row.
std::vector<int> line_lane(const std::vector<int>& rows, double x0, double slope) {
    std::vector<int> xs;
    for (int y : rows) xs.push_back(static_cast<int>(std::lround(x0 + slope * (y - 240))));
    return xs;
}

std::vector<int> shifted(std::vector<int> xs, int dx) {
    for (int& x : xs) x = x >= 0 ? x + dx : x;
    return xs;
}

// `var` is the offset variance; heading and curvature variances scale so that
// each term contributes about as much at 30-100 m.
RoadState straight_road(double y_c, double width, double var) {
    RoadState r;
    r.valid = true;
    r.x = cv::Vec4d(y_c, 0.0, 0.0, width);
    r.P = cv::Matx44d::diag(cv::Vec4d(var, var * 1e-3, var * 1e-8, var));
    return r;
}

CameraModel test_camera(double pitch, double yaw, double height) {
    CameraModel c = default_camera_model({1280, 720}, 60.0);
    c.pitch_rad = pitch;
    c.yaw_rad = yaw;
    c.height_m = height;
    c.update();
    return c;
}

}  // namespace

// ---- official TuSimple metric ----

TEST(tusimple_perfect_prediction_scores_100_percent) {
    std::vector<int> rows = rows_240_710();
    std::vector<std::vector<int>> gt = {line_lane(rows, 600, -0.8), line_lane(rows, 700, 0.8)};
    TuSimpleScore s;
    CHECK(tusimple_bench(gt, gt, rows, 10.0, s));
    CHECK_NEAR(s.accuracy, 1.0, 1e-12);
    CHECK_NEAR(s.fp, 0.0, 1e-12);
    CHECK_NEAR(s.fn, 0.0, 1e-12);
    CHECK(s.matched == 2);
}

TEST(tusimple_30px_shift_matches_nothing) {
    std::vector<int> rows = rows_240_710();
    std::vector<std::vector<int>> gt = {line_lane(rows, 400, 0.0), line_lane(rows, 900, 0.0)};
    std::vector<std::vector<int>> pred = {shifted(gt[0], 30), shifted(gt[1], -30)};
    TuSimpleScore s;
    CHECK(tusimple_bench(pred, gt, rows, 10.0, s));
    CHECK_NEAR(s.accuracy, 0.0, 1e-12);
    CHECK_NEAR(s.fp, 1.0, 1e-12);
    CHECK_NEAR(s.fn, 1.0, 1e-12);
    // 19 px is inside the 20 px threshold of a vertical lane.
    pred = {shifted(gt[0], 19), shifted(gt[1], -19)};
    CHECK(tusimple_bench(pred, gt, rows, 10.0, s));
    CHECK_NEAR(s.accuracy, 1.0, 1e-12);
}

TEST(tusimple_missing_lane_is_false_negative) {
    std::vector<int> rows = rows_240_710();
    std::vector<std::vector<int>> gt = {line_lane(rows, 600, -0.8), line_lane(rows, 700, 0.8)};
    TuSimpleScore s;
    CHECK(tusimple_bench({gt[0]}, gt, rows, 10.0, s));
    CHECK_NEAR(s.accuracy, 0.5, 1e-12);
    CHECK_NEAR(s.fp, 0.0, 1e-12);
    CHECK_NEAR(s.fn, 0.5, 1e-12);
    // No prediction at all: FP is defined as 0, every gt lane is missed.
    CHECK(tusimple_bench({}, gt, rows, 10.0, s));
    CHECK_NEAR(s.accuracy, 0.0, 1e-12);
    CHECK_NEAR(s.fp, 0.0, 1e-12);
    CHECK_NEAR(s.fn, 1.0, 1e-12);
}

TEST(tusimple_extra_lane_is_false_positive) {
    std::vector<int> rows = rows_240_710();
    std::vector<std::vector<int>> gt = {line_lane(rows, 600, -0.8), line_lane(rows, 700, 0.8)};
    std::vector<std::vector<int>> pred = {gt[0], gt[1], line_lane(rows, 640, 2.0)};
    TuSimpleScore s;
    CHECK(tusimple_bench(pred, gt, rows, 10.0, s));
    CHECK_NEAR(s.accuracy, 1.0, 1e-12);
    CHECK_NEAR(s.fp, 1.0 / 3.0, 1e-12);
    CHECK_NEAR(s.fn, 0.0, 1e-12);
    // More than #gt + 2 predictions: the official early-out (0, 0, 1).
    pred.push_back(line_lane(rows, 100, 0.0));
    pred.push_back(line_lane(rows, 1200, 0.0));
    CHECK(tusimple_bench(pred, gt, rows, 10.0, s));
    CHECK(s.rejected);
    CHECK_NEAR(s.accuracy, 0.0, 1e-12);
    CHECK_NEAR(s.fp, 0.0, 1e-12);
    CHECK_NEAR(s.fn, 1.0, 1e-12);
    // Too slow (> 200 ms) is rejected the same way.
    CHECK(tusimple_bench({gt[0], gt[1]}, gt, rows, 250.0, s));
    CHECK(s.rejected);
    // Wrong length is a format error.
    CHECK(!tusimple_bench({std::vector<int>(3, 5)}, gt, rows, 10.0, s));
}

TEST(tusimple_threshold_grows_with_lane_angle) {
    std::vector<int> rows = rows_240_710();
    // dx/dy = 1: 45 degrees, threshold 20 / cos 45 = 28.3 px.
    std::vector<int> slanted = line_lane(rows, 200, 1.0);
    std::vector<int> vertical = line_lane(rows, 1000, 0.0);
    CHECK_NEAR(tusimple_lane_angle(slanted, rows), CV_PI / 4, 1e-3);
    CHECK_NEAR(tusimple_lane_angle(vertical, rows), 0.0, 1e-12);
    TuSimpleScore s;
    CHECK(tusimple_bench({shifted(slanted, 25)}, {slanted}, rows, 10.0, s));
    CHECK_NEAR(s.accuracy, 1.0, 1e-12);
    CHECK(tusimple_bench({shifted(vertical, 25)}, {vertical}, rows, 10.0, s));
    CHECK_NEAR(s.accuracy, 0.0, 1e-12);
}

TEST(tusimple_point_accuracy_counts_all_h_samples) {
    std::vector<int> rows = rows_240_710();  // 48 rows
    std::vector<int> gt = line_lane(rows, 640, 0.0);
    for (int i = 0; i < 12; ++i) gt[i] = -2;  // lane starts at row 360
    // Missing in both counts as a hit ...
    std::vector<int> pred = gt;
    CHECK_NEAR(tusimple_line_accuracy(pred, gt, 20.0), 1.0, 1e-12);
    // ... drawing past the labelled end counts as a miss: 36 / 48 = 0.75.
    pred = line_lane(rows, 640, 0.0);
    CHECK_NEAR(tusimple_line_accuracy(pred, gt, 20.0), 0.75, 1e-12);
    TuSimpleScore s;
    CHECK(tusimple_bench({pred}, {gt}, rows, 10.0, s));
    CHECK(s.matched == 0);
    CHECK_NEAR(s.fn, 1.0, 1e-12);
}

TEST(tusimple_more_than_four_gt_lanes_forgives_the_worst) {
    std::vector<int> rows = rows_240_710();
    std::vector<std::vector<int>> gt;
    for (int i = 0; i < 5; ++i) gt.push_back(line_lane(rows, 100 + 250 * i, 0.0));
    std::vector<std::vector<int>> pred(gt.begin(), gt.begin() + 4);
    TuSimpleScore s;
    CHECK(tusimple_bench(pred, gt, rows, 10.0, s));
    CHECK_NEAR(s.accuracy, 1.0, 1e-12);  // (4 + 0 - min) / min(4, 5)
    CHECK_NEAR(s.fn, 0.0, 1e-12);        // one FN forgiven
    CHECK_NEAR(s.fp, 0.0, 1e-12);
}

TEST(tusimple_label_line_parses) {
    TuSimpleLabel l;
    std::string e;
    CHECK(parse_tusimple_label("{\"lanes\": [[-2, 632, 625], [-2, -2, 700]], \"h_samples\": [240, 250, 260], "
                               "\"raw_file\": \"clips/0530/1492626760788443246_0/20.jpg\"}",
                               l, e));
    CHECK(l.raw_file == "clips/0530/1492626760788443246_0/20.jpg");
    CHECK(l.h_samples.size() == 3 && l.h_samples[2] == 260);
    CHECK(l.lanes.size() == 2 && l.lanes[0][0] == -2 && l.lanes[0][1] == 632 && l.lanes[1][2] == 700);
    CHECK(!parse_tusimple_label("{\"lanes\": [[1, 2]], \"h_samples\": [240, 250, 260], \"raw_file\": \"a\"}", l, e));
}

TEST(external_predictions_are_matched_by_raw_file_and_scored) {
    std::vector<TuSimpleLabel> gt(3);
    std::string e;
    CHECK(parse_tusimple_label("{\"lanes\": [[-2, 600, 590], [-2, 700, 710]], \"h_samples\": [240, 250, 260], "
                               "\"raw_file\": \"clips/a/20.jpg\"}", gt[0], e));
    CHECK(parse_tusimple_label("{\"lanes\": [[500, 490, 480]], \"h_samples\": [240, 250, 260], "
                               "\"raw_file\": \"clips/b/20.jpg\"}", gt[1], e));
    CHECK(parse_tusimple_label("{\"lanes\": [[300, 310, 320]], \"h_samples\": [240, 250, 260], "
                               "\"raw_file\": \"clips/c/20.jpg\"}", gt[2], e));

    // training/predict.py lines: extra "slots" / "frame_id" keys are ignored.
    std::vector<TuSimplePrediction> pred(3);
    CHECK(parse_tusimple_prediction(
        "{\"raw_file\": \"clips/a/20.jpg\", \"h_samples\": [240, 250, 260], \"lanes\": [[-2, 605, 595], "
        "[-2, 695, 705]], \"slots\": [\"left\", \"right\"], \"run_time\": 7.1, \"frame_id\": null}",
        pred[0], e));
    CHECK(pred[0].lanes.size() == 2 && pred[0].lanes[1][2] == 705);
    CHECK_NEAR(pred[0].run_time_ms, 7.1, 1e-9);
    // Official submission format (no h_samples); the lane is 40 px off.
    CHECK(parse_tusimple_prediction("{\"raw_file\": \"clips/b/20.jpg\", \"lanes\": [[540, 530, 520]], "
                                    "\"run_time\": 12}", pred[1], e));
    CHECK(pred[1].h_samples.empty());
    CHECK(parse_tusimple_prediction("{\"raw_file\": \"clips/zzz/20.jpg\", \"lanes\": [], \"run_time\": 1}",
                                    pred[2], e));
    CHECK(!parse_tusimple_prediction("{\"lanes\": []}", pred[2], e));
    CHECK(parse_tusimple_prediction("{\"raw_file\": \"clips/zzz/20.jpg\", \"lanes\": [], \"run_time\": 1}",
                                    pred[2], e));

    ScoreReport r = score_tusimple_predictions(gt, pred);
    CHECK(r.images.size() == 3);
    CHECK(r.images[0].status == "ok" && r.images[0].score.matched == 2);
    CHECK_NEAR(r.images[0].score.accuracy, 1.0, 1e-12);
    CHECK(r.images[1].status == "ok" && r.images[1].score.matched == 0);
    CHECK_NEAR(r.images[1].score.fp, 1.0, 1e-12);
    CHECK(r.images[2].status == "missing" && r.missing == 1);
    CHECK_NEAR(r.images[2].score.fn, 1.0, 1e-12);
    CHECK(r.unknown.size() == 1 && r.unknown[0] == "clips/zzz/20.jpg");
    CHECK_NEAR(r.accuracy, 1.0 / 3.0, 1e-12);
    CHECK_NEAR(r.fp, 1.0 / 3.0, 1e-12);
    CHECK_NEAR(r.fn, 2.0 / 3.0, 1e-12);
    CHECK_NEAR(r.accuracy_scored, 0.5, 1e-12);

    // Rows that differ from the label's are a format error, scored as no lanes.
    pred[0].h_samples = {240, 250, 270};
    r = score_tusimple_predictions(gt, pred);
    CHECK(r.images[0].status == "format" && r.format_errors == 1);
    CHECK_NEAR(r.images[0].score.accuracy, 0.0, 1e-12);
}

// ---- rasterizer ----

TEST(rasterizer_places_straight_road_at_expected_pixels) {
    double pitch = 0.05, h = 1.5;
    CameraModel cam = test_camera(pitch, 0.0, h);
    RoadState road = straight_road(0.0, 3.6, 1e-4);
    std::vector<int> rows = logger_h_samples(cam);
    RasterizeOptions opt;  // 6 .. 40 m, like the logger
    std::vector<int> left = rasterize_boundary(road, 0.5, cam, rows, opt);
    std::vector<int> right = rasterize_boundary(road, -0.5, cam, rows, opt);
    double f = cam.K(0, 0), cx = cam.K(0, 2), cy = cam.K(1, 2);
    double cp = std::cos(pitch), sp = std::sin(pitch);
    int drawn = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        double dv = rows[i] - cy;
        // Ground distance seen by this row, then the pinhole projection of
        // (X, +-1.8, 0) for a camera pitched down by `pitch`.
        double X = h * (f * cp - dv * sp) / (dv * cp + f * sp);
        double depth = sp * h + cp * X;
        bool in_range = X >= opt.x_min && X <= opt.x_max;
        CHECK((left[i] >= 0) == in_range);
        CHECK((right[i] >= 0) == in_range);
        if (!in_range) continue;
        ++drawn;
        CHECK_NEAR(left[i], cx - f * 1.8 / depth, 1.0);
        CHECK_NEAR(right[i], cx + f * 1.8 / depth, 1.0);
    }
    CHECK(drawn > 10);
    // Uncertain road: nothing is drawn.
    std::vector<int> none = rasterize_boundary(straight_road(0.0, 3.6, 1.0), 0.5, cam, rows, opt);
    CHECK(std::count(none.begin(), none.end(), kNoLanePoint) == static_cast<std::ptrdiff_t>(none.size()));
}

TEST(rasterizer_raw_output_undistorts_back_onto_the_lane) {
    CameraModel cam = test_camera(0.06, 0.01, 1.5);
    RoadState road = straight_road(0.3, 3.7, 1e-4);
    road.x[2] = 0.0008;
    std::vector<int> rows = rows_240_710();
    RasterizeOptions opt{0.0, 40.0, 0.4};
    // No distortion: identical to the undistorted rasterizer.
    CHECK(rasterize_boundary_raw(road, 0.5, cam, rows, opt) == rasterize_boundary(road, 0.5, cam, rows, opt));

    cam.dist = cv::Mat(cv::Matx<double, 1, 5>(-0.25, 0.08, 0.0, 0.0, 0.0), true);
    std::vector<int> xs = rasterize_boundary_raw(road, 0.5, cam, rows, opt);
    int checked = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (xs[i] < 0) continue;
        std::vector<cv::Point2d> raw = {{static_cast<double>(xs[i]), static_cast<double>(rows[i])}}, und;
        cv::undistortPoints(raw, und, cam.K, cam.dist, cv::noArray(), cam.K);
        double X, Y;
        CHECK(cam.image_to_ground(und[0], X, Y));
        // 0.5 px of rounding is ~X / f metres sideways.
        CHECK_NEAR(Y, road.lateral(X, 0.5), 0.02 + X * 0.6 / cam.K(0, 0));
        ++checked;
    }
    CHECK(checked > 15);
}

TEST(predicted_lanes_skip_disabled_and_empty_slots) {
    CameraModel cam = test_camera(0.06, 0.0, 1.5);
    RoadState road = straight_road(0.0, 3.7, 1e-4);
    std::vector<int> rows = rows_240_710();
    RasterizeOptions opt{0.0, 40.0, 0.4};
    bool inner[kLaneSlots] = {true, true, false, false};
    bool all[kLaneSlots] = {true, true, true, true};
    CHECK(predict_tusimple_lanes(road, cam, rows, opt, inner, 2).size() == 2);
    CHECK(predict_tusimple_lanes(road, cam, rows, opt, all, 2).size() == 4);
    road.valid = false;
    CHECK(predict_tusimple_lanes(road, cam, rows, opt, all, 2).empty());
}

// ---- geometry against labels ----

// Labels a perfect camera would produce for a straight road.
static std::vector<TuSimpleLabel> synthetic_labels(const CameraModel& truth, int n) {
    std::vector<TuSimpleLabel> out;
    RasterizeOptions opt{0.0, 200.0, 0.4};
    for (int i = 0; i < n; ++i) {
        TuSimpleLabel l;
        l.raw_file = "clips/x/" + std::to_string(i) + "/20.jpg";
        l.h_samples = rows_240_710();
        RoadState road = straight_road(-0.6 + 1.2 * i / n, 3.7, 1e-6);
        for (LaneSlot s : {LaneSlot::LeftOuter, LaneSlot::Left, LaneSlot::Right, LaneSlot::RightOuter}) {
            l.lanes.push_back(rasterize_boundary(road, slot_width_multiple(s), truth, l.h_samples, opt));
        }
        out.push_back(l);
    }
    return out;
}

TEST(label_back_projection_recovers_lateral_offset) {
    CameraModel cam = test_camera(0.06, 0.01, 1.6);
    std::vector<TuSimpleLabel> labels = synthetic_labels(cam, 1);
    double Y;
    CHECK(label_lateral_at(labels[0].lanes[1], labels[0].h_samples, cam, 10.0, Y));
    CHECK_NEAR(Y, -0.6 + 1.85, 0.03);
    CHECK(label_lateral_at(labels[0].lanes[2], labels[0].h_samples, cam, 20.0, Y));
    CHECK_NEAR(Y, -0.6 - 1.85, 0.06);
    CHECK(!label_lateral_at(labels[0].lanes[1], labels[0].h_samples, cam, 1.0, Y));  // below the image
}

TEST(mount_fit_recovers_pitch_yaw_and_height_from_labels) {
    CameraModel truth = test_camera(0.065, 0.012, 1.7);
    std::vector<TuSimpleLabel> labels = synthetic_labels(truth, 40);
    CameraModel cam = test_camera(0.0, 0.0, 1.4);
    MountFit fit;
    std::string e;
    CHECK(fit_mount_from_labels(labels, 300, 3.7, cam, fit, e));
    CHECK(fit.images_used == 40);
    CHECK_NEAR(cam.pitch_rad, truth.pitch_rad, 0.002);
    CHECK_NEAR(cam.yaw_rad, truth.yaw_rad, 0.002);
    CHECK_NEAR(cam.height_m, truth.height_m, 0.05);
    CHECK(cam.mount_calibrated);
}

// ---- session labels ----

TEST(session_records_and_labels_parse_and_summarise) {
    SessionRecord a, b, c;
    std::string e;
    CHECK(parse_session_record(
        "{\"schema_version\":1,\"frame_id\":0,\"media_time_s\":0,\"image\":\"images/00000000.jpg\","
        "\"reasons\":[\"keyframe\",\"status_change\"],\"labels\":{\"human\":null},\"status\":\"PARTIAL\","
        "\"confidence\":0.167788,\"road_state\":{\"valid\":true,\"x\":[-0.34,0.0075,-0.00074905,3.84],"
        "\"P\":[0.21,-0.00055,1.22169e-05,0.39]},\"measurements\":[{\"mahalanobis2\":null}]}",
        a, e));
    CHECK(a.frame_id == 0 && a.status == "PARTIAL" && a.human.empty() && a.road_valid);
    CHECK_NEAR(a.confidence, 0.167788, 1e-9);
    CHECK(a.reasons.size() == 2 && a.reasons[1] == "status_change");
    CHECK(parse_session_record("{\"frame_id\":7,\"reasons\":[\"human_label\"],\"labels\":{\"human\":\"good\"},"
                               "\"status\":\"LOCKED\",\"confidence\":0.9,\"road_state\":{\"valid\":true}}",
                               b, e));
    CHECK(b.human == "good");
    CHECK(parse_session_record("{\"frame_id\":9,\"reasons\":[\"keyframe\"],\"labels\":{\"human\":null},"
                               "\"status\":\"LOCKED\",\"confidence\":0.8,\"road_state\":{\"valid\":true}}",
                               c, e));
    long long id = -1;
    std::string human;
    CHECK(parse_label_line("{\"frame_id\":0,\"human\":\"bad\"}", id, human, e));
    CHECK(id == 0 && human == "bad");
    CHECK(!parse_label_line("{\"frame_id\":0}", id, human, e));

    // labels.jsonl: frame 0 bad, frame 7 relabelled bad then good, 42 unknown.
    std::map<long long, std::string> labels = {{0, "bad"}, {7, "good"}, {42, "good"}};
    SessionReport r = summarize_session({a, b, c}, labels);
    CHECK(r.records == 3 && r.good == 1 && r.bad == 1);
    CHECK(r.labels_without_record == 1);
    CHECK(r.by_status["LOCKED"].records == 2 && r.by_status["LOCKED"].good == 1);
    CHECK(r.by_status["PARTIAL"].bad == 1);
    CHECK(r.by_reason["keyframe"].records == 2 && r.by_reason["keyframe"].bad == 1);
    CHECK_NEAR(r.auc, 1.0, 1e-12);
    CHECK_NEAR(r.brier, 0.5 * (0.1 * 0.1 + 0.167788 * 0.167788), 1e-9);
    CHECK(r.by_confidence[4].good == 1 && r.by_confidence[0].bad == 1);
}
