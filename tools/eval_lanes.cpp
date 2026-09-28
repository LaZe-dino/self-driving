// atlas_eval: lane-detection accuracy against human labels.
//   TuSimple benchmark (official metric) and human good/bad review labels of
//   logged sessions. See docs/EVALUATION.md.
#include "session_eval.hpp"
#include "tusimple_eval.hpp"
#include "tusimple_score.hpp"
#include <iostream>
#include <string>

namespace {

void usage() {
    std::cout <<
        "atlas_eval: lane accuracy against human labels (docs/EVALUATION.md)\n"
        "  atlas_eval --tusimple ROOT --labels test_label.json [--labels more.json ...]\n"
        "     ROOT is the folder containing clips/; --labels paths are used as given\n"
        "     --camera-model FILE.yml   intrinsics (+ mount unless --fit-mount)\n"
        "     --fov DEG                 assumed horizontal FOV without a model (default 60)\n"
        "     --fit-mount               fit pitch/yaw/height to the labelled lanes (default without\n"
        "                               --camera-model); --no-fit-mount keeps a level 1.4 m mount\n"
        "     --fit-labels FILE         labels for the mount fit (default: --labels); repeatable\n"
        "     --fit-images N            labelled images used by the fit (default 300)\n"
        "     --lane-width M            real lane width for the height fit (default 3.7)\n"
        "     --save-camera FILE.yml    write the camera model used\n"
        "     --limit N                 evaluate the first N labelled images\n"
        "     --near-m M --far-m M      drawn range ahead of the car (default 0 .. 40)\n"
        "     --all-slots               always predict outer lanes (default: only if seen in the last 1 s)\n"
        "     --objects [--object-model FILE.onnx]   mask vehicles (default off)\n"
        "     --out results.json        summary + per-image results\n"
        "     --predictions pred.json   predictions in the official submission format\n"
        "  atlas_eval --score PRED.json --labels test_label.json [--labels more.json ...] [--out results.json]\n"
        "     score an external prediction file (TuSimple format, e.g. training/predict.py) by raw_file\n"
        "  atlas_eval --session data/sessions/DIR [--out results.json]\n";
}

}  // namespace

int main(int argc, char** argv) {
    TuSimpleEvalOptions o;
    std::string session_dir;
    std::string score_path;
    std::string mode;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        std::string v;
        auto value = [&]() -> bool {
            if (i + 1 >= argc) {
                std::cerr << a << " needs a value\n";
                return false;
            }
            v = argv[++i];
            return true;
        };
        auto number = [&](double& dst) -> bool {
            if (!value()) return false;
            try {
                dst = std::stod(v);
            } catch (const std::exception&) {
                std::cerr << a << ": not a number: " << v << "\n";
                return false;
            }
            return true;
        };
        double x = 0;
        bool ok = true;
        if (a == "--tusimple") {
            mode = "tusimple";
            ok = value();
            o.root = v;
        } else if (a == "--session") {
            mode = "session";
            ok = value();
            session_dir = v;
        } else if (a == "--score") {
            mode = "score";
            ok = value();
            score_path = v;
        } else if (a == "--labels") {
            ok = value();
            o.label_files.push_back(v);
        } else if (a == "--fit-labels") {
            ok = value();
            o.fit_label_files.push_back(v);
        } else if (a == "--camera-model") {
            ok = value();
            o.camera_model_path = v;
        } else if (a == "--fov") {
            ok = number(o.fov_deg);
        } else if (a == "--fit-mount") {
            o.fit_mount = true;
        } else if (a == "--no-fit-mount") {
            o.no_fit_mount = true;
        } else if (a == "--fit-images") {
            ok = number(x);
            o.fit_images = static_cast<int>(x);
        } else if (a == "--lane-width") {
            ok = number(o.lane_width_m);
        } else if (a == "--save-camera") {
            ok = value();
            o.save_camera_path = v;
        } else if (a == "--limit") {
            ok = number(x);
            o.limit = static_cast<long long>(x);
        } else if (a == "--near-m") {
            ok = number(o.raster.x_min);
        } else if (a == "--far-m") {
            ok = number(o.raster.x_max);
        } else if (a == "--all-slots") {
            o.all_slots = true;
        } else if (a == "--objects") {
            o.objects = true;
        } else if (a == "--object-model") {
            ok = value();
            o.object_params.model_path = v;
            o.objects = true;
        } else if (a == "--out") {
            ok = value();
            o.out_json = v;
        } else if (a == "--predictions") {
            ok = value();
            o.predictions_path = v;
        } else if (a == "--help" || a == "-h") {
            usage();
            return 0;
        } else {
            std::cerr << "unknown argument: " << a << "\n";
            ok = false;
        }
        if (!ok) {
            usage();
            return 1;
        }
    }
    if (mode == "session") return run_session_eval(session_dir, o.out_json);
    if (mode == "score") {
        if (o.label_files.empty()) {
            std::cerr << "--score needs --labels FILE.json\n";
            return 1;
        }
        return run_tusimple_score(score_path, o.label_files, o.out_json);
    }
    if (mode == "tusimple") {
        if (o.label_files.empty()) {
            std::cerr << "--tusimple needs --labels FILE.json\n";
            return 1;
        }
        if (o.fit_mount && o.no_fit_mount) {
            std::cerr << "--fit-mount and --no-fit-mount are exclusive\n";
            return 1;
        }
        return run_tusimple_eval(o);
    }
    usage();
    return 1;
}
